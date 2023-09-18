"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import argparse
from io import BytesIO
import multiprocessing
from pebble import ProcessPool
from concurrent.futures import TimeoutError
from tqdm import tqdm
from typing import Tuple
import os
from pathlib import Path
import logging
import pypdf
from PIL import Image
import pytesseract
from nougat.dataset.split_md_to_pages import *
from nougat.dataset.parser.html2md import *
from nougat.dataset.pdffigures import call_pdffigures

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def process_paper(
    fname: str,
    pdf_file: Path,
    html_file: Path,
    json_file: Path,
    args: argparse.Namespace,
) -> Tuple[int, int]:
    """
    Process a single paper.

    Args:
        fname (str): The paper's filename.
        pdf_file (Path): The path to the PDF file.
        html_file (Path): The path to the HTML file.
        json_file (Path): The path to the JSON file containing the extracted figures.
        args (argparse.Namespace): The command-line arguments.

    Returns:
        Tuple[int, int]: The number of total pages and the number of recognized pages.
    """
    total_pages = 0
    num_recognized_pages = 0
    try:
        pdf = pypdf.PdfReader(pdf_file)
        total_pages = len(pdf.pages)
        outpath: Path = args.out / fname
        # skip this paper if already processed
        dirs_with_same_stem = list(args.out.glob(fname.partition("v")[0] + "*"))
        if (
            len(dirs_with_same_stem) > 0
            and len(list(dirs_with_same_stem[0].iterdir())) > 0
            and not args.recompute
        ):
            logger.info(
                "%s (or another version thereof) already processed. Skipping paper",
                fname,
            )
            return total_pages, len(list(outpath.glob("*.mmd")))
        html = BeautifulSoup(
            htmlmin.minify(
                open(html_file, "r", encoding="utf-8").read().replace("\xa0", " "),
                remove_all_empty_space=True,
            ),
            features="html.parser",
        )
        doc = parse_latexml(html)
        if doc is None:
            return
        out, fig = format_document(doc, keep_refs=True)

        if args.markdown:
            md_out = args.markdown / (fname + ".mmd")
            with open(md_out, "w", encoding="utf-8") as f:
                f.write(out)

        if json_file is None:
            json_file = call_pdffigures(pdf_file, args.figure)
        if json_file:
            figure_info = json.load(open(json_file, "r", encoding="utf-8"))
        else:
            figure_info = None
        split = split_markdown(
            out, pdf_file, figure_info=figure_info, doc_fig=fig, min_score=0.9
        )
        if split is None:
            return
        pages, meta = split
        num_recognized_pages = sum([len(p) > 0 for p in pages])
        if all([len(p) == 0 for p in pages]):
            return
        os.makedirs(outpath, exist_ok=True)
        recognized_indices = []
        for i, content in enumerate(pages):
            with (outpath / "meta.json").open("w", encoding="utf-8") as f:
                f.write(json.dumps(meta))
            if content:
                if re.search(r"\[(?:\?\?(?:. )?)+\]", content):
                    # there are wrongly parsed references in the page eg [??].
                    continue
                with (outpath / ("%02d.mmd" % (i + 1))).open(
                    "w", encoding="utf-8"
                ) as f:
                    f.write(content)
                recognized_indices.append(i)
        rasterize_paper(pdf_file, outpath, dpi=args.dpi, pages=recognized_indices)
        if args.tesseract:
            for i in recognized_indices:
                ocr = pytesseract.image_to_string(
                    Image.open((outpath / ("%02d.png" % (i + 1)))), lang="eng"
                )
                ocr = re.sub(r"\n+\s+?([^\s])", r"\n\n\1", ocr).strip()
                with (outpath / ("%02d_OCR.txt" % (i + 1))).open(
                    "w", encoding="utf-8"
                ) as f_ocr:
                    f_ocr.write(ocr)
    except Exception as e:
        logger.error(e)

    return total_pages, num_recognized_pages


def process_htmls(args):
    for input_dir in (args.pdfs, args.html):
        if not input_dir.exists() and not input_dir.is_dir():
            logger.error("%s does not exist or is no dir.", input_dir)
            return
    htmls: List[Path] = args.html.glob("*.html")
    args.out.mkdir(exist_ok=True)
    if args.markdown:
        args.markdown.mkdir(exist_ok=True)

    with ProcessPool(max_workers=args.workers) as pool:
        total_pages, total_pages_extracted = 0, 0
        tasks = {}
        for j, html_file in enumerate(htmls):
            fname = html_file.stem
            pdf_file = args.pdfs / (fname + ".pdf")
            if not pdf_file.exists():
                logger.info("%s pdf could not be found.", fname)
                continue
            json_file = args.figure / (fname + ".json")
            if not json_file.exists():
                logger.info("%s figure json could not be found.", fname)
                json_file = None
            tasks[fname] = pool.schedule(
                process_paper,
                args=[fname, pdf_file, html_file, json_file, args],
                timeout=args.timeout,
            )

        for fname in tqdm(tasks):
            try:
                res = tasks[fname].result()
                if res is None:
                    logger.info("%s is faulty", fname)
                    continue
                num_pages, num_recognized_pages = res
                total_pages += num_pages
                total_pages_extracted += num_recognized_pages
                logger.info(
                    "%s: %i/%i pages recognized. Percentage: %.2f%%",
                    fname,
                    num_recognized_pages,
                    num_pages,
                    (100 * num_recognized_pages / max(1, num_pages)),
                )
            except TimeoutError:
                logger.info("%s timed out", fname)
    if total_pages > 0:
        logger.info(
            "In total: %i/%i pages recognized. Percentage: %.2f%%",
            total_pages_extracted,
            total_pages,
            (100 * total_pages_extracted / max(1, total_pages)),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--html", type=Path, help="HTML files", required=True)
    parser.add_argument("--pdfs", type=Path, help="PDF files", required=True)
    parser.add_argument("--out", type=Path, help="Output dir", required=True)
    parser.add_argument("--recompute", action="store_true", help="recompute all splits")
    parser.add_argument(
        "--markdown", type=Path, help="Markdown output dir", default=None
    )
    parser.add_argument(
        "--figure",
        type=Path,
        help="Figure info JSON dir",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=multiprocessing.cpu_count(),
        help="How many processes to use",
    )
    parser.add_argument(
        "--dpi", type=int, default=96, help="What resolution the pages will be saved at"
    )
    parser.add_argument(
        "--timeout", type=float, default=120, help="max time per paper in seconds"
    )
    parser.add_argument(
        "--tesseract",
        action="store_true",
        help="Tesseract OCR prediction for each page",
    )
    args = parser.parse_args()
    print(args)
    process_htmls(args)
