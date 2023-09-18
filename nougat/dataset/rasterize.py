"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import argparse
import pdf2image
import pypdf
from pathlib import Path
from tqdm import tqdm
import io
from typing import Optional, List


def rasterize_paper(
    pdf: Path,
    outpath: Optional[Path] = None,
    dpi: int = 96,
    return_pil=False,
    pages=None,
) -> Optional[List[io.BytesIO]]:
    """
    Rasterize a PDF file to PNG images.

    Args:
        pdf (Path): The path to the PDF file.
        outpath (Optional[Path], optional): The output directory. If None, the PIL images will be returned instead. Defaults to None.
        dpi (int, optional): The output DPI. Defaults to 96.
        return_pil (bool, optional): Whether to return the PIL images instead of writing them to disk. Defaults to False.
        pages (Optional[List[int]], optional): The pages to rasterize. If None, all pages will be rasterized. Defaults to None.

    Returns:
        Optional[List[io.BytesIO]]: The PIL images if `return_pil` is True, otherwise None.
    """
    pils = []
    if outpath is None:
        return_pil = True
    try:
        if isinstance(pdf, (str, Path)):
            pdf = pypdf.PdfReader(pdf)
        if pages is None:
            pages = range(len(pdf.pages))
        for i in pages:
            page_bytes = io.BytesIO()
            writer = pypdf.PdfWriter()
            writer.add_page(pdf.pages[i])
            writer.write(page_bytes)
            page_bytes = page_bytes.getvalue()
            img = pdf2image.convert_from_bytes(
                page_bytes,
                dpi=dpi,
                fmt="ppm" if outpath is None else "png",
                output_folder=None if outpath is None else outpath,
                single_file=True,
                output_file="%02d" % (i + 1),
            )[0]
            if return_pil:
                img_bytes = io.BytesIO()
                img.save(img_bytes, format=img.format)
                pils.append(img_bytes)
    except Exception:
        pass
    if return_pil:
        return pils


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdfs", nargs="+", type=Path, help="PDF files", required=True)
    parser.add_argument("--out", type=Path, help="Output dir", default=None)
    parser.add_argument(
        "--dpi", type=int, default=96, help="What resolution the pages will be saved"
    )
    parser.add_argument(
        "--pages", type=int, nargs="+", default=None, help="list of page numbers"
    )
    args = parser.parse_args()
    if args.pages:
        args.pages = [p - 1 for p in args.pages]
    for pdf_file in tqdm(args.pdfs):
        assert pdf_file.exists() and pdf_file.is_file()
        outpath: Path = args.out or (pdf_file.parent / pdf_file.stem)
        outpath.mkdir(exist_ok=True)
        rasterize_paper(pdf_file, outpath, pages=args.pages, dpi=args.dpi)
