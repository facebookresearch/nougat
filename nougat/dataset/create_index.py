"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
"""
This script creates an index of all available pages and parses the meta data for all pages into a separate file.
Optionally TesseractOCR is called for each image.
"""
import argparse
import json
from typing import Dict, List
import numpy as np
from pathlib import Path
import multiprocessing
from pebble import ProcessPool
from PIL import Image
import pytesseract
import re
import logging
from tqdm import tqdm


logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def convert_pt2px(pt, dpi=96):
    if isinstance(pt, list):
        return [round(dpi / 72 * p) for p in pt]
    elif isinstance(pt, dict):
        for k in pt:
            pt[k] = round(dpi / 72 * pt[k])
        return pt


def read_metadata(data: Dict) -> List[List[Dict]]:
    N = data["num_pages"]
    out = [[] for _ in range(N)]
    # pdffigures2 meta data
    if "pdffigures" in data and data["pdffigures"]:
        for item in data["pdffigures"]:
            p = item.pop("page", None)
            if p is None or p >= N:
                continue
            item["source"] = "fig"
            if "regionBoundary" in item:
                item["regionBoundary"] = convert_pt2px(item["regionBoundary"])
            if "captionBoundary" in item:
                item["captionBoundary"] = convert_pt2px(item["captionBoundary"])
            out[p].append(item)

    return out


def index_paper(directory: Path, args: argparse.Namespace):
    """
    Pack all image-text pairs into a single h5 file and save it at `args.out`
    """
    paper = directory.name
    markdowns = directory.glob("*.mmd")
    meta_file = directory / "meta.json"
    data_samples = []
    if not meta_file.exists():
        return
    # load meta info
    try:
        meta = read_metadata(json.load(meta_file.open("r", encoding="utf-8")))
    except json.JSONDecodeError:
        return

    for md_path in markdowns:
        image = md_path.parent / (md_path.stem + ".png")
        i = int(image.stem) - 1
        if not image.exists():
            continue
        if i >= len(meta):
            continue
        data_sample = {}
        ocr_path = image.parent / (image.stem + "_OCR.txt")
        if args.tesseract and not ocr_path.exists():
            try:
                pil = Image.open(image)
                ocr = pytesseract.image_to_string(pil, lang="eng", timeout=2)
                ocr = re.sub(r"\n+\s+?([^\s])", r"\n\n\1", ocr).strip()
                with ocr_path.open("w", encoding="utf-8") as f_ocr:
                    f_ocr.write(ocr)
            except RuntimeError:
                logger.info("Page %s of paper %s timed out", image.stem, paper)
                pass
        if ocr_path.exists():
            data_sample["ocr"] = str(ocr_path.relative_to(args.root))
        data_sample["image"] = str(image.relative_to(args.root))
        data_sample["markdown"] = md_path.read_text(encoding="utf8").strip()
        data_sample["meta"] = meta[i]
        data_samples.append(data_sample)
    return data_samples


def create_index(args):
    if not args.dir.exists() and not args.dir.is_dir():
        logger.error("%s does not exist or is no dir.", args.dir)
        return
    papers = []
    depth = 0
    p = args.dir
    while True:
        p = next(p.iterdir())
        if p.is_file():
            break
        else:
            depth += 1
    papers = args.dir.glob("*/" * depth)
    index = []
    with ProcessPool(max_workers=args.workers) as pool:
        tasks = {}
        for j, paper in enumerate(papers):
            fname = paper.name
            tasks[fname] = pool.schedule(
                index_paper,
                args=[paper, args],
                timeout=args.timeout,
            )

        for fname in tqdm(tasks):
            try:
                res = tasks[fname].result()
                if res is None:
                    logger.info("%s is faulty", fname)
                    continue
                index.append(res)
            except TimeoutError:
                logger.info("%s timed out", fname)

        with args.out.open("w", encoding="utf-8") as f:
            for item in index:
                for page in item:
                    if len(page) == 0:
                        continue
                    f.write(json.dumps(page) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True, help="Index file")
    parser.add_argument(
        "--dir", type=Path, required=True, help="Parent directory for input dirs"
    )
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument(
        "--tesseract",
        action="store_true",
        help="Tesseract OCR prediction for each page",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=multiprocessing.cpu_count(),
        help="How many processes to use",
    )
    parser.add_argument(
        "--dpi", type=int, default=96, help="DPI the images were saved with"
    )
    parser.add_argument("--timeout", type=int, default=240, help="Max time per paper")
    args = parser.parse_args()
    if args.root is None:
        args.root = args.dir
    else:
        # check if dir is subdir of root
        args.dir.relative_to(args.root)
    create_index(args)
