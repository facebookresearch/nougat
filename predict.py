"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import sys
from pathlib import Path
import logging
import re
import argparse
import re
from functools import partial
import torch
from torch.utils.data import ConcatDataset
from tqdm import tqdm
from nougat import NougatModel
from nougat.utils.dataset import LazyDataset
from nougat.utils.device import move_to_device
from nougat.utils.checkpoint import get_checkpoint
from nougat.postprocessing import markdown_compatible
import pypdf

logging.basicConfig(level=logging.INFO)

if torch.cuda.is_available():
    BATCH_SIZE = int(
        torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1000 * 0.3
    )
    if BATCH_SIZE == 0:
        logging.warning("GPU VRAM is too small. Computing on CPU.")
elif torch.backends.mps.is_available():
    # I don't know if there's an equivalent API so heuristically choosing bs=4
    BATCH_SIZE = 4
else:
    # don't know what a good value is here. Would not recommend to run on CPU
    BATCH_SIZE = 1
    logging.warning("No GPU found. Conversion on CPU is very slow.")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batchsize",
        "-b",
        type=int,
        default=BATCH_SIZE,
        help="Batch size to use.",
    )
    parser.add_argument(
        "--checkpoint",
        "-c",
        type=Path,
        default=None,
        help="Path to checkpoint directory.",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default='0.1.0-small',
        help=f"Model tag to use.",
    )
    parser.add_argument("--out", "-o", type=Path, help="Output directory.")
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Recompute already computed PDF, discarding previous predictions.",
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Add postprocessing step for markdown compatibility.",
    )
    parser.add_argument(
        "--no-skipping",
        dest="skipping",
        action="store_false",
        help="Don't apply failure detection heuristic.",
    )
    parser.add_argument("pdf", nargs="+", type=Path, help="PDF(s) to process.")
    args = parser.parse_args()
    if args.checkpoint is None or not args.checkpoint.exists():
        args.checkpoint = get_checkpoint(args.checkpoint, model_tag=args.model)
    if args.out is None:
        logging.warning("No output directory. Output will be printed to console.")
    else:
        if not args.out.exists():
            logging.info("Output directory does not exist. Creating output directory.")
            args.out.mkdir(parents=True)
        if not args.out.is_dir():
            logging.error("Output has to be directory.")
            sys.exit(1)
    if len(args.pdf) == 1 and not args.pdf[0].suffix == ".pdf":
        # input is a list of pdfs
        try:
            args.pdf = [
                Path(l) for l in open(args.pdf[0]).read().split("\n") if len(l) > 0
            ]
        except:
            pass
    return args


def main():
    args = get_args()
    model = NougatModel.from_pretrained(args.checkpoint)
    if args.batchsize > 0:
        model = move_to_device(model)
    else:
        # set batch size to 1. Need to check if there are benefits for CPU conversion for >1
        args.batchsize = 1
    model.eval()
    datasets = []
    for pdf in args.pdf:
        if not pdf.exists():
            continue
        if args.out:
            out_path = args.out / pdf.with_suffix(".mmd").name
            if out_path.exists() and not args.recompute:
                logging.info(
                    f"Skipping {pdf.name}, already computed. Run with --recompute to convert again."
                )
                continue
        try:
            dataset = LazyDataset(
                pdf, partial(model.encoder.prepare_input, random_padding=False)
            )
        except pypdf.errors.PdfStreamError:
            logging.info(f"Could not load file {str(pdf)}.")
            continue
        datasets.append(dataset)
    if len(datasets) == 0:
        return
    dataloader = torch.utils.data.DataLoader(
        ConcatDataset(datasets),
        batch_size=args.batchsize,
        shuffle=False,
        collate_fn=LazyDataset.ignore_none_collate,
    )

    predictions = []
    file_index = 0
    page_num = 0
    for i, (sample, is_last_page) in enumerate(tqdm(dataloader)):
        model_output = model.inference(image_tensors=sample, early_stopping=args.skipping)
        # check if model output is faulty
        for j, output in enumerate(model_output["predictions"]):
            if page_num == 0:
                logging.info(
                    "Processing file %s with %i pages"
                    % (datasets[file_index].name, datasets[file_index].size)
                )
            page_num += 1
            if output.strip() == "[MISSING_PAGE_POST]":
                # uncaught repetitions -- most likely empty page
                predictions.append(f"\n\n[MISSING_PAGE_EMPTY:{page_num}]\n\n")
            elif args.skipping and model_output["repeats"][j] is not None:
                if model_output["repeats"][j] > 0:
                    # If we end up here, it means the output is most likely not complete and was truncated.
                    logging.warning(f"Skipping page {page_num} due to repetitions.")
                    predictions.append(f"\n\n[MISSING_PAGE_FAIL:{page_num}]\n\n")
                else:
                    # If we end up here, it means the document page is too different from the training domain.
                    # This can happen e.g. for cover pages.
                    predictions.append(
                        f"\n\n[MISSING_PAGE_EMPTY:{i*args.batchsize+j+1}]\n\n"
                    )
            else:
                if args.markdown:
                    output = markdown_compatible(output)
                predictions.append(output)
            if is_last_page[j]:
                out = "".join(predictions).strip()
                out = re.sub(r"\n{3,}", "\n\n", out).strip()
                if args.out:
                    out_path = args.out / Path(is_last_page[j]).with_suffix(".mmd").name
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    out_path.write_text(out, encoding="utf-8")
                else:
                    print(out, "\n\n")
                predictions = []
                page_num = 0
                file_index += 1


if __name__ == "__main__":
    main()
