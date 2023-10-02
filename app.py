"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import os
import sys
from functools import partial
from http import HTTPStatus
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from pathlib import Path
import hashlib
from fastapi.middleware.cors import CORSMiddleware
import pypdfium2
import torch
from nougat import NougatModel
from nougat.postprocessing import markdown_compatible, close_envs
from nougat.utils.dataset import ImageDataset
from nougat.utils.checkpoint import get_checkpoint
from nougat.dataset.rasterize import rasterize_paper
from nougat.utils.device import move_to_device, default_batch_size
from tqdm import tqdm


SAVE_DIR = Path("./pdfs")
BATCHSIZE = int(os.environ.get("NOUGAT_BATCHSIZE", default_batch_size()))
NOUGAT_CHECKPOINT = get_checkpoint()
if NOUGAT_CHECKPOINT is None:
    print(
        "Set environment variable 'NOUGAT_CHECKPOINT' with a path to the model checkpoint!"
    )
    sys.exit(1)

app = FastAPI(title="Nougat API")
origins = ["http://localhost", "http://127.0.0.1"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
model = None


@app.on_event("startup")
async def load_model(
    checkpoint: str = NOUGAT_CHECKPOINT,
):
    global model, BATCHSIZE
    if model is None:
        model = NougatModel.from_pretrained(checkpoint)
        model = move_to_device(model, cuda=BATCHSIZE > 0)
        if BATCHSIZE <= 0:
            BATCHSIZE = 1
        model.eval()


@app.get("/")
def root():
    """Health check."""
    response = {
        "status-code": HTTPStatus.OK,
        "data": {},
    }
    return response


@app.post("/predict/")
async def predict(
    file: UploadFile = File(...), start: int = None, stop: int = None
) -> str:
    """
    Perform predictions on a PDF document and return the extracted text in Markdown format.

    Args:
        file (UploadFile): The uploaded PDF file to process.
        start (int, optional): The starting page number for prediction.
        stop (int, optional): The ending page number for prediction.

    Returns:
        str: The extracted text in Markdown format.
    """
    pdfbin = file.file.read()
    pdf = pypdfium2.PdfDocument(pdfbin)
    md5 = hashlib.md5(pdfbin).hexdigest()
    save_path = SAVE_DIR / md5

    if start is not None and stop is not None:
        pages = list(range(start - 1, stop))
    else:
        pages = list(range(len(pdf)))
    predictions = [""] * len(pages)
    dellist = []
    if save_path.exists():
        for computed in (save_path / "pages").glob("*.mmd"):
            try:
                idx = int(computed.stem) - 1
                if idx in pages:
                    i = pages.index(idx)
                    print("skip page", idx + 1)
                    predictions[i] = computed.read_text(encoding="utf-8")
                    dellist.append(idx)
            except Exception as e:
                print(e)
    compute_pages = pages.copy()
    for el in dellist:
        compute_pages.remove(el)
    images = rasterize_paper(pdf, pages=compute_pages)
    global model

    dataset = ImageDataset(
        images,
        partial(model.encoder.prepare_input, random_padding=False),
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCHSIZE,
        pin_memory=True,
        shuffle=False,
    )

    for idx, sample in tqdm(enumerate(dataloader), total=len(dataloader)):
        if sample is None:
            continue
        model_output = model.inference(image_tensors=sample)
        for j, output in enumerate(model_output["predictions"]):
            if model_output["repeats"][j] is not None:
                if model_output["repeats"][j] > 0:
                    disclaimer = "\n\n+++ ==WARNING: Truncated because of repetitions==\n%s\n+++\n\n"
                else:
                    disclaimer = (
                        "\n\n+++ ==ERROR: No output for this page==\n%s\n+++\n\n"
                    )
                rest = close_envs(model_output["repetitions"][j]).strip()
                if len(rest) > 0:
                    disclaimer = disclaimer % rest
                else:
                    disclaimer = ""
            else:
                disclaimer = ""

            predictions[pages.index(compute_pages[idx * BATCHSIZE + j])] = (
                markdown_compatible(output) + disclaimer
            )

    (save_path / "pages").mkdir(parents=True, exist_ok=True)
    pdf.save(save_path / "doc.pdf")
    if len(images) > 0:
        thumb = Image.open(images[0])
        thumb.thumbnail((400, 400))
        thumb.save(save_path / "thumb.jpg")
    for idx, page_num in enumerate(pages):
        (save_path / "pages" / ("%02d.mmd" % (page_num + 1))).write_text(
            predictions[idx], encoding="utf-8"
        )
    final = "".join(predictions).strip()
    (save_path / "doc.mmd").write_text(final, encoding="utf-8")
    return final


def main():
    import uvicorn

    uvicorn.run("app:app", port=8503)


if __name__ == "__main__":
    main()
