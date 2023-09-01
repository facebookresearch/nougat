"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
from typing import Optional
import requests
import os
import tqdm
import io
from pathlib import Path
import torch

BASE_URL = "https://github.com/facebookresearch/nougat/releases/download"
MODEL_TAG = "0.1.0-small"


# source: https://stackoverflow.com/a/71459251
def download_as_bytes_with_progress(url: str, name: str = None) -> bytes:
    """
    Download a file from a URL and return the contents as bytes, with progress bar.

    Args:
        url: The URL of the file to download.
        name: The name of the file to save to. If None, the filename will be the same as the URL.

    Returns:
        bytes: The contents of the file.
    """
    resp = requests.get(url, stream=True, allow_redirects=True)
    total = int(resp.headers.get("content-length", 0))
    bio = io.BytesIO()
    if name is None:
        name = url
    with tqdm.tqdm(
        desc=name,
        total=total,
        unit="b",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in resp.iter_content(chunk_size=65536):
            bar.update(len(chunk))
            bio.write(chunk)
    return bio.getvalue()


def download_checkpoint(checkpoint: Path):
    print("downloading nougat checkpoint version", MODEL_TAG, "to path", checkpoint)
    files = [
        "config.json",
        "pytorch_model.bin",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    for file in files:
        download_url = f"{BASE_URL}/{MODEL_TAG}/{file}"
        binary_file = download_as_bytes_with_progress(download_url, file)
        if len(binary_file) > 15:  # sanity check
            (checkpoint / file).write_bytes(binary_file)


def get_checkpoint(
    checkpoint_path: Optional[os.PathLike] = None, download: bool = True
) -> Path:
    """
    Get the path to the Nougat model checkpoint.

    This function retrieves the path to the Nougat model checkpoint. If the checkpoint does not
    exist or is empty, it can optionally download the checkpoint.

    Args:
        checkpoint_path (Optional[os.PathLike]): The path to the checkpoint. If not provided,
            it will check the "NOUGAT_CHECKPOINT" environment variable or use the default location.
        download (bool): Whether to download the checkpoint if it doesn't exist or is empty.
            Default is True.

    Returns:
        Path: The path to the Nougat model checkpoint.
    """
    checkpoint = Path(
        checkpoint_path
        or os.environ.get("NOUGAT_CHECKPOINT", torch.hub.get_dir() + "/nougat")
    )
    if checkpoint.exists() and checkpoint.is_file():
        checkpoint = checkpoint.parent
    if download and (not checkpoint.exists() or len(os.listdir(checkpoint)) < 5):
        checkpoint.mkdir(parents=True, exist_ok=True)
        download_checkpoint(checkpoint)
    return checkpoint


if __name__ == "__main__":
    get_checkpoint()
