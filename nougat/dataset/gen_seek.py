"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
from tqdm import tqdm
import json
from pathlib import Path
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("src_file", nargs="+", type=Path, help="JSONL file in question")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    for file in args.src_file:
        seek_map = []
        seek_pos = 0
        with open(file) as f:
            with tqdm(smoothing=0.0) as pbar:
                line = f.readline()
                while line:
                    seek_map.append(seek_pos)
                    seek_pos = f.tell()
                    line = f.readline()
                    pbar.update(1)

        out_file = file.parent / (file.stem + ".seek.map")
        with open(out_file, "w") as f:
            f.write(json.dumps(seek_map))
