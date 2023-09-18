"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import torch


def move_to_device(model):
    if torch.cuda.is_available():
        return model.to("cuda").to(torch.bfloat16)
    try:
        if torch.backends.mps.is_available():
            return model.to("mps")
    except AttributeError:
        pass
    return model.to(torch.bfloat16)
