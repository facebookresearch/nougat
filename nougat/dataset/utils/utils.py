"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import re


def remove_pretty_linebreaks(string: str) -> str:
    """replaces linebreaks with spaces when there would be no
    difference between them in the markdown format

    Args:
        string (str): String to process

    Returns:
        str: Formatted string
    """
    return re.sub(r"(?<!\n)\n([^\n\d\*#\[])", r" \1", string).strip()
