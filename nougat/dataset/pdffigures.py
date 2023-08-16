"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import os
import subprocess
import logging

PDFFIGURES2_JAR_PATH = os.environ.get("PDFFIGURES_PATH", None)
logger = logging.getLogger()
if PDFFIGURES2_JAR_PATH is None:
    logger.warning(
        "You need to configure the path to the pdffigures2 executable in this file (nougat/dataset/pdffigures.py) or set the environment variable 'PDFFIGURES_PATH'."
    )


def call_pdffigures(
    pdf_path: str, figures_dir: str, timeout: int = 30, verbose: bool = False
):
    os.makedirs(figures_dir, exist_ok=True)
    kwargs = (
        {} if verbose else {"stderr": subprocess.DEVNULL, "stdout": subprocess.DEVNULL}
    )
    if PDFFIGURES2_JAR_PATH is None:
        return
    process = subprocess.Popen(
        "java"
        " -jar {pdffigures_jar_path}"
        " -d {figures_dir}/"
        " -c"
        " -q"
        " {pdf_path}".format(
            pdffigures_jar_path=PDFFIGURES2_JAR_PATH,
            pdf_path=pdf_path,
            figures_dir=figures_dir,
        ),
        shell=True,
        **kwargs
    )

    try:
        exit_code = process.wait(timeout=timeout)
        if exit_code != 0:
            logger.error("Extracting figures from file %s failed.", pdf_path)
            return False
    except subprocess.TimeoutExpired as e:
        logger.error(
            "pdffigures2 command did not terminate in 30 seconds, "
            "terminating. Error: %s",
            e,
        )
        process.terminate()  # give up
        return False
    pdf_name = os.path.basename(pdf_path).partition(".pdf")[0]
    dest_file = os.path.join(figures_dir, (pdf_name + ".json"))

    return dest_file
