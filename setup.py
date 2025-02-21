"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import os
from setuptools import find_packages, setup

ROOT = os.path.abspath(os.path.dirname(__file__))


def read_version():
    data = {}
    path = os.path.join(ROOT, "nougat", "_version.py")
    with open(path, "r", encoding="utf-8") as f:
        exec(f.read(), data)
    return data["__version__"]


def read_long_description():
    path = os.path.join(ROOT, "README.md")
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return text


setup(
    name="nougat-ocr",
    version=read_version(),
    description="Nougat: Neural Optical Understanding for Academic Documents",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    author="Lukas Blecher",
    author_email="lblecher@meta.com",
    url="https://github.com/facebookresearch/nougat",
    license="MIT",
    packages=find_packages(
        exclude=[
            "result",
        ]
    ),
    py_modules=["predict", "app", "train", "test"],
    python_requires=">=3.7",
    install_requires=[
        "transformers>=4.25.1,<=4.38.2",
        "timm==0.5.4",
        "orjson",
        "opencv-python-headless",
        "datasets[vision]",
        "lightning>=2.0.0,<2022",
        "nltk",
        "rapidfuzz",
        "sentencepiece",
        "sconf>=0.2.3",
        "albumentations>=1.0.0,<=1.4.24",
        "pypdf>=3.1.0",
        "pypdfium2",
    ],
    extras_require={
        "api": [
            "fastapi",
            "uvicorn[standard]",
            "python-multipart",
        ],
        "dataset": [
            "pytesseract",
            "beautifulsoup4",
            "scikit-learn",
            "Pebble",
            "pylatexenc",
            "fuzzysearch",
            "unidecode",
            "htmlmin",
            "pdfminer.six>=20221105",
        ],
    },
    entry_points={
        "console_scripts": [
            "nougat = predict:main",
            "nougat_api = app:main",
        ],
    },
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
