<div align="center">
<h1>Nougat: 学术文档神经光学理解模型 (Neural Optical Understanding for Academic Documents)</h1>

[![论文](https://img.shields.io/badge/Paper-arxiv.2308.13418-white)](https://arxiv.org/abs/2308.13418)
[![GitHub](https://img.shields.io/github/license/facebookresearch/nougat)](https://github.com/facebookresearch/nougat)
[![PyPI](https://img.shields.io/pypi/v/nougat-ocr?logo=pypi)](https://pypi.org/project/nougat-ocr)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![代码规范: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Hugging Face 空间](https://img.shields.io/badge/🤗%20Hugging%20Face-Community%20Space-blue)](https://huggingface.co/spaces/ysharma/nougat)

[English](README.md) | [简体中文](README.zh.md)

</div>

本项目为 Nougat 的官方代码仓库。Nougat 是一款专门用于解析学术文档 PDF 的神经光学理解模型，能够精准识别并提取文档中的 LaTeX 数学公式和表格内容。

项目主页：https://facebookresearch.github.io/nougat/

## 安装指南

通过 pip 安装：
```sh
pip install nougat-ocr
```

从源码仓库安装：
```sh
pip install git+https://github.com/facebookresearch/nougat
```

> **Windows 系统注意**：如需使用 GPU 加速，请先确保安装适配的 PyTorch 版本。详细安装步骤请参阅 [PyTorch 官方指南](https://pytorch.org/get-started/locally/)。

如需通过 API 服务调用模型或构建数据集，需安装额外依赖组件：

```sh
pip install "nougat-ocr[api]"
# 或
pip install "nougat-ocr[dataset]"
```

### 对 PDF 文档进行推理预测

#### 命令行接口 (CLI)

处理单个 PDF 文档：

```sh
nougat path/to/file.pdf -o output_directory
```

也可以传入包含 PDF 文件的目录路径，或者每行包含一个 PDF 路径的文本文件：

```sh
nougat path/to/directory -o output_directory
```

命令行参数说明：

```
usage: nougat [-h] [--batchsize BATCHSIZE] [--checkpoint CHECKPOINT] [--model MODEL] [--out OUT]
              [--recompute] [--markdown] [--no-skipping] pdf [pdf ...]

positional arguments:
  pdf                   待处理的 PDF 文件路径。

options:
  -h, --help            显示帮助信息并退出。
  --batchsize BATCHSIZE, -b BATCHSIZE
                        批处理大小（Batch Size）。
  --checkpoint CHECKPOINT, -c CHECKPOINT
                        模型权重目录路径。
  --model MODEL_TAG, -m MODEL_TAG
                        所使用的模型标签版本。
  --out OUT, -o OUT     输出结果存储目录。
  --recompute           重新计算已存在的 PDF，丢弃此前的预测结果。
  --full-precision      使用 float32 代替 bfloat16。在部分 CPU 环境下可提升转换速度。
  --no-markdown         跳过 Markdown 兼容性后处理步骤。
  --markdown            添加 Markdown 兼容性后处理步骤（默认启用）。
  --no-skipping         禁用失败检测启发式规则。
  --pages PAGES, -p PAGES
                        指定处理页码，例如 '1-4,7' 表示处理第 1 至 4 页及第 7 页。仅适用于单个 PDF。
```

默认模型标签为 `0.1.0-small`。如需使用 Base 规模模型，请指定 `-m 0.1.0-base`：

```sh
nougat path/to/file.pdf -o output_directory -m 0.1.0-base
```

在输出目录中，每个 PDF 将保存为一个 `.mmd` 文件。该文件采用轻量级标记语言，大部分兼容 [Mathpix Markdown](https://github.com/Mathpix/mathpix-markdown-it)（采用 LaTeX 格式呈现表格）。

> **说明**：在部分硬件设备上，失败检测启发式规则可能会出现误判。如果遇到频繁返回 `[MISSING_PAGE]` 的情况，可尝试添加 `--no-skipping` 参数运行。相关讨论详见：[#11](https://github.com/facebookresearch/nougat/issues/11)、[#67](https://github.com/facebookresearch/nougat/issues/67)。

#### API 服务

安装额外依赖后，可使用 `app.py` 启动 HTTP API 服务：

```sh
nougat_api
```

启动后向 `http://127.0.0.1:8503/predict/` 发送 POST 请求即可获取 PDF 的解析结果。接口支持通过 `start` 和 `stop` 查询参数限制计算的页码范围（包含边界页）。

接口返回包含文档 Markdown 文本的字符串：

```sh
curl -X 'POST' \
  'http://127.0.0.1:8503/predict/' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@<PDFFILE.pdf>;type=application/pdf'
```

如需将转换范围限制在第 1 至 5 页，在请求 URL 中添加参数即可：`http://127.0.0.1:8503/predict/?start=1&stop=5`。

## 数据集

### 生成数据集

构建训练数据集需要准备：

1. 包含源 PDF 文件的目录；
2. 包含对应 `.html` 文件的目录（由 [LaTeXML](https://math.nist.gov/~BMiller/LaTeXML/) 处理 `.tex` 文件生成），且目录结构与 PDF 目录保持一致；
3. [pdffigures2](https://github.com/allenai/pdffigures2) 二进制文件，并设置对应的环境变量：`export PDFFIGURES_PATH="/path/to/binary.jar"`。

随后执行切分命令：

```sh
python -m nougat.dataset.split_htmls_to_pages --html path/html/root --pdfs path/pdf/root --out path/paired/output --figure path/pdffigures/outputs
```

可选参数说明：

| 参数 | 说明 |
| :--- | :--- |
| `--recompute` | 重新计算所有切分数据 |
| `--markdown MARKDOWN` | Markdown 输出目录路径 |
| `--workers WORKERS` | 使用的并行进程数量 |
| `--dpi DPI` | 页面图像保存的分辨率 |
| `--timeout TIMEOUT` | 单篇论文的最大处理超时时间（秒） |
| `--tesseract` | 对每个页面生成 Tesseract OCR 预测基准 |

接下来创建包含所有图像路径、Markdown 文本及元信息的 `jsonl` 索引文件：

```sh
python -m nougat.dataset.create_index --dir path/paired/output --out index.jsonl
```

对每个 `jsonl` 文件生成 seek 映射表以加速数据加载：

```sh
python -m nougat.dataset.gen_seek file.jsonl
```

最终生成的数据目录结构如下：

```
root/
├── images
├── train.jsonl
├── train.seek.map
├── test.jsonl
├── test.seek.map
├── validation.jsonl
└── validation.seek.map
```

注意：`path/paired/output`（即上文中的 `images`）中的 `.mmd` 与 `.json` 文件后续不再需要，可予以清理，以便在同步至 S3 存储桶时减少一半的文件数量。

## 模型训练

如需训练或微调 Nougat 模型，执行以下命令：

```sh
python train.py --config config/train_nougat.yaml
```

## 模型评估

执行评估命令：

```sh
python test.py --checkpoint path/to/checkpoint --dataset path/to/test.jsonl --save_path path/to/results.json
```

若需获取不同文本模态的细分评估指标，执行：

```sh
python -m nougat.metrics path/to/results.json
```

## 常见问题 (FAQ)

- **为什么只输出 `[MISSING_PAGE]`？**

  Nougat 是在 arXiv 和 PMC 上的科技学术论文上训练得到的。所处理的文档是否与该类论文结构相近？此外需确认文档的语言类型：Nougat 在英文论文上表现最佳，其他拉丁语系语言可能有一定可用性，但**不支持中文、俄文、日文等语言**。
  若确认符合上述要求，则可能是由于在 CPU 或老旧 GPU 上计算时失败检测启发式规则产生误报（详见 [#11](https://github.com/facebookresearch/nougat/issues/11)）。可尝试添加 `--no-skipping` 参数解决。

- **从何处下载预训练模型权重？**

  权重文件已上传至 GitHub 仓库的 Release 发布页面。在程序首次执行时也会自动触发下载。可通过 `--model 0.1.0-{base,small}` 参数选择所需的模型规格。

## 引用

如需在学术研究中引用本项目，请使用以下格式：

```bibtex
@misc{blecher2023nougat,
      title={Nougat: Neural Optical Understanding for Academic Documents},
      author={Lukas Blecher and Guillem Cucurull and Thomas Scialom and Robert Stojnic},
      year={2023},
      eprint={2308.13418},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## 致谢

本项目基于 [Donut](https://github.com/clovaai/donut/) 代码库开发构建。

## 许可证

Nougat 代码库采用 [MIT 许可证](LICENSE) 开源。

Nougat 预训练模型权重遵循 [CC-BY-NC 许可证](LICENSE-MODEL.md)。
