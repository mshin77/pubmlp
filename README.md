<img src="https://raw.githubusercontent.com/mshin77/pubmlp/main/logo.svg" alt="pubmlp Logo" align="right" width="220px"/>

[![PyPI version](https://img.shields.io/pypi/v/pubmlp)](https://pypi.org/project/pubmlp/)
[![Python versions](https://img.shields.io/pypi/pyversions/pubmlp)](https://pypi.org/project/pubmlp/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](https://opensource.org/licenses/MIT)

Multimodal publication classifier with LLM and deep learning. Fuses transformer embeddings from [Hugging Face](https://huggingface.co/docs/transformers) with tabular features through a multilayer perceptron (MLP) on [PyTorch](https://pytorch.org/) for human-in-the-loop screening. Screen by matched rules, by semantic similarity through [sentence-transformers](https://www.sbert.net/), by the classifier with active learning, or by a language model as a second rater. Read exports in every format [bibliometrix](https://www.bibliometrix.org/) accepts, extract page-anchored evidence with [pdfplumber](https://github.com/jsvine/pdfplumber), and report SAFE stopping and [PRISMA 2020](https://www.prisma-statement.org/) Item 8.

## Installation

```bash
pip install pubmlp
```

With optional dependencies:

```bash
pip install "pubmlp[screening]"  # regex and semantic screening, stratified sampling
pip install "pubmlp[fulltext]"   # PDF reading with page-anchored evidence
pip install "pubmlp[benchmark]"  # SYNERGY benchmark datasets
```

From GitHub:

```bash
pip install git+https://github.com/mshin77/pubmlp.git
```

## Getting Started

See [Quick Start](https://mshin77.github.io/pubmlp/getting-started.html) and [Screening Workflow](https://mshin77.github.io/pubmlp/vignettes/screening-workflow.html) for tutorials.

## Citation

- Shin, M. (2026). *pubmlp: Multimodal publication classifier with LLM and deep learning* (Python package version 0.6.0) [Computer software]. <https://github.com/mshin77/pubmlp>

## Reference

- Shin, M., & McKenna, J. (2026). Exploring the research landscape on single-case design methodology using technology through text mining and large language models. *Journal of Behavioral Education*. https://doi.org/10.1007/s10864-026-09630-1