# pubmlp

Multimodal publication classifier with LLM and deep learning. Fuses transformer embeddings with tabular features through a multilayer perceptron (MLP) for human-in-the-loop screening workflows.

## Installation

```bash
pip install pubmlp
```

With optional dependencies:

```bash
pip install pubmlp[screening]  # screening tools (openpyxl, nltk)
pip install pubmlp[dev]        # development (pytest, ruff)
pip install pubmlp[docs]       # documentation (sphinx)
```

## Citation

- Shin, M. (2026). *pubmlp: Multimodal publication classifier with LLM and deep learning* (Python package version 0.1.0) \[Computer software\]. <a href="https://github.com/mshin77/pubmlp">https://github.com/mshin77/pubmlp</a>

```{toctree}
:hidden:

getting-started
vignettes/screening-workflow
api
support
```
