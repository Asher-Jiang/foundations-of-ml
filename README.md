# Foundations of ML — Homework Code

Each homework lives in its own directory (`homework_1/`, `homework_2/`, ...)
with its own pinned `requirements.txt`. Scripts locate their data files
relative to their own path, so they can be run from any working directory.
Commands below are given relative to the repository root.

## Setup

Requires Python 3.13.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Or, with [uv](https://docs.astral.sh/uv/): `uv venv --python 3.13 && source .venv/bin/activate`
(and use `uv pip install` in place of `pip install` below).

## Homework 1

```bash
pip install -r homework_1/requirements.txt
python homework_1/problem4.py                 # QQ plots + normality test of residuals (~1 s)
python homework_1/problem5_regression.py      # linear / t-distribution regression on ale_dataset.csv (~5 s)
python homework_1/problem5_classification.py  # logistic / probit classification on breast cancer data (~20 s)
```

`problem4.py` writes its figure to `homework_1/problem4_qq.png` and then calls
`plt.show()`. In a headless or non-interactive environment, set
`MPLBACKEND=Agg` so that `plt.show()` does not block:

```bash
MPLBACKEND=Agg python homework_1/problem4.py
```

Train/test splits are random (no fixed seed), so reported metrics vary
slightly between runs.
