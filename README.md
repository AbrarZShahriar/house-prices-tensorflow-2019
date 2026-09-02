# House Prices with the TensorFlow Low-Level API

This repository contains a 2019 notebook and report for Kaggle's [House Prices:
Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
competition. The project explores preprocessing and feature engineering before
training fully connected regression networks with TensorFlow's original
graph-and-session API.

> Project status: historical machine-learning assignment. The notebook records
> a Python 3.7 environment and TensorFlow 1.x-style code. It is not expected to
> run unchanged with current TensorFlow releases.

## Result

The final model used two hidden layers with 500 and 300 neurons, batch
normalization, ReLU activations, and dropout with a keep probability of 0.8.
The report records a Kaggle score of `0.11694` and rank 685, which was within
the top 15.6% at the time of submission.

## Contents

- [`HousePrice2019May30_A1822.ipynb`](./HousePrice2019May30_A1822.ipynb) - the
  notebook with preprocessing, experiments, learning curves, and the final
  model.
- [`HousePrice2019May30_A1822.pdf`](./HousePrice2019May30_A1822.pdf) - a static
  rendered copy of the notebook and report.

## Data and environment

The Kaggle data is not included. The notebook reads `../input/train.csv` and
`../input/test.csv`, so place the competition files in a sibling `input`
directory or change those paths in the notebook.

The notebook imports NumPy, pandas, seaborn, Matplotlib, SciPy, scikit-learn,
and TensorFlow. Reproducing the original run requires a compatible legacy
Python and TensorFlow 1.x environment. A current environment requires a code
port to supported TensorFlow or `tensorflow.compat.v1` APIs. Exact package
versions and a frozen data snapshot were not recorded, so the historical score
is documented rather than claimed as a reproducible benchmark.
