# Online Retail — Customer Return Prediction

A machine learning project that predicts whether a customer will come back and make
another purchase, based on their past buying behaviour.

This is my first end-to-end ML project, so the focus is on getting the **workflow**
right — a clean, leakage-free setup — rather than squeezing out the highest possible score.

## Problem

Given a customer's activity in an **observation window**, predict whether they will
purchase again in a later **label window**.

The two windows are separated by a single cutoff date. Features come only from *before*
the cutoff and the target comes only from *after* it, so information from the future
never leaks into the features.

- **Observation window:** transactions before `2010-08-31` → used to build features
- **Label window:** transactions on/after `2010-08-31` → used to build the target
  (`1` = customer returned, `0` = did not)

The boundary is controlled by a single `CUTOFF` constant in the notebook, so the whole
experiment can be shifted in time by changing one line.

## Features

All features are computed per customer, from the observation window only:

- `monetary` — total revenue
- `frequency` — number of distinct invoices
- `recency` — days since last purchase
- `tenure` — days since first purchase
- `cancel_rate` — share of cancelled lines (invoices starting with `C`)
- `n_products` — number of distinct products bought
- `trend` — second-half spend minus first-half spend (spending momentum)

## Models

- `DummyClassifier` — baseline (predicts the majority class)
- `LogisticRegression` — linear model with feature scaling
- `RandomForestClassifier` — tree ensemble
- `XGBoost` — gradient boosting

Tree models are reported with **train and test** scores side by side to make
overfitting visible. All models are compared on the same test set in a final table.

## Results (summary)

- **Logistic Regression** is the most balanced model: it beats the baseline and
  generalises well, and its coefficients are interpretable.
- **Random Forest** overfits with the chosen hyperparameters (high train, lower test).
- **XGBoost** is too powerful for such a small dataset and overfits.

See the notebook for the exact numbers and plots.

## Project structure

- `data/` — dataset (not tracked in git — see below)
- `model_clean.ipynb` — the project notebook
- `README.md`

## Getting the data

The dataset is **not included** in the repository (it is git-ignored).

1. Download the **Online Retail II** dataset from the UCI Machine Learning Repository:
   https://archive.ics.uci.edu/dataset/502/online+retail+ii
2. Place the Excel file at:
   ```
   data/online_retail_II.xlsx
   ```

## How to run

```bash
pip install pandas numpy matplotlib scikit-learn xgboost openpyxl jupyter
jupyter notebook model_clean.ipynb
```

Then run all cells top to bottom.

## Tech stack

Python · pandas · scikit-learn · XGBoost · matplotlib
