# AI Car Price Data Analyst

Automated EDA and a lightweight ML pipeline for car price data — CLI and Streamlit UI to demonstrate end‑to‑end data analysis, modeling and reporting.

## Overview

AI Car Price Data Analyst is a compact, portfolio‑ready project that loads tabular car data (CSV), runs automated exploratory data analysis (EDA), trains simple regression models, generates visualizations, and exports a human‑readable report. The Streamlit app provides an interactive front‑end for uploading datasets and running the same pipeline.

This repository focuses on demonstrating practical data skills (ingestion, cleaning, EDA, modeling, evaluation, visualization, testing and a small UI) rather than maximizing prediction accuracy for the bundled dataset.

## Features

- Robust CSV loading and validation (defaults to `data/archive/example.csv`).
- Automated EDA:
  - numeric vs categorical detection,
  - row/column counts and data preview,
  - missing value percentages,
  - simple data quality flags (single‑value columns, high missingness).
- Price‑specific analysis (when `Price` exists):
  - correlation computation and qualitative strength labels,
  - short human‑readable interpretation.
- Two modeling workflows:
  - Model v1: LinearRegression on numeric features (baseline).
  - Model v2: RandomForest and GradientBoosting with one‑hot encoding for categoricals; selects best model by R² and reports feature importances.
- Visualization generation (PNG): histograms, correlation heatmap, top models and brands.
- CLI script (`src/main.py`) writes `report.txt`.
- Streamlit app (`src/app.py`) for interactive upload, run and download.
- Unit tests with pytest covering core functions.

## Demo

CLI:

```bash
python src/main.py
# Generates report.txt and plots/ PNG files
```

Streamlit:

```bash
streamlit run src/app.py
# Open http://localhost:8501
```

## Installation

Recommended: create a clean environment (conda / mamba / venv).

Example with mamba/conda:

```bash
mamba create -n ai-data python=3.10 -y
mamba activate ai-data
pip install -r requirements.txt
```

Minimal required packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn streamlit pytest
```

## Usage

CLI

```bash
# place your CSV in data/ (or let the script discover it)
python src/main.py
# Output: report.txt (root) and generated images under plots/
```

Streamlit

- Open the app, upload a CSV or select the built‑in sample dataset, then press "Run analysis".
- Download the full text report from the UI.

Behavior notes:

- If `Price` is missing, price‑specific analysis and models are skipped and the app informs the user.
- Plots are saved to `plots/` and displayed in the Streamlit UI when available.

## Project structure

```
ai-data-analyst/
├─ README.md
├─ data/                 # input CSVs (e.g. example.csv based on Kaggle cars-pre)
├─ plots/                # generated PNG charts
├─ reports/              # (optional) saved reports/images
├─ report.txt            # sample CLI output
├─ src/
│  ├─ main.py            # core pipeline: EDA, models, plots, reporting
│  └─ app.py             # Streamlit UI
└─ tests/
   └─ test_main.py       # pytest unit tests
```

## Models & methodology

- Model v1: LinearRegression trained on numeric features only. Simple baseline to validate signal and pipeline.
- Model v2: Preprocessing via ColumnTransformer (numeric passthrough + OneHotEncoder for categoricals). Compares RandomForestRegressor and GradientBoostingRegressor, selects best by R², and extracts feature importances for tree models.
- Evaluation metrics: R² and Mean Absolute Error (MAE). Simple train/test split (80/20).

Design choices:

- Emphasis on reproducible, interpretable pipeline and clear reporting.
- Conservative preprocessing: drop rows with missing values for modeling in this early version.
- One‑hot encoding for categorical variables to keep models interpretable and straightforward.

## Visualizations

- Distribution/histograms for numeric columns.
- Correlation matrix heatmap for numeric features (saved as PNG).
- Bar plots for most common car models and brands (if present).
- Saved to `plots/` and displayed in the Streamlit app.

## Tests

Run unit tests with pytest:

```bash
pip install pytest
pytest
```

Tests exercise:

- data loading (success + FileNotFoundError),
- basic overview and column analysis,
- price relationship reporting,
- baseline modeling behavior on small synthetic datasets.

## Dataset notes & limitations

The bundled sample is derived from a Kaggle cars dataset (cars‑pre). That dataset contains weak/noisy signals for price in this implementation; you may observe very low or negative R² values for baseline models. This repository demonstrates a complete, reproducible analysis pipeline (EDA, preprocessing, basic feature engineering, model training, evaluation and reporting) rather than optimized predictive performance on this particular synthetic dataset.

If you want to test predictive performance, use a larger, higher‑quality dataset or spend more effort on feature engineering and cleaning (outlier handling, target transformation, cross‑validation, hyperparameter tuning).

## Skills demonstrated

- Data ingestion and validation (pandas)
- Exploratory data analysis (EDA) and reporting
- Basic preprocessing and feature handling
- Model training, selection and evaluation (scikit‑learn)
- Visualization (matplotlib / seaborn)
- Unit testing (pytest)
- Simple interactive UI with Streamlit
- Reproducible scripting and reporting

## Tech stack

- Python 3.10
- pandas, numpy
- matplotlib, seaborn
- scikit‑learn
- Streamlit
- pytest

## Verified environment

The project was developed and tested with the following stack:

**Python**

- Python 3.10.12

**Core data & ML libraries**

- pandas 2.2.3
- numpy 2.1.3
- matplotlib 3.9.2
- seaborn 0.13.2
- scikit-learn 1.7.0
- scipy 1.15.3

**Web UI**

- streamlit 1.51.0

**Testing**

- pytest 8.4.0

> Note: The `requirements.txt` file in this repo pins these versions for reproducibility.  
> Newer compatible versions should also work, but this setup is known to be stable.

## Dependency files (what they are and how to use them)

- `environment.yml` (or `environment,yml` in this repo): conda/mamba environment specification. Use this file to create an isolated conda environment with pinned package versions for reproducible results:

  ```bash
  mamba env create -f environment.yml
  conda activate ai-data
  ```

  If the file in the repository has a different name (e.g. `environment,yml`), either rename it to `environment.yml` or pass the exact filename to the create command.

- `requirements.txt`: pip-style requirements for use with virtualenv / venv or inside an existing conda environment. Install packages with:

  ```bash
  pip install -r requirements.txt
  ```

- Why both?

  - Use `environment.yml` when you prefer conda/mamba for binary dependency management and exact reproducibility.
  - Use `requirements.txt` for lightweight installs with pip or when deploying into environments that expect pip requirements.

- Notes:
  - Both files pin versions that were verified during development. Update versions intentionally and test the pipeline again if you change them.
  - The pip section in `environment.yml` installs a couple of packages (Streamlit, pytest) that are not always required for minimal CLI runs; use `pip` to install extras as needed.

## Roadmap / Extensions

Short term

- Add richer preprocessing (imputation, outlier handling).
- Add cross‑validation and hyperparameter search.
- Save richer report formats (Markdown / HTML) with embedded images.

Medium term

- Add feature engineering pipelines and model explainability (SHAP).
- Deploy Streamlit app (Heroku / Streamlit Cloud / Docker).

Long term

- Convert into a reusable library with API, CLI flags, and better configuration.
