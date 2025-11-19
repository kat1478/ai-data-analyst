# AI Data Analyst

A Python-based tool for automatic exploratory data analysis (EDA). The project loads CSV files, performs structural inspection of the dataset, detects missing values and suspicious columns, and generates a clean, readable text report.

## Current Phase: Phase 1 — Ready

Phase 1 (core analysis) is complete and working. You can run the project locally to produce automatic EDA reports.

What Phase 1 provides

- Load CSV files from data/
- Basic dataset overview (rows, columns, preview)
- Separate numerical and categorical features
- Calculate missing value percentages
- Detect simple data issues (columns with too many missing values or with only one unique value)
- Save a readable report to report.txt

## Project Structure

```
ai-data-analyst/
├── README.md
├── data/
│   └── example.csv
├── plots/        # (for Phase 2)
├── tests/
│   └── test_main.py
└── src/
    └── main.py
```

## How to Run (Phase 1)

1. Create and activate the environment (example using mamba / conda):

```
mamba create -n ai-data python=3.10 -y
mamba activate ai-data
```

If you use conda only:

```
conda create -n ai-data python=3.10 -y
conda activate ai-data
```

2. Install dependencies:

```
pip install pandas numpy matplotlib seaborn
```

3. Place your CSV in the `data/` folder (or let the script search recursively).

4. Run the analysis:

```
python src/main.py
```

5. Output:

- report.txt created in the project root with the EDA summary.

## Tests

Run tests with pytest:

```
pip install pytest
pytest
```

## Next steps (Phase 2+)

Planned enhancements:

- Add automatic plots (histograms, heatmaps) and save images to plots/
- Add correlation summaries and simple models (scikit-learn)
- Build a lightweight UI (Streamlit) for file upload and interactive results
- Polish README with screenshots and a short case-study for portfolio

## Purpose

This project is a compact, demonstrable example of end-to-end data work: loading data, cleaning/inspecting it, and producing reproducible, presentable analysis — useful for portfolio and job applications.
