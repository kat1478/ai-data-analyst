# AI Data Analyst

A Python-based tool for automatic exploratory data analysis (EDA). The project loads CSV files, performs structural inspection of the dataset, detects missing values and suspicious columns, and generates a clean, readable text report.

## Current Phase: Basic Analysis (Phase 1)

At this stage, the project supports:

- Loading CSV files
- Displaying basic dataset overview (rows, columns, preview)
- Separating numerical and categorical features
- Calculating missing value percentages
- Detecting simple data issues (columns with too many missing values or with only one unique value)
- Saving the full analysis report to a text file

## Project Structure

```
ai-data-analyst/
├── README.md
├── data/
│   └── example.csv
├── plots/        # (to be used in Phase 2)
└── src/
    └── main.py
```

## How to Run

1. Create and activate the environment:

```
mamba create -n ai-data python=3.10 -y
mamba activate ai-data
```

2. Install dependencies:

```
pip install pandas numpy matplotlib seaborn
```

3. Run the analysis:

```
python src/main.py
```

4. The output report will be generated as `report.txt` in the project directory.

## Next Steps (Phase 2+)

Planned enhancements:

- Automatic generation of plots (histograms, correlation heatmaps)
- Outlier detection
- Simple machine learning models for predictive analysis
- A web-based interface (Streamlit) for interactive analysis
- AI-generated insights describing detected patterns

## Purpose

This project is designed as a portfolio-friendly, end-to-end demonstration of practical data analysis automation, useful for showcasing skills in Python, data processing, and building analytical tools.
