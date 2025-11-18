import pandas as pd
from pathlib import Path


def load_data(path: str) -> pd.DataFrame:
    """Load CSV file into a pandas DataFrame."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {csv_path.resolve()}")
    df = pd.read_csv(csv_path)
    return df


def basic_overview(df: pd.DataFrame) -> str:
    """Create a basic overview of the dataset."""
    rows, cols = df.shape
    info_lines = []
    info_lines.append("=== BASIC OVERVIEW ===")
    info_lines.append(f"Number of rows: {rows}")
    info_lines.append(f"Number of columns: {cols}")
    info_lines.append("\nFirst 5 rows:")
    info_lines.append(df.head().to_string())
    return "\n".join(info_lines)


def analyze_columns(df: pd.DataFrame) -> str:
    """Analyze column types, missing values and suspicious columns."""
    report_lines = []
    report_lines.append("\n=== COLUMN ANALYSIS ===")

    # Numeric vs categorical
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=["number"]).columns.tolist()

    report_lines.append("\nNumeric columns:")
    report_lines.append(", ".join(numeric_cols) if numeric_cols else "None")

    report_lines.append("\nCategorical columns:")
    report_lines.append(", ".join(categorical_cols) if categorical_cols else "None")

    # Missing values
    report_lines.append("\nMissing values (%):")
    missing_percent = df.isna().mean() * 100
    for col, val in missing_percent.items():
        report_lines.append(f"- {col}: {val:.2f}%")

    # Suspicious columns
    report_lines.append("\nSuspicious columns:")
    any_suspicious = False
    for col in df.columns:
        unique_vals = df[col].nunique(dropna=True)
        if missing_percent[col] > 50:
            report_lines.append(f"- {col}: >50% missing values")
            any_suspicious = True
        if unique_vals == 1:
            report_lines.append(f"- {col}: only one unique value")
            any_suspicious = True

    if not any_suspicious:
        report_lines.append("None detected based on simple heuristics.")

    return "\n".join(report_lines)


def save_report(text: str, path: str) -> None:
    """Save report text to a file."""
    report_path = Path(path)
    report_path.write_text(text, encoding="utf-8")


def find_data_file(filename: str) -> Path:
    """Search the project data/ directory (recursively) for filename and return the first match."""
    base = Path(__file__).parent.parent / "data"
    matches = list(base.rglob(filename))
    if not matches:
        raise FileNotFoundError(f"Could not find {filename} under {base.resolve()}")
    return matches[0]


def main():
    # search for example.csv anywhere under data/ (handles an extra folder)
    data_path = find_data_file("example.csv")

    df = load_data(str(data_path))
    basic = basic_overview(df)
    columns_report = analyze_columns(df)

    full_report = basic + "\n\n" + columns_report

    report_path = Path(__file__).parent.parent / "report.txt"
    save_report(full_report, report_path)

    print("Report generated and saved to:", report_path.resolve())


if __name__ == "__main__":
    main()
