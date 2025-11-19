import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error




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

def analyze_price_relationships(df: pd.DataFrame, target_col: str = "Price") -> str:
    """Analyze correlations between numeric features and the target column (e.g. Price)."""
    report_lines = []
    report_lines.append("\n=== PRICE RELATIONSHIPS ===")

    if target_col not in df.columns:
        report_lines.append(f"Target column '{target_col}' not found in the dataset.")
        return "\n".join(report_lines)

    # select only numeric columns + target
    numeric_df = df.select_dtypes(include=["number"])

    if target_col not in numeric_df.columns:
        report_lines.append(f"Target column '{target_col}' is not numeric.")
        return "\n".join(report_lines)

    corr = numeric_df.corr()[target_col].drop(labels=[target_col])

    if corr.empty:
        report_lines.append("No other numeric columns to correlate with target.")
        return "\n".join(report_lines)

    report_lines.append(f"\nCorrelations with {target_col}:")

    # sort from strongest to weakest
    corr_sorted = corr.sort_values(key=lambda x: x.abs(), ascending=False)

    for feature, value in corr_sorted.items():
        strength = ""
        abs_val = abs(value)
        if abs_val >= 0.7:
            strength = "strong"
        elif abs_val >= 0.4:
            strength = "moderate"
        elif abs_val >= 0.2:
            strength = "weak"
        else:
            strength = "very weak"

        direction = "positive" if value > 0 else "negative"
        report_lines.append(
            f"- {feature}: {value:.3f} ({strength}, {direction} correlation)"
        )

    # simple interpretation based on the top few features
    report_lines.append("\nInterpretation:")
    top_features = corr_sorted.head(3)
    for feature, value in top_features.items():
        direction_word = "higher" if value > 0 else "lower"
        price_trend = "higher prices" if value > 0 else "lower prices"
        report_lines.append(
            f"* {feature}: {direction_word} values tend to be associated with {price_trend} (corr = {value:.3f})."
        )

    return "\n".join(report_lines)

def train_price_model(df: pd.DataFrame, target_col: str = "Price") -> str:
    """Train a simple regression model to predict Price from numeric features and report basic metrics."""
    report_lines = []
    report_lines.append("\n=== PRICE PREDICTION MODEL ===")

    if target_col not in df.columns:
        report_lines.append(f"Target column '{target_col}' not found in the dataset.")
        return "\n".join(report_lines)

    # only numeric features
    numeric_df = df.select_dtypes(include=["number"]).copy()

    if target_col not in numeric_df.columns:
        report_lines.append(f"Target column '{target_col}' is not numeric.")
        return "\n".join(report_lines)

    # X - features, y - target
    X = numeric_df.drop(columns=[target_col], errors="ignore")
    y = numeric_df[target_col]

    # Simple preprocessing: drop rows with missing values
    data = pd.concat([X, y], axis=1).dropna()
    if data.shape[0] < 10:
        report_lines.append("Not enough rows after dropping missing values to train a model.")
        return "\n".join(report_lines)

    X = data.drop(columns=[target_col])
    y = data[target_col]

    # Split into training and test sets, *80-20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Training the model
    model = LinearRegression()
    model.fit(X_train, y_train)

   
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    report_lines.append("Trained LinearRegression model on numeric features.")
    report_lines.append(f"Number of training samples: {len(X_train)}")
    report_lines.append(f"Number of test samples: {len(X_test)}")
    report_lines.append(f"R² score on test set: {r2:.3f}")
    report_lines.append(f"Mean Absolute Error (MAE) on test set: {mae:.2f}")

    if r2 < 0:
        quality = "Model performs worse than a constant baseline (very poor fit)."
    elif r2 < 0.3:
        quality = "Model explains only a small portion of the variance (weak fit)."
    elif r2 < 0.6:
        quality = "Model has a moderate fit to the data."
    elif r2 < 0.8:
        quality = "Model explains a large portion of variance (good fit)."
    else:
        quality = "Model fits the data very well (high R²)."

    report_lines.append(f"Interpretation: {quality}")

    return "\n".join(report_lines)


def generate_plots(df: pd.DataFrame) -> str:
    """Generate histograms for numeric columns and a correlation heatmap, save them to plots/ folder."""
    plots_dir = Path(__file__).parent.parent / "plots"
    plots_dir.mkdir(exist_ok=True)

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    generated_files = []

    # Histograms for numeric features
    for col in numeric_cols:
        plt.figure()
        df[col].hist(bins=30)
        plt.title(f"Histogram of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        out_path = plots_dir / f"{col}_hist.png"
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        generated_files.append(out_path.name)

    # Correlation heatmap
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr()

        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Correlation Matrix (numeric features)")
        heatmap_path = plots_dir / "correlation_matrix.png"
        plt.savefig(heatmap_path, bbox_inches="tight")
        plt.close()
        generated_files.append(heatmap_path.name)

    report_lines = []
    report_lines.append("\n=== VISUALIZATIONS ===")
    if generated_files:
        report_lines.append("Generated plots:")
        for name in generated_files:
            report_lines.append(f"- {name}")
    else:
        report_lines.append("No numeric columns available to generate plots.")

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
    data_path = find_data_file("example.csv")

    df = load_data(str(data_path))
    basic = basic_overview(df)
    columns_report = analyze_columns(df)
    price_report = analyze_price_relationships(df, target_col="Price")
    model_report = train_price_model(df, target_col="Price")
    viz_report = generate_plots(df)

    full_report = (
        basic
        + "\n\n"
        + columns_report
        + "\n\n"
        + price_report
        + "\n\n"
        + model_report
        + "\n\n"
        + viz_report
    )


    report_path = Path(__file__).parent.parent / "report.txt"
    save_report(full_report, report_path)

    print("Report generated and saved to:", report_path.resolve())

if __name__ == "__main__":
    main()
