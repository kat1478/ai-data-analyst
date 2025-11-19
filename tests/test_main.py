import sys
import pandas as pd
import pytest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.main import (
    basic_overview,
    analyze_columns,
    load_data,
    analyze_price_relationships,
    train_price_model,
    train_price_model_v2,
)




def test_basic_overview_contains_shape_and_head(tmp_path):
    df = pd.DataFrame({
        "A": [1, 2, 3],
        "B": ["x", "y", "z"],
    })

    report = basic_overview(df)

    assert "=== BASIC OVERVIEW ===" in report
    assert "Number of rows: 3" in report
    assert "Number of columns: 2" in report
    assert "A" in report
    assert "B" in report


def test_analyze_columns_detects_numeric_and_categorical():
    df = pd.DataFrame({
        "num": [1, 2, 3],
        "cat": ["a", "b", "c"],
        "all_same": [1, 1, 1],
    })

    report = analyze_columns(df)

    assert "Numeric columns:" in report
    assert "num" in report
    assert "Categorical columns:" in report
    assert "cat" in report

    assert "- num: 0.00%" in report
    assert "- cat: 0.00%" in report

    assert "Suspicious columns:" in report
    assert "all_same" in report
    assert "only one unique value" in report


def test_load_data_raises_if_missing(tmp_path):
    fake_path = tmp_path / "nonexistent.csv"

    with pytest.raises(FileNotFoundError):
        load_data(str(fake_path))


def test_load_data_reads_existing_file(tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("x,y\n1,2\n3,4\n", encoding="utf-8")

    df = load_data(str(csv_path))

    assert not df.empty
    assert list(df.columns) == ["x", "y"]
    assert len(df) == 2

def test_analyze_price_relationships_no_price_column():
    df = pd.DataFrame({
        "A": [1, 2, 3],
        "B": [4, 5, 6],
    })

    report = analyze_price_relationships(df, target_col="Price")

    assert "=== PRICE RELATIONSHIPS ===" in report
    assert "Target column 'Price' not found in the dataset." in report


def test_analyze_price_relationships_simple_case():
    # Construct simple data:
    # Price increases with feature1 and decreases with feature2.
    df = pd.DataFrame({
        "Price":   [10, 20, 30, 40, 50],
        "feature1": [1, 2, 3, 4, 5],   
        "feature2": [5, 4, 3, 2, 1],   
    })

    report = analyze_price_relationships(df, target_col="Price")

    assert "=== PRICE RELATIONSHIPS ===" in report
    assert "Correlations with Price:" in report

    assert "feature1" in report
    assert "feature2" in report

    # ensure words like strong/moderate/weak etc. appear
    assert "correlation" in report

    assert "Interpretation:" in report
    assert "feature1" in report
    assert "feature2" in report


def test_train_price_model_no_price_column():
    # lack of target column
    df = pd.DataFrame({
        "A": [1, 2, 3],
        "B": [4, 5, 6],
    })

    report = train_price_model(df, target_col="Price")

    assert "=== PRICE PREDICTION MODEL ===" in report
    assert "Target column 'Price' not found in the dataset." in report


def test_train_price_model_not_enough_rows():
    # only 2 rows
    df = pd.DataFrame({
        "Price": [1.0, 2.0],
        "feature": [10.0, 20.0],
    })

    report = train_price_model(df, target_col="Price")

    assert "=== PRICE PREDICTION MODEL ===" in report
    assert "Not enough rows after dropping missing values to train a model." in report


def test_train_price_model_on_perfect_linear_data():
    # create data with perfect linear relationship:
    # Price = 2 * feature
    feature_values = list(range(0, 200))
    prices = [2 * x for x in feature_values]

    df = pd.DataFrame({
        "Price": prices,
        "feature": feature_values,
    })

    report = train_price_model(df, target_col="Price")

    # check if training was reported
    assert "Trained LinearRegression model on numeric features." in report

    # R²
    # looking for "R² score on test set:"
    r2_line = next(
        (line for line in report.splitlines() if "R² score on test set:" in line),
        None,
    )
    assert r2_line is not None

    # numeric value extraction
    r2_str = r2_line.split(":")[-1].strip()
    r2_value = float(r2_str)

    # expecting perfect R²
    assert r2_value > 0.9

def test_train_price_model_v2_runs_and_reports():
    # syntetic dataset
    n = 120
    years = list(range(2000, 2000 + n))
    brands = ["BrandA" if i < n / 2 else "BrandB" for i in range(n)]
    engine_size = [2.0 + (i % 3) * 0.5 for i in range(n)]

    # BrandB cars are more expensive
    base_price = [10000 + 500 * (year - 2000) for year in years]
    brand_effect = [0 if b == "BrandA" else 8000 for b in brands]
    price = [bp + be for bp, be in zip(base_price, brand_effect)]

    df = pd.DataFrame(
        {
            "Price": price,
            "Year": years,
            "Engine Size": engine_size,
            "Brand": brands,
            "Condition": ["New"] * n,
        }
    )

    report = train_price_model_v2(df, target_col="Price")

    # check only for presence of key report sections
    assert "=== PRICE PREDICTION MODEL V2" in report
    assert "RandomForest" in report
    assert "GradientBoosting" in report
    assert "Best model:" in report
    # if feature_importances_ is available, it should be reported
    assert "Top 5 most important features" in report or "does not expose feature_importances" in report
