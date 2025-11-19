import sys
import pandas as pd
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.main import basic_overview, analyze_columns, load_data


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
