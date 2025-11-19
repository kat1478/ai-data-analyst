import sys
from pathlib import Path

import streamlit as st
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.main import (
    load_data,
    basic_overview,
    analyze_columns,
    analyze_price_relationships,
    train_price_model,
    train_price_model_v2,
    generate_plots,
)

PLOTS_DIR = PROJECT_ROOT / "plots"
DEFAULT_CSV = PROJECT_ROOT / "data" / "archive" / "example.csv"


def run_analysis(df: pd.DataFrame):
    st.subheader("1️⃣ Basic overview")
    basic = basic_overview(df)
    st.text(basic)

    st.subheader("2️⃣ Column analysis")
    col_report = analyze_columns(df)
    st.text(col_report)

    has_price = "Price" in df.columns

    if has_price:
        st.subheader("3️⃣ Price relationships")
        price_report = analyze_price_relationships(df, target_col="Price")
        st.text(price_report)

        st.subheader("4️⃣ ML models")
        model_v1 = train_price_model(df, target_col="Price")
        st.text(model_v1)

        model_v2 = train_price_model_v2(df, target_col="Price")
        st.text(model_v2)
    else:
        st.info(
            "No 'Price' column detected – skipping price-specific analysis and models."
        )

    st.subheader("5️⃣ Visualizations")
    viz_report = generate_plots(df)
    st.text(viz_report)

    plot_files = [
        line[2:].strip()
        for line in viz_report.splitlines()
        if line.strip().startswith("- ")
    ]

    for fname in plot_files:
        img_path = PLOTS_DIR / fname
        if img_path.exists():
            st.image(str(img_path), caption=fname)



def main():
    st.title("AI Car Price Data Analyst 🚗")
    st.write(
        "Upload a CSV with car data or use the built-in Kaggle dataset. "
        "The app will run automatic EDA, train ML models and show visualizations."
    )

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    use_default = st.checkbox(
        "Use built-in sample dataset (Kaggle cars-pre)", value=uploaded_file is None
    )

    df: pd.DataFrame | None = None

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.info("Using uploaded file.")
    elif use_default:
        try:
            df = load_data(str(DEFAULT_CSV))
            st.info(f"Using built-in dataset: {DEFAULT_CSV.name}")
        except FileNotFoundError:
            st.error(
                f"Default dataset not found at {DEFAULT_CSV}. "
                "Make sure example.csv is in the data/ folder."
            )

    if df is not None:
        st.write("Preview of data:")
        st.dataframe(df.head())

        if st.button("Run analysis", type="primary"):
            run_analysis(df)
    else:
        st.warning("Upload a CSV file or select the built-in dataset.")


if __name__ == "__main__":
    main()
