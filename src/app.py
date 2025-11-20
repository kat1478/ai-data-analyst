import sys
from pathlib import Path

import pandas as pd
import streamlit as st

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


def build_full_report(df: pd.DataFrame) -> tuple[str, str, list[str]]:
    """Run the full pipeline and return:
    - full report text (str)
    - visualization section text (str)
    - list of plot filenames (list[str])
    """
    basic = basic_overview(df)
    col_report = analyze_columns(df)

    has_price = "Price" in df.columns
    if has_price:
        price_report = analyze_price_relationships(df, target_col="Price")
        model_v1 = train_price_model(df, target_col="Price")
        model_v2 = train_price_model_v2(df, target_col="Price")
    else:
        price_report = "No 'Price' column detected – skipping price-specific analysis.\n"
        model_v1 = "Skipping baseline model (no 'Price' column).\n"
        model_v2 = "Skipping advanced models (no 'Price' column).\n"

    viz_report = generate_plots(df)

    full_report = (
        basic
        + "\n\n"
        + col_report
        + "\n\n"
        + price_report
        + "\n\n"
        + model_v1
        + "\n\n"
        + model_v2
        + "\n\n"
        + viz_report
    )

    plot_files = [
        line[2:].strip()
        for line in viz_report.splitlines()
        if line.strip().startswith("- ")
    ]

    return full_report, viz_report, plot_files


def main():
    st.set_page_config(
        page_title="AI Car Price Data Analyst",
        page_icon="🚗",
        layout="wide",
    )

    st.title("AI Car Price Data Analyst 🚗")

    tabs = st.tabs(["🏠 Home", "📊 Analysis"])

    # --------- TAB 1: HOME / UPLOAD ----------
    with tabs[0]:
        st.markdown(
            "Upload a CSV with car data or use the built-in Kaggle dataset. "
            "Then go to the **Analysis** tab to run automatic EDA, ML models and visualizations."
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

        # save df to session_state so the Analysis tab can access it
        if df is not None:
            st.session_state["df"] = df
            st.subheader("Preview of data:")
            st.dataframe(df.head())
            st.success("Data loaded. Go to the **Analysis** tab to run the pipeline.")
        else:
            st.warning("Upload a CSV file or select the built-in dataset.")

    # --------- TAB 2: ANALYSIS ----------
    with tabs[1]:
        st.subheader("📊 Analysis and report")

        if "df" not in st.session_state:
            st.info("No data loaded yet. Go to the **Home** tab and upload/select a dataset.")
            return

        df = st.session_state["df"]

        # RUN ANALYSIS button controlled via session_state
        if "analysis_ready" not in st.session_state:
            st.session_state["analysis_ready"] = False

        run_btn = st.button("Run analysis", type="primary")

        if run_btn:
            st.session_state["analysis_ready"] = True

        if st.session_state["analysis_ready"]:
            full_report, viz_report, plot_files = build_full_report(df)

            # report sections in the UI
            st.markdown("### 1️⃣ Basic overview")
            st.text(basic_overview(df))

            st.markdown("### 2️⃣ Column analysis")
            st.text(analyze_columns(df))

            if "Price" in df.columns:
                st.markdown("### 3️⃣ Price relationships")
                st.text(analyze_price_relationships(df, target_col="Price"))

                st.markdown("### 4️⃣ ML models")
                st.text(train_price_model(df, target_col="Price"))
                st.text(train_price_model_v2(df, target_col="Price"))
            else:
                st.info("No 'Price' column detected – skipping price-specific analysis and models.")

            st.markdown("### 5️⃣ Visualizations")
            st.text(viz_report)

            # images
            for fname in plot_files:
                img_path = PLOTS_DIR / fname
                if img_path.exists():
                    st.image(str(img_path), caption=fname)

            # download report
            st.markdown("---")
            st.download_button(
                label="📥 Download full report as TXT",
                data=full_report,
                file_name="report.txt",
                mime="text/plain",
            )
        else:
            st.info("Click **Run analysis** to generate the report and visualizations.")


if __name__ == "__main__":
    main()
