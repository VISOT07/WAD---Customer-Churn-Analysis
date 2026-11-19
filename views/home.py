"""
Page 1: Home & Data
===================
Features:
    F1  Load dataset   - use the sample dataset or upload a CSV file
    F2  Data overview  - size, preview, missing values, statistics
"""

import pandas as pd
import plotly.express as px
import streamlit as st

from utils import (
    DATA_PATH,
    NUMERIC_COLS,
    TARGET,
    clean_data,
    get_data,
    get_data_name,
    missing_columns,
)

# The two choices of the data source selector
SAMPLE_OPTION = "Use sample dataset"
UPLOAD_OPTION = "Upload my CSV"


# =============================================================================
# SECTION 1: LOAD DATA (F1)
# =============================================================================

def handle_upload(user_has_uploaded: bool) -> None:
    """
    Show the file uploader and save a valid uploaded file.

    The uploaded data is saved in st.session_state, so the other pages
    can use it too.
    """
    uploaded_file = st.file_uploader("Upload a customer CSV", type="csv")

    if uploaded_file is None:
        if not user_has_uploaded:
            st.info("Upload a CSV file. Until then, the sample dataset is shown.")
        return

    uploaded_data = clean_data(pd.read_csv(uploaded_file))

    missing = missing_columns(uploaded_data)
    if len(missing) > 0:
        st.error(f"This file is missing these columns: {', '.join(missing)}")
        return

    st.session_state.df = uploaded_data
    st.session_state.data_name = uploaded_file.name


def show_data_source() -> None:
    """Let the user choose the sample dataset or their own CSV file."""
    st.subheader("1. Load data")

    with st.container(border=True):
        user_has_uploaded = st.session_state.get("df") is not None

        # Keep "Upload my CSV" selected if the user uploaded a file before
        if user_has_uploaded:
            selected_index = 1
        else:
            selected_index = 0

        source = st.radio(
            "Data source",
            [SAMPLE_OPTION, UPLOAD_OPTION],
            index=selected_index,
            horizontal=True,
            label_visibility="collapsed",
        )

        if source == UPLOAD_OPTION:
            handle_upload(user_has_uploaded)
        else:
            # Go back to the sample dataset
            st.session_state.df = None
            st.session_state.data_name = DATA_PATH.name

        st.markdown(f"Using: **{get_data_name()}**")


# =============================================================================
# SECTION 2: DATA OVERVIEW (F2)
# =============================================================================

def show_summary_numbers(df: pd.DataFrame) -> None:
    """Show four numbers: rows, columns, missing values and churn rate."""
    number_of_rows = len(df)
    number_of_columns = df.shape[1]
    number_missing = int(df.isna().sum().sum())
    churn_rate = (df[TARGET] == "Yes").mean()

    column_1, column_2, column_3, column_4 = st.columns(4)
    column_1.metric("Rows (customers)", f"{number_of_rows:,}")
    column_2.metric("Columns", number_of_columns)
    column_3.metric("Missing values", f"{number_missing:,}")
    column_4.metric("Churn rate", f"{churn_rate:.1%}")


def show_data_preview(df: pd.DataFrame) -> None:
    """Show the first 5 rows of the dataset."""
    with st.container(border=True):
        st.markdown("**Data preview (first 5 rows)**")
        st.dataframe(df.head(), hide_index=True)


def show_missing_values(df: pd.DataFrame) -> None:
    """Show a bar chart of the number of missing values in each column."""
    with st.container(border=True):
        st.markdown("**Missing values by column**")

        missing_counts = df.isna().sum()
        missing_counts = missing_counts[missing_counts > 0]
        missing_counts = missing_counts.sort_values()

        if missing_counts.empty:
            st.success("No missing values.")
            return

        figure = px.bar(
            x=missing_counts.values,
            y=missing_counts.index,
            orientation="h",
            text=missing_counts.values,
            labels={"x": "Missing values", "y": ""},
        )
        figure.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(figure)
        st.caption("Missing numbers are filled with the median during training.")


def show_summary_statistics(df: pd.DataFrame) -> None:
    """Show count, mean, min, max and more for the numeric columns."""
    with st.container(border=True):
        st.markdown("**Summary statistics**")
        statistics = df[NUMERIC_COLS].describe().round(2)
        st.dataframe(statistics)


# =============================================================================
# PAGE
# =============================================================================

def main() -> None:
    """Draw the page from top to bottom."""
    st.title("Customer Churn Prediction")
    st.caption("Find customers who are likely to leave, see why, and act early.")

    # Section 1: load data
    show_data_source()
    df = get_data()

    # Section 2: data overview
    st.subheader("2. Data overview")
    show_summary_numbers(df)
    show_data_preview(df)

    left, right = st.columns(2)
    with left:
        show_missing_values(df)
    with right:
        show_summary_statistics(df)


main()
