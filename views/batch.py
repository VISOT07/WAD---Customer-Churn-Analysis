"""
Page 5: Batch Prediction
========================
Features:
    F7  Batch prediction - upload a CSV and predict every customer
    F8  Risk level       - Low / Medium / High for each customer
    F9  Download results - save the predictions as a CSV file
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import (
    FEATURES,
    ID_COL,
    RISK_COLORS,
    TARGET,
    clean_data,
    get_data,
    missing_columns,
    require_model,
    risk_level,
)

RISK_LEVELS = ["High", "Medium", "Low"]

# Columns shown in the results table (the download has every column)
RESULT_COLUMNS = [
    ID_COL,
    "Contract",
    "tenure",
    "MonthlyCharges",
    "Churn probability",
    "Prediction",
    "Risk level",
    "Churn (actual)",
]


# =============================================================================
# UPLOAD (F7)
# =============================================================================

def to_csv_bytes(table: pd.DataFrame) -> bytes:
    """Convert a table to CSV text, ready for a download button."""
    return table.to_csv(index=False).encode("utf-8")


def show_template_download(container) -> None:
    """Show a button that downloads an example CSV with the expected columns."""
    sample_data = get_data()

    template_columns = []
    if ID_COL in sample_data.columns:
        template_columns.append(ID_COL)
    template_columns = template_columns + FEATURES

    template = sample_data[template_columns].head(5)

    container.download_button(
        "Download CSV template",
        to_csv_bytes(template),
        file_name="churn_template.csv",
        mime="text/csv",
    )


def show_upload_section():
    """
    Show the upload box and the "Run predictions" button.

    The page stops here if there is no file or the file is not valid.

    Returns:
        uploaded_file: the file the user uploaded
        customers:     the cleaned data of the file
        run_clicked:   True if the user clicked "Run predictions"
    """
    with st.container(border=True):
        left, right = st.columns([3, 1])
        left.write(
            "The file needs the same columns as the training data. "
            "The Churn column is not needed."
        )
        show_template_download(right)

        uploaded_file = st.file_uploader("Upload a CSV of customers", type="csv")
        if uploaded_file is None:
            st.info("Upload a CSV to get started. The template shows the expected columns.")
            st.stop()

        customers = clean_data(pd.read_csv(uploaded_file))

        missing = missing_columns(customers, need_target=False)
        if len(missing) > 0:
            st.error(f"This file is missing these columns: {', '.join(missing)}")
            st.stop()

        button_column, info_column = st.columns([1, 3])
        run_clicked = button_column.button("Run predictions", type="primary")
        info_column.markdown(f"Uploaded: **{uploaded_file.name}** · {len(customers):,} rows")

    return uploaded_file, customers, run_clicked


def predict_customers(model, customers: pd.DataFrame) -> pd.DataFrame:
    """
    Predict churn for every customer in the table.

    Returns a copy of the table with three new columns:
    "Churn probability", "Prediction" and "Risk level".
    The customers with the highest probability come first.
    """
    # Probability of churn for each customer (second column of predict_proba)
    probabilities = model.predict_proba(customers[FEATURES])[:, 1]

    result = customers.copy()

    # If the file has the real answer, keep it for comparison
    if TARGET in result.columns:
        result = result.rename(columns={TARGET: "Churn (actual)"})

    risk_levels = []
    for probability in probabilities:
        risk_levels.append(risk_level(probability))

    result["Churn probability"] = probabilities.round(4)
    result["Prediction"] = np.where(probabilities >= 0.5, "Yes", "No")
    result["Risk level"] = risk_levels

    result = result.sort_values("Churn probability", ascending=False)
    return result


# =============================================================================
# SUMMARY (F8)
# =============================================================================

def show_summary(result: pd.DataFrame) -> None:
    """Show how many customers are in each risk level."""
    st.subheader("Summary")

    number_of_customers = len(result)
    counts = result["Risk level"].value_counts()

    column_1, column_2, column_3, column_4 = st.columns(4)
    column_1.metric("Customers scored", f"{number_of_customers:,}")
    column_2.metric("High risk", f"{counts.get('High', 0):,}")
    column_3.metric("Medium risk", f"{counts.get('Medium', 0):,}")
    column_4.metric("Low risk", f"{counts.get('Low', 0):,}")

    # One stacked bar: the share of customers in each risk level
    figure = go.Figure()
    for level in RISK_LEVELS:
        share = counts.get(level, 0) / number_of_customers * 100
        figure.add_trace(go.Bar(
            x=[share],
            y=["Customers"],
            orientation="h",
            name=level,
            marker_color=RISK_COLORS[level],
            text=f"{share:.0f}%",
            textposition="inside",
            hovertemplate=f"{level}: %{{x:.1f}}%<extra></extra>",
        ))

    figure.update_layout(
        barmode="stack",
        height=130,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(range=[0, 100], ticksuffix="%"),
        yaxis=dict(visible=False),
        legend=dict(orientation="h", y=-0.6),
    )
    st.plotly_chart(figure)


# =============================================================================
# RESULTS TABLE (F7, F9)
# =============================================================================

def show_results_table(result: pd.DataFrame) -> None:
    """Show the predictions with a search box, a risk filter and a download button."""
    with st.container(border=True):
        st.markdown("**Results**")
        search_column, filter_column, download_column = st.columns([2, 2, 1])

        # Search box (only if the file has a customer ID column)
        search_text = ""
        if ID_COL in result.columns:
            search_text = search_column.text_input(
                "Search customerID",
                placeholder="e.g. 7590-VHVEG",
            )

        # Risk level filter
        selected_levels = filter_column.multiselect(
            "Risk level",
            RISK_LEVELS,
            default=RISK_LEVELS,
        )

        # Download button (F9): always downloads every row and column
        download_column.download_button(
            "Download results (CSV)",
            to_csv_bytes(result),
            file_name="churn_predictions.csv",
            mime="text/csv",
            type="primary",
        )

        # Apply the filter and the search
        visible_rows = result[result["Risk level"].isin(selected_levels)]
        if search_text:
            customer_ids = visible_rows[ID_COL].astype(str)
            is_match = customer_ids.str.contains(search_text, case=False, na=False)
            visible_rows = visible_rows[is_match]

        # Show only the main columns in the table
        visible_columns = []
        for column in RESULT_COLUMNS:
            if column in visible_rows.columns:
                visible_columns.append(column)

        st.dataframe(
            visible_rows[visible_columns],
            hide_index=True,
            column_config={
                "Churn probability": st.column_config.ProgressColumn(
                    "Churn probability",
                    format="percent",
                    min_value=0,
                    max_value=1,
                ),
                "MonthlyCharges": st.column_config.NumberColumn(
                    "MonthlyCharges",
                    format="$%.2f",
                ),
            },
        )
        st.caption(
            f"Showing {len(visible_rows):,} of {len(result):,} customers, highest risk first. "
            "The download includes every row and column."
        )


# =============================================================================
# PAGE
# =============================================================================

def main() -> None:
    """Draw the page from top to bottom."""
    model, metrics = require_model()

    st.title("Batch Prediction")
    st.caption("Upload many customers at once and get a churn prediction for every row.")

    # Step 1: upload a file
    uploaded_file, customers, run_clicked = show_upload_section()

    # This text identifies the uploaded file. It is used to check that
    # the saved results belong to the file that is uploaded now.
    file_key = f"{uploaded_file.name}-{uploaded_file.size}"

    # Step 2: predict when the user clicks the button.
    # The results are saved in st.session_state, so they stay on the page
    # when the user changes a filter.
    if run_clicked:
        st.session_state.batch_result = predict_customers(model, customers)
        st.session_state.batch_file = file_key

    # Step 3: stop here if there are no results for this file yet
    result = st.session_state.get("batch_result")
    saved_file_key = st.session_state.get("batch_file")
    if result is None or saved_file_key != file_key:
        st.stop()

    # Step 4: show the results
    show_summary(result)
    show_results_table(result)


main()
