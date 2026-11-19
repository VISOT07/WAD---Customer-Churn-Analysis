"""
Page 2: Churn Insights
======================
Features:
    F3  Churn insights dashboard - filters, summary numbers and 5 charts
"""

import pandas as pd
import plotly.express as px
import streamlit as st

from utils import CHURN_COLORS, TARGET, churn_rate_by, get_data, get_data_name

# The filter choice that means "do not filter this column"
ALL = "All"


# =============================================================================
# FILTERS
# =============================================================================

def get_filter_choices(df: pd.DataFrame, column: str) -> list:
    """Return the choices of one filter: "All" and every value of the column."""
    values = sorted(df[column].dropna().unique())
    return [ALL] + values


def show_filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Show the four filters and return the customers that match them.

    Filters: contract, internet service, payment method and tenure range.
    """
    shortest_tenure = int(df["tenure"].min())
    longest_tenure = int(df["tenure"].max())

    with st.container(border=True):
        column_1, column_2, column_3, column_4 = st.columns(4)

        contract = column_1.selectbox("Contract", get_filter_choices(df, "Contract"))
        internet = column_2.selectbox("Internet service", get_filter_choices(df, "InternetService"))
        payment = column_3.selectbox("Payment method", get_filter_choices(df, "PaymentMethod"))
        tenure_range = column_4.slider(
            "Tenure (months)",
            min_value=shortest_tenure,
            max_value=longest_tenure,
            value=(shortest_tenure, longest_tenure),
        )

    # Apply the filters one by one
    filtered = df[df["tenure"].between(tenure_range[0], tenure_range[1])]

    if contract != ALL:
        filtered = filtered[filtered["Contract"] == contract]

    if internet != ALL:
        filtered = filtered[filtered["InternetService"] == internet]

    if payment != ALL:
        filtered = filtered[filtered["PaymentMethod"] == payment]

    return filtered


# =============================================================================
# SUMMARY NUMBERS
# =============================================================================

def show_summary_numbers(data: pd.DataFrame) -> None:
    """Show four numbers about the customers who churned."""
    churned = data[data[TARGET] == "Yes"]

    number_of_customers = len(data)
    number_churned = len(churned)
    churn_rate = number_churned / number_of_customers

    # The averages cannot be calculated when nobody churned
    if number_churned > 0:
        average_tenure = f"{churned['tenure'].mean():.0f} mo"
        average_charge = f"${churned['MonthlyCharges'].mean():.2f}"
    else:
        average_tenure = "–"
        average_charge = "–"

    column_1, column_2, column_3, column_4 = st.columns(4)
    column_1.metric("Churn rate", f"{churn_rate:.1%}")
    column_2.metric(
        "Churned customers",
        f"{number_churned:,}",
        help=f"Out of {number_of_customers:,} customers",
    )
    column_3.metric("Avg tenure (churned)", average_tenure)
    column_4.metric("Avg monthly charge (churned)", average_charge)


# =============================================================================
# CHARTS
# =============================================================================

def display_chart(figure, height: int = 320) -> None:
    """Give a chart the standard size and margins, then show it."""
    figure.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=10, b=10),
        legend_title_text="",
    )
    st.plotly_chart(figure)


def show_churn_share_chart(data: pd.DataFrame) -> None:
    """Chart 1: donut chart of customers who churned and who stayed."""
    with st.container(border=True):
        st.markdown("**Churned vs stayed**")

        # Count the customers with Churn = "Yes" and Churn = "No"
        value_counts = data[TARGET].value_counts()
        counts = pd.DataFrame({
            TARGET: value_counts.index,
            "customers": value_counts.values,
        })
        counts["label"] = counts[TARGET].map({"Yes": "Churned", "No": "Stayed"})

        figure = px.pie(
            counts,
            names="label",
            values="customers",
            hole=0.55,
            color=TARGET,
            color_discrete_map=CHURN_COLORS,
        )
        figure.update_traces(textinfo="percent+label")
        display_chart(figure)


def show_contract_chart(data: pd.DataFrame) -> None:
    """Chart 2: bar chart of the churn rate for each contract type."""
    with st.container(border=True):
        st.markdown("**Churn rate by contract type**")

        churn_by_contract = churn_rate_by(data, "Contract")

        figure = px.bar(
            churn_by_contract,
            x="Contract",
            y="churn_rate",
            text="churn_rate",
            labels={"churn_rate": "Churn rate (%)"},
            color_discrete_sequence=[CHURN_COLORS["Yes"]],
        )
        figure.update_traces(texttemplate="%{text}%")
        display_chart(figure)


def show_tenure_chart(data: pd.DataFrame) -> None:
    """Chart 3: histogram of tenure, for churned and stayed customers."""
    with st.container(border=True):
        st.markdown("**Tenure distribution**")

        figure = px.histogram(
            data,
            x="tenure",
            color=TARGET,
            nbins=24,
            barmode="overlay",
            opacity=0.75,
            color_discrete_map=CHURN_COLORS,
            labels={"tenure": "Tenure (months)", TARGET: "Churned"},
        )
        figure.update_layout(yaxis_title="Customers")
        display_chart(figure)


def show_monthly_charges_chart(data: pd.DataFrame) -> None:
    """Chart 4: box plot of monthly charges, for churned and stayed customers."""
    with st.container(border=True):
        st.markdown("**Monthly charges by churn**")

        figure = px.box(
            data,
            x=TARGET,
            y="MonthlyCharges",
            color=TARGET,
            color_discrete_map=CHURN_COLORS,
            labels={TARGET: "Churned", "MonthlyCharges": "Monthly charges ($)"},
        )
        figure.update_layout(showlegend=False)
        display_chart(figure)


def show_payment_chart(data: pd.DataFrame) -> None:
    """Chart 5: bar chart of the churn rate for each payment method."""
    with st.container(border=True):
        st.markdown("**Churn rate by payment method**")

        churn_by_payment = churn_rate_by(data, "PaymentMethod")
        churn_by_payment = churn_by_payment.sort_values("churn_rate")

        figure = px.bar(
            churn_by_payment,
            x="churn_rate",
            y="PaymentMethod",
            orientation="h",
            text="churn_rate",
            labels={"churn_rate": "Churn rate (%)", "PaymentMethod": ""},
            color_discrete_sequence=[CHURN_COLORS["Yes"]],
        )
        figure.update_traces(texttemplate="%{text}%")
        display_chart(figure, height=280)


# =============================================================================
# PAGE
# =============================================================================

def main() -> None:
    """Draw the page from top to bottom."""
    df = get_data()

    st.title("Churn Insights")
    st.caption(
        f"Explore which customers leave and what they have in common · data: {get_data_name()}"
    )

    # Filters
    data = show_filters(df)
    if data.empty:
        st.warning("No customers match these filters.")
        st.stop()

    # Summary numbers
    show_summary_numbers(data)

    # Charts, two in each row
    left, right = st.columns(2)
    with left:
        show_churn_share_chart(data)
    with right:
        show_contract_chart(data)

    left, right = st.columns(2)
    with left:
        show_tenure_chart(data)
    with right:
        show_monthly_charges_chart(data)

    show_payment_chart(data)


main()
