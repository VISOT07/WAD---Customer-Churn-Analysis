"""
Page 4: Predict a Customer
==========================
Features:
    F5   Single customer prediction - form, prediction and probability
    F6   Prediction explanation     - which values raise or lower the risk
    F8   Risk level                 - Low / Medium / High
    F10  Retention suggestions      - actions for Medium and High risk
"""

import pandas as pd
import plotly.express as px
import streamlit as st

from utils import (
    CHURN_COLORS,
    FEATURES,
    HIGH_MIN,
    LABELS,
    LOW_MAX,
    explain_prediction,
    fix_service_values,
    require_model,
    retention_suggestions,
    risk_badge,
    risk_level,
)

# The six internet add-on services, shown three in each row of the form
ADDON_COLUMNS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


# =============================================================================
# INPUT FORM (F5)
# =============================================================================

def select_category(container, column: str, options: dict, default: str = None) -> str:
    """
    Show a dropdown list for one categorical column.

    Args:
        container: the Streamlit column where the dropdown is placed
        column:    the dataset column name, for example "Contract"
        options:   the choices of every column (from the training results)
        default:   the choice selected at the start

    Returns:
        The choice the user selected.
    """
    choices = options[column]

    if default in choices:
        default_index = choices.index(default)
    else:
        default_index = 0

    return container.selectbox(
        LABELS[column],
        choices,
        index=default_index,
        key=f"in_{column}",
    )


def show_input_form(options: dict, ranges: dict):
    """
    Show the form with the 19 customer details.

    Returns:
        values:    a dictionary, column name -> value entered by the user
        submitted: True if the user clicked "Predict churn"
    """
    values = {}

    with st.form("predict_form", border=True):

        # ----- Group 1: customer -----
        st.markdown("**Customer**")
        columns = st.columns(4)
        values["gender"] = select_category(columns[0], "gender", options)
        values["SeniorCitizen"] = select_category(columns[1], "SeniorCitizen", options, "No")
        values["Partner"] = select_category(columns[2], "Partner", options, "No")
        values["Dependents"] = select_category(columns[3], "Dependents", options, "No")

        # ----- Group 2: account -----
        st.markdown("**Account**")
        columns = st.columns(3)
        values["tenure"] = columns[0].number_input(
            LABELS["tenure"],
            min_value=0,
            max_value=int(ranges["tenure"]["max"]) * 2,
            value=int(ranges["tenure"]["median"]),
            step=1,
            key="in_tenure",
        )
        values["Contract"] = select_category(columns[1], "Contract", options, "Month-to-month")
        values["PaperlessBilling"] = select_category(columns[2], "PaperlessBilling", options, "Yes")

        columns = st.columns(3)
        values["PaymentMethod"] = select_category(columns[0], "PaymentMethod", options, "Electronic check")
        values["MonthlyCharges"] = columns[1].number_input(
            LABELS["MonthlyCharges"],
            min_value=0.0,
            value=round(ranges["MonthlyCharges"]["median"], 2),
            step=1.0,
            format="%.2f",
            key="in_MonthlyCharges",
        )
        values["TotalCharges"] = columns[2].number_input(
            LABELS["TotalCharges"],
            min_value=0.0,
            value=round(ranges["TotalCharges"]["median"], 2),
            step=10.0,
            format="%.2f",
            key="in_TotalCharges",
            help="Usually close to tenure × monthly charges.",
        )

        # ----- Group 3: services -----
        st.markdown("**Services**")
        columns = st.columns(3)
        values["PhoneService"] = select_category(columns[0], "PhoneService", options, "Yes")
        values["MultipleLines"] = select_category(columns[1], "MultipleLines", options, "No")
        values["InternetService"] = select_category(columns[2], "InternetService", options, "Fiber optic")

        columns = st.columns(3)
        for position, column in enumerate(ADDON_COLUMNS):
            container = columns[position % 3]   # position 0, 1, 2, then 0, 1, 2 again
            values[column] = select_category(container, column, options, "No")

        submitted = st.form_submit_button("Predict churn", type="primary")

    return values, submitted


def save_customer(values: dict) -> None:
    """
    Correct the service answers and save the customer in st.session_state.

    Saving the customer keeps the result on the page when Streamlit
    runs the script again.
    """
    fixed_values = fix_service_values(values)

    # Remember which answers were corrected, to tell the user
    changed_labels = []
    for column in values:
        if values[column] != fixed_values[column]:
            changed_labels.append(LABELS[column])

    st.session_state.last_customer = fixed_values
    st.session_state.last_changed = changed_labels


# =============================================================================
# RESULT (F5, F6, F8)
# =============================================================================

def show_prediction(probability: float, risk: str) -> None:
    """Show the prediction, the risk level and the churn probability."""
    if probability >= 0.5:
        prediction_text = "Likely to churn"
    else:
        prediction_text = "Likely to stay"

    with st.container(border=True):
        st.caption("Prediction")
        st.markdown(f"## {prediction_text}")
        st.markdown(risk_badge(risk))
        st.metric("Churn probability", f"{probability:.0%}")
        st.progress(probability)
        st.caption(
            f"Low < {LOW_MAX:.0%} · Medium {LOW_MAX:.0%}–{HIGH_MIN:.0%} · High ≥ {HIGH_MIN:.0%}"
        )


def show_explanation(model, customer_row: pd.DataFrame, background: dict) -> None:
    """Show a bar chart of the values that raise or lower this customer's risk."""
    with st.container(border=True):
        st.markdown("**Why this prediction**")

        explanation = explain_prediction(model, customer_row, background)

        if explanation.empty:
            st.write("No single value changes the prediction much for this customer.")
        else:
            figure = px.bar(
                explanation,
                x="change",
                y="label",
                orientation="h",
                color="effect",
                color_discrete_map={
                    "Raises risk": CHURN_COLORS["Yes"],
                    "Lowers risk": CHURN_COLORS["No"],
                },
                labels={
                    "change": "Change in churn probability (percentage points)",
                    "label": "",
                },
            )
            figure.update_layout(
                height=300,
                margin=dict(l=10, r=10, t=10, b=10),
                legend_title_text="",
                legend=dict(orientation="h", y=-0.3),
            )
            st.plotly_chart(figure)

        st.caption(
            "Compared with what other customers usually have. "
            "Right = pushes toward churn, left = pushes toward staying."
        )


# =============================================================================
# SUGGESTED ACTIONS (F10)
# =============================================================================

def show_suggestions(customer: dict, risk: str, high_monthly_charge: float) -> None:
    """Show a numbered list of actions to keep the customer."""
    with st.container(border=True):
        st.markdown("**Suggested actions**")

        suggestions = retention_suggestions(customer, risk, high_monthly_charge)

        if len(suggestions) == 0:
            st.success("Low risk. No action needed right now.")
            return

        numbered_lines = []
        for number, suggestion in enumerate(suggestions, start=1):
            numbered_lines.append(f"{number}. {suggestion}")

        st.markdown("\n".join(numbered_lines))
        st.caption("Rule-based tips, shown for Medium and High risk customers.")


# =============================================================================
# PAGE
# =============================================================================

def main() -> None:
    """Draw the page from top to bottom."""
    model, metrics = require_model()

    st.title("Predict a Customer")
    st.caption("Enter one customer's details to get their churn prediction and risk level.")

    # Step 1: the input form
    values, submitted = show_input_form(metrics["options"], metrics["numeric_ranges"])
    if submitted:
        save_customer(values)

    # Step 2: stop here if the user did not ask for a prediction yet
    customer = st.session_state.get("last_customer")
    if customer is None:
        st.info("Fill in the form and click **Predict churn** to see the result.")
        st.stop()

    changed_labels = st.session_state.get("last_changed")
    if changed_labels:
        st.caption(
            "Adjusted to match the chosen phone/internet service: " + ", ".join(changed_labels)
        )

    # Step 3: predict.
    # predict_proba returns two numbers for each customer:
    # [probability of staying, probability of churning]. We take the second.
    customer_row = pd.DataFrame([customer])[FEATURES]
    probability = float(model.predict_proba(customer_row)[0, 1])
    risk = risk_level(probability)

    # Step 4: show the result
    st.subheader("Result")
    left, right = st.columns(2)
    with left:
        show_prediction(probability, risk)
    with right:
        show_explanation(model, customer_row, metrics["background"])

    # Step 5: show the suggested actions
    show_suggestions(customer, risk, metrics["high_monthly_charge"])


main()
