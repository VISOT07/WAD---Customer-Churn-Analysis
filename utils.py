"""
utils.py
========
Shared settings and helper functions for the Customer Churn Prediction app.

This file is imported by train_model.py and by every page in the views/ folder.

Contents
--------
    1. File paths
    2. Column names
    3. Settings (risk levels and chart colors)
    4. Data cleaning functions
    5. Data and model loading functions
    6. Prediction helper functions
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# =============================================================================
# 1. FILE PATHS
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_PATH = BASE_DIR / "models" / "model.pkl"
METRICS_PATH = BASE_DIR / "models" / "metrics.json"


# =============================================================================
# 2. COLUMN NAMES (Telco Customer Churn dataset)
# =============================================================================

# The column the model predicts
TARGET = "Churn"

# The customer ID column (not used as a feature)
ID_COL = "customerID"

# Columns that contain numbers
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

# Columns that contain categories (text values)
CATEGORICAL_COLS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

# All input columns of the model
FEATURES = CATEGORICAL_COLS + NUMERIC_COLS

# Services that a customer can only have with an internet connection
INTERNET_ADDONS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

# Friendly column names shown in the app
LABELS = {
    "gender": "Gender",
    "SeniorCitizen": "Senior citizen",
    "Partner": "Partner",
    "Dependents": "Dependents",
    "tenure": "Tenure (months)",
    "PhoneService": "Phone service",
    "MultipleLines": "Multiple lines",
    "InternetService": "Internet service",
    "OnlineSecurity": "Online security",
    "OnlineBackup": "Online backup",
    "DeviceProtection": "Device protection",
    "TechSupport": "Tech support",
    "StreamingTV": "Streaming TV",
    "StreamingMovies": "Streaming movies",
    "Contract": "Contract",
    "PaperlessBilling": "Paperless billing",
    "PaymentMethod": "Payment method",
    "MonthlyCharges": "Monthly charges ($)",
    "TotalCharges": "Total charges ($)",
}


# =============================================================================
# 3. SETTINGS
# =============================================================================

# Risk levels, based on the churn probability:
#   Low    : probability below 40%
#   Medium : probability from 40% up to 70%
#   High   : probability of 70% or more
LOW_MAX = 0.40
HIGH_MIN = 0.70

# A customer with this many months or fewer counts as a new customer
NEW_CUSTOMER_MONTHS = 12

# Chart colors (blue and orange are easy to tell apart for most people)
CHURN_COLORS = {"Yes": "#E8590C", "No": "#1C7ED6"}
RISK_COLORS = {"High": "#E8590C", "Medium": "#FAB005", "Low": "#1C7ED6"}


# =============================================================================
# 4. DATA CLEANING FUNCTIONS
# =============================================================================

def convert_to_yes_no(column: pd.Series) -> pd.Series:
    """
    Convert a column to the values "Yes" and "No".

    The raw data can store the same answer in different ways
    (1/0, True/False, yes/no). This function makes them all the same.
    Missing values stay missing.
    """
    yes_no_map = {
        "1": "Yes",
        "0": "No",
        "1.0": "Yes",
        "0.0": "No",
        "True": "Yes",
        "False": "No",
    }

    text = column.astype(str).str.strip().str.capitalize()
    converted = text.replace(yes_no_map)
    converted = converted.where(column.notna(), np.nan)
    return converted


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix the known problems in the raw churn CSV file.

    Steps:
        1. Remove extra spaces from column names and text values.
        2. Convert the numeric columns to real numbers.
        3. Convert SeniorCitizen from 0/1 to No/Yes.
        4. Convert the target column (Churn) to Yes/No.
        5. Store text columns as normal Python text.

    Returns a cleaned copy. The original DataFrame is not changed.
    """
    df = df.copy()

    # Step 1: remove extra spaces
    df.columns = df.columns.str.strip()
    text_columns = df.select_dtypes(include=["object", "string"]).columns
    for column in text_columns:
        df[column] = df[column].str.strip()

    # Step 2: convert numeric columns to numbers.
    # TotalCharges has some blank values in the raw file.
    # errors="coerce" turns a blank value into a missing value (NaN).
    for column in NUMERIC_COLS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    # Step 3: SeniorCitizen is stored as 0/1, the other columns use No/Yes
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = convert_to_yes_no(df["SeniorCitizen"])

    # Step 4: the target column
    if TARGET in df.columns:
        df[TARGET] = convert_to_yes_no(df[TARGET])

    # Step 5: store text columns as normal Python text
    # (this works with every version of scikit-learn)
    text_columns = CATEGORICAL_COLS + [TARGET, ID_COL]
    for column in text_columns:
        if column in df.columns:
            is_present = df[column].notna()
            df[column] = df[column].astype(object).where(is_present, np.nan)

    return df


def missing_columns(df: pd.DataFrame, need_target: bool = True) -> list:
    """
    Return the names of the required columns that are not in the DataFrame.

    An empty list means the DataFrame has every column the app needs.
    Set need_target=False for new customers, who have no Churn column yet.
    """
    required_columns = list(FEATURES)
    if need_target:
        required_columns.append(TARGET)

    missing = []
    for column in required_columns:
        if column not in df.columns:
            missing.append(column)
    return missing


def churn_rate_by(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Calculate the churn rate for each value of one column.

    Example: churn_rate_by(df, "Contract") returns one row per contract type
    with two numbers: churn_rate (in percent) and customers (count).
    """
    table = df.copy()
    table["churned"] = (table[TARGET] == "Yes").astype(int)

    groups = table.groupby(column, observed=True)["churned"]

    result = pd.DataFrame()
    result["churn_rate"] = (groups.mean() * 100).round(1)
    result["customers"] = groups.count()
    result = result.reset_index()
    return result


# =============================================================================
# 5. DATA AND MODEL LOADING FUNCTIONS
# =============================================================================
# The @st.cache_... lines tell Streamlit to keep the result in memory,
# so the file is read only once and the app stays fast.

@st.cache_data
def load_default_data() -> pd.DataFrame:
    """Read the default dataset from the data/ folder and clean it."""
    raw_data = pd.read_csv(DATA_PATH)
    return clean_data(raw_data)


def get_data() -> pd.DataFrame:
    """
    Return the dataset that the app is using now.

    This is the file the user uploaded on the Home page.
    If the user did not upload a file, it is the default dataset.
    """
    uploaded_data = st.session_state.get("df")
    if uploaded_data is not None:
        return uploaded_data

    if not DATA_PATH.exists():
        relative_path = DATA_PATH.relative_to(BASE_DIR)
        st.error(f"Dataset not found. Put your CSV at `{relative_path}`.")
        st.stop()

    return load_default_data()


def get_data_name() -> str:
    """Return the file name of the dataset that the app is using now."""
    uploaded_name = st.session_state.get("data_name")
    if uploaded_name:
        return uploaded_name
    return DATA_PATH.name


@st.cache_resource
def load_model():
    """Load the trained model. Returns None if it was not trained yet."""
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    """Load the training results. Returns None if there are none yet."""
    if not METRICS_PATH.exists():
        return None
    with open(METRICS_PATH, encoding="utf-8") as file:
        return json.load(file)


def require_model():
    """
    Return the model and the training results.

    If the model was not trained yet, show an error message and stop the page.
    """
    model = load_model()
    metrics = load_metrics()

    if model is None or metrics is None:
        st.error("No trained model found. Run `python train_model.py` first, then refresh this page.")
        st.stop()

    return model, metrics


# =============================================================================
# 6. PREDICTION HELPER FUNCTIONS
# =============================================================================

def risk_level(probability: float) -> str:
    """Convert a churn probability (0 to 1) to "Low", "Medium" or "High"."""
    if probability >= HIGH_MIN:
        return "High"
    if probability >= LOW_MAX:
        return "Medium"
    return "Low"


def risk_badge(risk: str) -> str:
    """Return a colored text label for a risk level (Streamlit markdown)."""
    badge_colors = {"High": "red", "Medium": "orange", "Low": "blue"}
    color = badge_colors[risk]
    return f":{color}-background[**{risk} risk**]"


def fix_service_values(values: dict) -> dict:
    """
    Make the service answers of one customer consistent.

    Rule 1: a customer without internet cannot have internet add-ons,
            so every add-on becomes "No internet service".
    Rule 2: a customer without phone service cannot have multiple lines,
            so MultipleLines becomes "No phone service".

    Returns a corrected copy of the values.
    """
    fixed = dict(values)

    # Rule 1: internet add-ons
    has_internet = fixed.get("InternetService") != "No"
    for column in INTERNET_ADDONS:
        if not has_internet:
            fixed[column] = "No internet service"
        elif fixed.get(column) == "No internet service":
            fixed[column] = "No"

    # Rule 2: multiple lines
    has_phone = fixed.get("PhoneService") != "No"
    if not has_phone:
        fixed["MultipleLines"] = "No phone service"
    elif fixed.get("MultipleLines") == "No phone service":
        fixed["MultipleLines"] = "No"

    return fixed


def format_value(value) -> str:
    """Format one customer value for display, for example 70.50 -> "70.5"."""
    if isinstance(value, float):
        text = f"{value:,.2f}"
        text = text.rstrip("0").rstrip(".")
        return text
    return str(value)


def explain_prediction(model, customer: pd.DataFrame, background: dict, top_n: int = 6) -> pd.DataFrame:
    """
    Explain the prediction for one customer with "what-if" questions.

    For each column we ask: "What if this customer had the values that
    other customers usually have?" Then we compare the churn probability.

        change > 0 : the customer's value RAISES the churn risk
        change < 0 : the customer's value LOWERS the churn risk

    Args:
        model:      the trained model
        customer:   a DataFrame with one row (the customer)
        background: the typical values of each column and how common they are
                    (created by train_model.py)
        top_n:      how many columns to return

    Returns:
        A DataFrame with the top_n columns that change the prediction most.
        "change" is in percentage points.
    """
    customer = customer[FEATURES].reset_index(drop=True)
    customer_values = customer.iloc[0].to_dict()

    # Step 1: the churn probability of the real customer
    customer_probability = model.predict_proba(customer)[0, 1]

    # Step 2: build the "what-if" customers.
    # Each one is a copy of the real customer with ONE value replaced.
    what_if_customers = []
    what_if_columns = []
    what_if_weights = []

    for column in FEATURES:
        typical_values = background[column]["values"]
        typical_weights = background[column]["weights"]

        for value, weight in zip(typical_values, typical_weights):
            what_if_customer = dict(customer_values)
            what_if_customer[column] = value

            what_if_customers.append(what_if_customer)
            what_if_columns.append(column)
            what_if_weights.append(weight)

    # Step 3: predict all "what-if" customers in one call (this is fast)
    what_if_table = pd.DataFrame(what_if_customers, columns=FEATURES)
    what_if_probabilities = model.predict_proba(what_if_table)[:, 1]

    predictions = pd.DataFrame({
        "column": what_if_columns,
        "weight": what_if_weights,
        "probability": what_if_probabilities,
    })

    # Step 4: for each column, compare the real probability with the
    # average "what-if" probability (common values count more)
    rows = []
    for column in FEATURES:
        column_predictions = predictions[predictions["column"] == column]
        weighted_sum = (column_predictions["probability"] * column_predictions["weight"]).sum()
        typical_probability = weighted_sum / column_predictions["weight"].sum()

        change = (customer_probability - typical_probability) * 100
        if change > 0:
            effect = "Raises risk"
        else:
            effect = "Lowers risk"

        column_label = LABELS.get(column, column)
        value_text = format_value(customer[column].iloc[0])

        rows.append({
            "feature": column,
            "label": f"{column_label}: {value_text}",
            "change": round(change, 1),
            "effect": effect,
        })

    explanation = pd.DataFrame(rows)

    # Step 5: keep the columns with the biggest change
    explanation["size"] = explanation["change"].abs()
    explanation = explanation[explanation["size"] >= 0.5]
    explanation = explanation.sort_values("size", ascending=False).head(top_n)

    # A horizontal bar chart draws the last row on top,
    # so we put the biggest change last.
    explanation = explanation.sort_values("size", ascending=True)
    explanation = explanation.drop(columns="size")
    return explanation


def retention_suggestions(values: dict, risk: str, high_monthly_charge: float) -> list:
    """
    Suggest actions to keep one customer, using simple rules.

    Args:
        values:              the customer's details (column name -> value)
        risk:                "Low", "Medium" or "High"
        high_monthly_charge: a monthly charge at or above this is "high"

    Returns:
        A list of suggestions. The list is empty for Low risk customers.
    """
    if risk == "Low":
        return []

    tenure = float(values.get("tenure") or 0)
    monthly_charges = float(values.get("MonthlyCharges") or 0)
    has_internet = values.get("InternetService") != "No"
    no_tech_support = values.get("TechSupport") == "No"
    no_online_security = values.get("OnlineSecurity") == "No"

    suggestions = []

    if values.get("Contract") == "Month-to-month":
        suggestions.append(
            "Offer a discount to switch from month-to-month to a 1-year or 2-year contract."
        )

    if has_internet and (no_tech_support or no_online_security):
        suggestions.append("Offer a free trial of Tech support or Online security.")

    if values.get("PaymentMethod") == "Electronic check":
        suggestions.append(
            "Encourage automatic payment (bank transfer or credit card), for example with a small bonus."
        )

    if tenure <= NEW_CUSTOMER_MONTHS:
        suggestions.append("New customer: send a welcome check-in and explain the plan's benefits.")

    if monthly_charges >= high_monthly_charge:
        suggestions.append(
            "Review the plan and offer a bundle or loyalty discount to lower the monthly bill."
        )

    if risk == "High":
        suggestions.append("Schedule a follow-up call from the retention team this week.")

    if len(suggestions) == 0:
        suggestions.append("Contact the customer to ask about their experience and any problems.")

    return suggestions
