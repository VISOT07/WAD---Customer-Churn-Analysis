"""
train_model.py
==============
Train the churn models and save the best one for the Streamlit app.

How to run (once before starting the app, and again when the data changes):

    python train_model.py

Steps of this script
--------------------
    1. Load and clean the dataset.
    2. Split the data into a training set and a test set.
    3. Train four models.
    4. Evaluate each model on the test set.
    5. Choose the best model.
    6. Save the best model and the results.

Output files
------------
    models/model.pkl      the best model (a full scikit-learn pipeline)
    models/metrics.json   the results of every model, used by the app pages
"""

from datetime import datetime
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from streamlit.logger import set_log_level
from xgboost import XGBClassifier

# Hide Streamlit messages, because this script runs without the Streamlit app
set_log_level("error")

from utils import (
    CATEGORICAL_COLS,
    DATA_PATH,
    FEATURES,
    METRICS_PATH,
    MODEL_PATH,
    NUMERIC_COLS,
    TARGET,
    clean_data,
    missing_columns,
)


# =============================================================================
# SETTINGS
# =============================================================================

TEST_SIZE = 0.2          # 20% of the data is kept for testing
RANDOM_STATE = 42        # fixed number, so the results are the same every run
BEST_BY = "f1"           # the metric used to choose the best model
MAX_ROC_POINTS = 200     # maximum number of points saved for the ROC curve


# =============================================================================
# STEP 1: LOAD THE DATA
# =============================================================================

def load_dataset() -> pd.DataFrame:
    """Read the CSV file, clean it and check that it has every needed column."""
    print(f"Loading {DATA_PATH} ...")
    raw_data = pd.read_csv(DATA_PATH)
    df = clean_data(raw_data)

    missing = missing_columns(df)
    if len(missing) > 0:
        raise SystemExit(f"These columns are missing from the dataset: {missing}")

    # A row without a Churn value cannot be used for training
    df = df.dropna(subset=[TARGET])
    return df


# =============================================================================
# STEP 3: BUILD THE MODELS
# =============================================================================

def build_preprocessor() -> ColumnTransformer:
    """
    Build the preprocessing step. It prepares the data for the model.

    Numeric columns:
        1. Fill missing values with the median.
        2. Scale the values (mean 0, standard deviation 1).

    Categorical columns:
        1. Fill missing values with the most common value.
        2. One-hot encode (one 0/1 column for each category).
    """
    numeric_steps = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_steps = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_steps, NUMERIC_COLS),
        ("cat", categorical_steps, CATEGORICAL_COLS),
    ])
    return preprocessor


def build_models(positive_weight: float) -> dict:
    """
    Create the four models to compare.

    The data is imbalanced: far fewer customers churn than stay.
    The class weights tell each model to pay more attention to churners.

    Args:
        positive_weight: number of non-churners divided by number of churners

    Returns:
        A dictionary: model name -> model (not trained yet)
    """
    logistic_regression = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
    )

    # Same settings as the team notebook (customer_churn_V2):
    # max_depth=5 keeps the tree small, so it does not memorise the training data
    decision_tree = DecisionTreeClassifier(
        max_depth=5,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    random_forest = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    xgboost = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=positive_weight,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    models = {
        "Logistic Regression": logistic_regression,
        "Decision Tree": decision_tree,
        "Random Forest": random_forest,
        "XGBoost": xgboost,
    }
    return models


# =============================================================================
# STEP 4: EVALUATE A MODEL
# =============================================================================

def round_list(values, digits: int = 4) -> list:
    """Round every number in a list (keeps the JSON file small)."""
    rounded = []
    for value in values:
        rounded.append(round(float(value), digits))
    return rounded


def calculate_roc_curve(y_test, probabilities) -> dict:
    """Calculate the points of the ROC curve."""
    false_positive_rate, true_positive_rate, _ = roc_curve(y_test, probabilities)

    # Keep at most MAX_ROC_POINTS points, evenly spread along the curve
    number_of_points = len(false_positive_rate)
    if number_of_points > MAX_ROC_POINTS:
        positions = np.linspace(0, number_of_points - 1, MAX_ROC_POINTS).astype(int)
        false_positive_rate = false_positive_rate[positions]
        true_positive_rate = true_positive_rate[positions]

    return {
        "fpr": round_list(false_positive_rate),
        "tpr": round_list(true_positive_rate),
    }


def calculate_feature_importance(pipeline, X_test, y_test) -> dict:
    """
    Calculate the permutation importance of each column.

    Method: shuffle the values of one column and measure how much the
    ROC-AUC drops. A bigger drop means the column is more important.

    Returns a dictionary: column name -> importance (most important first).
    """
    result = permutation_importance(
        pipeline,
        X_test,
        y_test,
        scoring="roc_auc",
        n_repeats=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    importance = pd.Series(result.importances_mean, index=FEATURES)
    importance = importance.sort_values(ascending=False).round(4)
    return importance.to_dict()


def evaluate(pipeline, X_test, y_test) -> dict:
    """
    Evaluate one trained model on the test set.

    Returns a dictionary with the metrics, the confusion matrix,
    the ROC curve and the feature importance.
    """
    # Probability of churn for each test customer
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    # A probability of 50% or more is predicted as churn (1)
    predictions = (probabilities >= 0.5).astype(int)

    results = {
        "accuracy": round(accuracy_score(y_test, predictions), 4),
        "precision": round(precision_score(y_test, predictions), 4),
        "recall": round(recall_score(y_test, predictions), 4),
        "f1": round(f1_score(y_test, predictions), 4),
        "roc_auc": round(roc_auc_score(y_test, probabilities), 4),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "roc_curve": calculate_roc_curve(y_test, probabilities),
        "importance": calculate_feature_importance(pipeline, X_test, y_test),
    }
    return results


# =============================================================================
# STEP 5: CHOOSE THE BEST MODEL
# =============================================================================

def choose_best_model(results: dict) -> str:
    """Return the name of the model with the highest BEST_BY score."""
    best_name = None
    best_score = -1.0

    for name, model_results in results.items():
        score = model_results[BEST_BY]
        if score > best_score:
            best_name = name
            best_score = score

    return best_name


# =============================================================================
# STEP 6: PREPARE THE EXTRA INFORMATION FOR THE APP
# =============================================================================

def build_background(X_train: pd.DataFrame) -> dict:
    """
    Collect the typical values of each column.

    The "Why this prediction" chart in the app uses these values.

    Categorical column: every category, with its share of customers.
    Numeric column:     10 values spread evenly from low to high.
    """
    background = {}

    for column in CATEGORICAL_COLS:
        shares = X_train[column].dropna().astype(str).value_counts(normalize=True)
        background[column] = {
            "values": shares.index.tolist(),
            "weights": round_list(shares.values),
        }

    for column in NUMERIC_COLS:
        # The 5th, 15th, 25th, ... 95th percentiles
        percentiles = np.arange(0.05, 1.0, 0.1)
        typical_values = X_train[column].quantile(percentiles)
        background[column] = {
            "values": round_list(typical_values.values, digits=2),
            "weights": [0.1] * len(typical_values),
        }

    return background


def build_form_options(X: pd.DataFrame) -> dict:
    """Collect the choices of each categorical column for the prediction form."""
    options = {}
    for column in CATEGORICAL_COLS:
        unique_values = X[column].dropna().astype(str).unique().tolist()
        options[column] = sorted(unique_values)
    return options


def build_numeric_ranges(X: pd.DataFrame) -> dict:
    """Collect the minimum, maximum and median of each numeric column."""
    ranges = {}
    for column in NUMERIC_COLS:
        ranges[column] = {
            "min": float(X[column].min()),
            "max": float(X[column].max()),
            "median": float(X[column].median()),
        }
    return ranges


# =============================================================================
# MAIN PROGRAM
# =============================================================================

def main() -> None:
    """Run all the steps: load, split, train, evaluate, choose, save."""

    # ----- Step 1: load the data -----
    df = load_dataset()
    X = df[FEATURES]
    y = (df[TARGET] == "Yes").astype(int)   # 1 = churned, 0 = stayed
    print(f"{len(df):,} customers, churn rate {y.mean():.1%}")

    # ----- Step 2: split into training and test sets -----
    # stratify=y keeps the same churn rate in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    number_stayed = (y_train == 0).sum()
    number_churned = (y_train == 1).sum()
    positive_weight = float(number_stayed / number_churned)

    # ----- Steps 3 and 4: train and evaluate each model -----
    models = build_models(positive_weight)
    results = {}
    trained_pipelines = {}

    for name, model in models.items():
        print(f"Training {name} ...")

        # A pipeline joins the preprocessing and the model in one object
        pipeline = Pipeline([
            ("prep", build_preprocessor()),
            ("model", model),
        ])
        pipeline.fit(X_train, y_train)

        model_results = evaluate(pipeline, X_test, y_test)
        results[name] = model_results
        trained_pipelines[name] = pipeline

        print(
            f"  accuracy {model_results['accuracy']:.3f} | "
            f"precision {model_results['precision']:.3f} | "
            f"recall {model_results['recall']:.3f} | "
            f"f1 {model_results['f1']:.3f} | "
            f"roc_auc {model_results['roc_auc']:.3f}"
        )

    # ----- Step 5: choose the best model -----
    best_name = choose_best_model(results)
    print(f"Best model by {BEST_BY}: {best_name}")

    # ----- Step 6: save the best model and the results -----
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(trained_pipelines[best_name], MODEL_PATH)

    metrics = {
        "best_model": best_name,
        "best_by": BEST_BY,
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "test_size": TEST_SIZE,
        "n_rows": int(len(df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "churn_rate": round(float(y.mean()), 4),
        "models": results,
        "background": build_background(X_train),
        "options": build_form_options(X),
        "numeric_ranges": build_numeric_ranges(X),
        "high_monthly_charge": float(X_train["MonthlyCharges"].quantile(0.75)),
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print(f"Saved {MODEL_PATH} and {METRICS_PATH}")


if __name__ == "__main__":
    main()
