"""
Page 3: Model Performance
=========================
Features:
    F4  Model performance - comparison table, confusion matrix,
                            ROC curve and feature importance
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils import CHURN_COLORS, LABELS, require_model

# Style of the best model's row in the comparison table
BEST_ROW_STYLE = "background-color: rgba(28, 126, 214, 0.15); font-weight: 600"

# Number of columns shown in the feature importance chart
TOP_FEATURES = 10


# =============================================================================
# TRAINING INFORMATION
# =============================================================================

def show_training_info(metrics: dict) -> None:
    """Show the test set size, the target and the best model."""
    best_model = metrics["best_model"]
    best_metric = metrics["best_by"].upper()

    with st.container(border=True):
        column_1, column_2, column_3 = st.columns(3)
        column_1.markdown(
            f"**Test set:** {metrics['test_size']:.0%} of data ({metrics['n_test']:,} customers)"
        )
        column_2.markdown("**Target:** Churn (Yes / No)")
        column_3.markdown(f"**Best model:** {best_model} (highest {best_metric})")


# =============================================================================
# COMPARISON TABLE
# =============================================================================

def build_comparison_table(results: dict) -> pd.DataFrame:
    """Build a table with one row for each model and one column for each metric."""
    rows = []
    for model_name, model_results in results.items():
        rows.append({
            "Model": model_name,
            "Accuracy": model_results["accuracy"],
            "Precision": model_results["precision"],
            "Recall": model_results["recall"],
            "F1-score": model_results["f1"],
            "ROC-AUC": model_results["roc_auc"],
        })
    return pd.DataFrame(rows)


def highlight_best_model(row: pd.Series, best_model: str) -> list:
    """Return the cell styles of one table row: highlighted for the best model."""
    if row["Model"] == best_model:
        return [BEST_ROW_STYLE] * len(row)
    return [""] * len(row)


def show_comparison_table(results: dict, best_model: str) -> None:
    """Show the model comparison table and explain the metrics."""
    with st.container(border=True):
        st.markdown("**Model comparison**")

        table = build_comparison_table(results)
        styled_table = table.style.apply(highlight_best_model, best_model=best_model, axis=1)
        styled_table = styled_table.format(precision=3)

        st.dataframe(styled_table, hide_index=True)
        st.caption("The highlighted model is saved and used for all predictions in this app.")

        with st.expander("What do these numbers mean?"):
            st.markdown(
                "- **Accuracy**: share of all predictions that are correct.\n"
                "- **Precision**: of the customers predicted to churn, how many really churned.\n"
                "- **Recall**: of the customers who really churned, how many the model found.\n"
                "- **F1-score**: balance between precision and recall.\n"
                "- **ROC-AUC**: how well the model ranks churners above non-churners "
                "(0.5 = random, 1 = perfect).\n\n"
                "For churn, **recall** matters a lot: missing a customer who leaves "
                "usually costs more than a false alarm."
            )


# =============================================================================
# CHARTS FOR ONE MODEL
# =============================================================================

def show_confusion_matrix(model_results: dict) -> None:
    """Show the confusion matrix: correct and wrong predictions on the test set."""
    with st.container(border=True):
        st.markdown("**Confusion matrix**")

        figure = px.imshow(
            model_results["confusion_matrix"],
            x=["Predicted: No", "Predicted: Yes"],
            y=["Actual: No", "Actual: Yes"],
            text_auto=True,
            color_continuous_scale="Blues",
        )
        figure.update_layout(
            height=340,
            margin=dict(l=10, r=10, t=10, b=10),
            coloraxis_showscale=False,
        )
        figure.update_traces(textfont_size=18)
        st.plotly_chart(figure)


def show_roc_curve(model_results: dict, model_name: str) -> None:
    """Show the ROC curve of the model and the line of a random guess."""
    with st.container(border=True):
        st.markdown(f"**ROC curve** · AUC = {model_results['roc_auc']:.3f}")

        figure = go.Figure()

        # Line 1: the model
        figure.add_trace(go.Scatter(
            x=model_results["roc_curve"]["fpr"],
            y=model_results["roc_curve"]["tpr"],
            mode="lines",
            name=model_name,
            line=dict(color=CHURN_COLORS["No"], width=3),
        ))

        # Line 2: a random guess (the diagonal)
        figure.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random guess",
            line=dict(color="gray", dash="dash"),
        ))

        figure.update_layout(
            height=340,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="False positive rate",
            yaxis_title="True positive rate",
            legend=dict(x=0.55, y=0.08),
        )
        st.plotly_chart(figure)


def show_feature_importance(model_results: dict) -> None:
    """Show the most important columns of the model as a bar chart."""
    with st.container(border=True):
        st.markdown("**Feature importance**")

        # The importance values are saved from most to least important.
        # Take the top ones, then sort ascending so the chart shows
        # the most important column on top.
        importance = pd.Series(model_results["importance"])
        importance = importance.head(TOP_FEATURES).sort_values()

        feature_labels = []
        for column in importance.index:
            feature_labels.append(LABELS.get(column, column))

        chart_data = pd.DataFrame({
            "feature": feature_labels,
            "importance": importance.values,
        })

        figure = px.bar(
            chart_data,
            x="importance",
            y="feature",
            orientation="h",
            labels={"importance": "Drop in ROC-AUC when shuffled", "feature": ""},
            color_discrete_sequence=[CHURN_COLORS["No"]],
        )
        figure.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(figure)
        st.caption(
            "Permutation importance: we shuffle one column and measure how much worse "
            "the model gets. A bigger drop means the column matters more."
        )


# =============================================================================
# PAGE
# =============================================================================

def main() -> None:
    """Draw the page from top to bottom."""
    model, metrics = require_model()
    best_model = metrics["best_model"]
    results = metrics["models"]

    st.title("Model Performance")
    st.caption(
        "How well each trained model predicts churn on the test set "
        "(customers the model never saw)."
    )

    show_training_info(metrics)
    show_comparison_table(results, best_model)

    # The user chooses which model to look at in detail
    model_names = list(results.keys())
    selected_name = st.selectbox(
        "Show details for",
        model_names,
        index=model_names.index(best_model),
    )
    selected_results = results[selected_name]

    left, right = st.columns(2)
    with left:
        show_confusion_matrix(selected_results)
    with right:
        show_roc_curve(selected_results, selected_name)

    show_feature_importance(selected_results)


main()
