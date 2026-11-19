"""
Page 6: About
=============
Features:
    F11  About / Help page - what the app does, the dataset, the model,
                             how to use the app and the team
"""

import pandas as pd
import streamlit as st

from utils import TARGET, get_data, get_data_name, load_metrics


def show_description() -> None:
    """Explain in a few sentences what the app does."""
    with st.container(border=True):
        st.markdown("**What this app does**")
        st.write(
            "This app uses a machine learning model trained on customer data to predict "
            "which customers are likely to stop using the service (churn). It shows the "
            "churn probability, the risk level and the main reasons, so the business can "
            "act before the customer leaves."
        )


def show_dataset_info(df: pd.DataFrame) -> None:
    """Show the file name, the size and the target of the dataset."""
    with st.container(border=True):
        st.markdown("**Dataset**")
        st.markdown(
            f"- **File:** {get_data_name()}\n"
            f"- **Rows:** {len(df):,}\n"
            f"- **Columns:** {df.shape[1]}\n"
            f"- **Target:** {TARGET} (Yes / No)\n"
            f"- **Source:** IBM Telco Customer Churn sample dataset (Kaggle)"
        )


def show_how_to_use() -> None:
    """Show the four steps of using the app, with a link to each page."""
    with st.container(border=True):
        st.markdown("**How to use**")
        st.page_link(
            "views/home.py",
            label="1. Load the sample data or upload your CSV",
            icon=":material/home:",
        )
        st.page_link(
            "views/insights.py",
            label="2. Explore patterns in Churn Insights",
            icon=":material/bar_chart:",
        )
        st.page_link(
            "views/predict.py",
            label="3. Check one customer in Predict a Customer",
            icon=":material/person_search:",
        )
        st.page_link(
            "views/batch.py",
            label="4. Score a whole list in Batch Prediction",
            icon=":material/table_view:",
        )


def show_model_info(metrics) -> None:
    """Show the algorithm and the test results of the model in use."""
    with st.container(border=True):
        st.markdown("**Model**")

        if metrics is None:
            st.warning("No model yet. Run `python train_model.py`.")
            return

        best_model = metrics["best_model"]
        best_results = metrics["models"][best_model]

        st.markdown(
            f"- **Algorithm:** {best_model}\n"
            f"- **Library:** scikit-learn pipeline (cleaning + encoding + model)\n"
            f"- **Saved as:** models/model.pkl (joblib)\n"
            f"- **Test results:** F1 {best_results['f1']:.3f} · "
            f"Recall {best_results['recall']:.3f} · "
            f"ROC-AUC {best_results['roc_auc']:.3f}\n"
            f"- **Trained:** {metrics['trained_at']}"
        )
        st.page_link(
            "views/model_performance.py",
            label="See full results",
            icon=":material/monitoring:",
        )


def show_team_info() -> None:
    """Show the team details. Replace the text in [brackets] with your own."""
    with st.container(border=True):
        st.markdown("**Team**")
        st.markdown(
            "- **Group:** [Group name]\n"
            "- **Members:** [Name 1], [Name 2], [Name 3]\n"
            "- **Course:** [Course name]\n"
            "- **Lecturer:** [Lecturer name]"
        )


def main() -> None:
    """Draw the page from top to bottom."""
    df = get_data()
    metrics = load_metrics()

    st.title("About this App")
    st.caption("How to use the app, and what data and model are behind it.")

    show_description()

    left, right = st.columns(2)
    with left:
        show_dataset_info(df)
        show_how_to_use()
    with right:
        show_model_info(metrics)
        show_team_info()


main()
