"""
app.py
======
Entry point of the Customer Churn Prediction app.

How to run:

    streamlit run app.py

This file does three things:
    1. Sets the page title and layout.
    2. Creates the navigation menu (one entry for each file in views/).
    3. Shows the app name and the model in the sidebar.
"""

import streamlit as st

from utils import load_metrics


def create_pages() -> list:
    """Create the list of pages shown in the navigation menu."""
    pages = [
        st.Page("views/home.py", title="Home & Data", icon=":material/home:", default=True),
        st.Page("views/insights.py", title="Churn Insights", icon=":material/bar_chart:"),
        st.Page("views/model_performance.py", title="Model Performance", icon=":material/monitoring:"),
        st.Page("views/predict.py", title="Predict a Customer", icon=":material/person_search:"),
        st.Page("views/batch.py", title="Batch Prediction", icon=":material/table_view:"),
        st.Page("views/about.py", title="About", icon=":material/info:"),
    ]
    return pages


def show_sidebar_info() -> None:
    """Show the app name and the model in use, under the page links."""
    metrics = load_metrics()

    with st.sidebar:
        st.markdown("### Churn Predictor")
        st.caption("Streamlit + Machine Learning")
        st.divider()

        st.caption("MODEL")
        if metrics is None:
            st.warning("No model yet. Run `python train_model.py`.")
        else:
            st.markdown(f"**{metrics['best_model']}**")
            st.caption(f"Loaded from models/model.pkl · trained {metrics['trained_at']}")


def main() -> None:
    """Set up the app and run the page that the user selected."""
    st.set_page_config(
        page_title="Churn Predictor",
        page_icon=":material/insights:",
        layout="wide",
    )

    pages = create_pages()
    selected_page = st.navigation(pages)

    show_sidebar_info()
    selected_page.run()


main()
