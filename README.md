# Customer Churn Prediction App

## Requirements

### Core Dependencies
- `streamlit` >= 1.50
- `pandas` >= 2.0
- `numpy` >= 1.26
- `scikit-learn` >= 1.4
- `xgboost` >= 2.0
- `plotly` >= 5.20
- `joblib` >= 1.3

### Only for the Notebook (`churn_analysis.ipynb`)
- `matplotlib` >= 3.8
- `notebook` >= 7.0

A Streamlit web app that uses a machine learning model to predict which customers are likely to churn, explain why, and suggest actions.

## Project structure

```
churn-app/
├── app.py                  # Entry point: page navigation + sidebar
├── train_model.py          # Trains 4 models, saves the best one
├── utils.py                # Settings, data cleaning, prediction helpers
├── churn_analysis.ipynb    # Notebook: EDA, model training and evaluation, step by step
├── requirements.txt
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv   # <- put your dataset here
├── models/                 # Created by train_model.py
│   ├── model.pkl
│   └── metrics.json
└── views/                  # One file per page
    ├── home.py               # F1 Load dataset, F2 Data overview
    ├── insights.py           # F3 Churn insights dashboard
    ├── model_performance.py  # F4 Model performance
    ├── predict.py            # F5 Prediction, F6 Explanation, F8 Risk level, F10 Suggestions
    ├── batch.py              # F7 Batch prediction, F8 Risk level, F9 Download
    └── about.py              # F11 About
```

## How to run

1. Create a virtual environment and install the packages:

   ```bash
   python -m venv .venv
   # Windows:      .venv\Scripts\activate
   # macOS/Linux:  source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. The dataset is already in `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`. To use another copy, replace that file and keep the same name.

3. Train the model (takes about 15 seconds):

   ```bash
   python train_model.py
   ```

4. Start the app:

   ```bash
   streamlit run app.py
   ```

   It opens in your browser at http://localhost:8501.

If you retrain the model while the app is running, restart the app so it loads the new model.

## Analysis notebook

`churn_analysis.ipynb` explains the whole analysis step by step, with charts: data quality checks, cleaning, exploration (EDA), training the 4 models (with a chart of the Decision Tree's first rules), evaluation (metrics, confusion matrices, ROC curves), feature importance, a single prediction and the conclusion.

It is saved with all outputs, so you can read it without running it (GitHub and VS Code show the charts). To run it yourself:

```bash
jupyter notebook churn_analysis.ipynb
```

The notebook uses the same settings as `train_model.py`, so it gets the same results as the app (Random Forest, F1-score 0.629). Its last code cell checks this.

## How the code is written

Every file follows the same structure, so it is easy to read:

1. **Header** at the top: what the file does and which features (F1–F11) it covers.
2. **Imports**, then **constants** (names in CAPITAL letters).
3. **Small functions**, each with a docstring that says what it does, what it needs and what it returns.
4. **`main()`** at the bottom: it calls the functions in order, from the top of the page to the bottom.

To understand a page, read its `main()` function first. Then read the functions it calls.

| File | Read this first |
|---|---|
| `train_model.py` | `main()` shows the 6 steps: load, split, train, evaluate, choose, save |
| `utils.py` | The numbered sections 1–6 listed in the header |
| `views/*.py` | `main()` at the bottom of each file |

## Dataset

IBM Telco Customer Churn (Kaggle file `WA_Fn-UseC_-Telco-Customer-Churn.csv`): 7,043 customers, 21 columns.
`customerID, gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges, Churn`

Cleaning done automatically (`clean_data` in `utils.py`):
- `TotalCharges` text → numbers (11 blank values become missing, then filled with the median)
- `SeniorCitizen` 0/1 → No/Yes
- `Churn` accepted as Yes/No, 1/0 or True/False
- `customerID` is not used as a feature

## How the model works

- **Preprocessing** (inside the saved pipeline): median fill + scaling for numbers, most-common fill + one-hot encoding for categories.
- **Models compared:** Logistic Regression, Decision Tree (`max_depth=5`, as in the team notebook), Random Forest, XGBoost. Class weights handle the imbalance (about 1 in 4 customers churn).
- **Split:** 80% train / 20% test, stratified, `random_state=42`.
- **Best model:** highest F1-score on the test set. Change `BEST_BY` in `train_model.py` to use `"recall"` or `"roc_auc"` instead.
- **Feature importance:** permutation importance (drop in ROC-AUC when a column is shuffled).
- **Why this prediction:** for each column, the customer's value is swapped for the values other customers usually have, and the change in churn probability is shown.
- **Risk levels:** Low < 40%, Medium 40–70%, High ≥ 70% (change `LOW_MAX` / `HIGH_MIN` in `utils.py`).

Test results (20% test set, 1,409 customers):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.737 | 0.503 | 0.783 | 0.613 | 0.841 |
| Decision Tree | 0.755 | 0.527 | 0.759 | 0.622 | 0.832 |
| **Random Forest** | **0.760** | **0.533** | **0.767** | **0.629** | **0.841** |
| XGBoost | 0.756 | 0.527 | 0.773 | 0.627 | 0.840 |

Your numbers may differ slightly depending on package versions.
