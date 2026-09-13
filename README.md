# Customer Churn Prediction & Retention Engine

**Status: Work in progress — initial project setup.**

## Project Goal

Build a machine learning project that classifies customer churn
using a public telecom dataset and explores candidate retention actions.

## Planned Workflow

1. Document the dataset source and inspect data quality.
2. Create a holdout split before fitting preprocessing.
3. Explore the training data.
4. Build preprocessing and baseline models.
5. Compare models using appropriate evaluation metrics.
6. Explain model predictions.
7. Add rule-based retention suggestions.
8. Build a Streamlit demonstration and automated tests.

## Current Progress
- Built a leakage-free scikit-learn preprocessing pipeline using ColumnTransformer and Pipelines.
- Created engineered features: AvgMonthlyCharge and TenureGroup.
- Applied imputation, scaling, and one-hot encoding.
- Saved the fitted preprocessor for reuse.
- Added reusable feature engineering logic and unit tests.
- Generated 50 final model-ready features.
### Day 6 — Feature Engineering & Preprocessing

The project now uses a reusable scikit-learn preprocessing pipeline.

- Numeric features → median imputation + standard scaling
- Categorical features → most-frequent imputation + one-hot encoding
- Unknown categories are handled safely using `handle_unknown="ignore"`
- Two interpretable engineered features were created:
  - `AvgMonthlyCharge`
  - `TenureGroup`
- Preprocessing is fitted only on training data to prevent data leakage.
- Final feature matrix contains **50 features**.


## Local Setup — Windows PowerShell

Run these commands from the project root:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If you need to create the virtual environment:

```powershell
& "C:\Users\gurmeet singh\AppData\Local\Programs\Python\Python310\python.exe" -m venv .venv
```

## Important Limitation

Retention suggestions will initially be rule-based ideas for testing.
They should not be interpreted as proven ways to prevent churn.