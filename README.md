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
- Trained 3 baseline models: Logistic Regression, Random Forest, and XGBoost.
- Evaluated models using Accuracy, Precision, Recall, F1-score, and ROC-AUC.
- Random Forest achieved the best baseline ROC-AUC of 0.8411.
- Added model training module (`src/train_model.py`) with unit tests.

```markdown
- Built a Streamlit dashboard with Overview, Dashboard, and Predict Churn pages.

## Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0.7801 | 0.5698 | 0.6909 | 0.6245 | 0.8411 |
| Logistic Regression | 0.7374 | 0.5026 | 0.7742 | 0.6095 | 0.8399 |
| XGBoost | 0.7473 | 0.5160 | 0.7366 | 0.6069 | 0.8393 |

![Model Comparison](reports/figures/08_model_comparison.png)

![Confusion Matrices](reports/figures/09_confusion_matrices.png)

## Day 8 — Explainability and Retention Recommendations

Day 8 adds model explainability and a rule-based retention engine.

### Explainability

- Random Forest feature importance
- Global SHAP analysis
- Individual customer SHAP explanation
- Highest-risk customer analysis

### Retention Engine

The recommendation engine is implemented in:

`src/recommendations.py`

It converts customer attributes and churn probability into:

- Risk level
- Retention recommendations
- Customer explanation report

## Live Demo

Run the app locally:

```bash
streamlit run app/streamlit_app.py

### Day 8 Outputs

- `reports/figures/13_feature_importance.png`
- `reports/figures/14_shap_summary.png`
- `reports/figures/15_shap_individual_explanation.png`
- `reports/feature_importance.csv`
- `reports/shap_global_importance.csv`
- `reports/day8_individual_shap_explanation.csv`
- `reports/day8_example_customer_explanation.json`


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