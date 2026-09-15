# Day 7 — Baseline Model Training Results

## Models Trained

1. Logistic Regression
2. Random Forest
3. XGBoost

## Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0.7801 | 0.5698 | 0.6909 | 0.6245 | 0.8411 |
| Logistic Regression | 0.7374 | 0.5026 | 0.7742 | 0.6095 | 0.8399 |
| XGBoost | 0.7473 | 0.5160 | 0.7366 | 0.6069 | 0.8393 |

## Best Model

Based on ROC-AUC, the best baseline model is:

**Random Forest**

ROC-AUC: **0.8411**

## Key Observations

- Random Forest achieved the highest ROC-AUC, accuracy, precision, and F1-score.
- Logistic Regression achieved the highest recall at 0.7742.
- XGBoost achieved a ROC-AUC of 0.8393.
- Recall is important for churn prediction because missing an actual churner can reduce the effectiveness of a retention campaign.
- Random Forest was selected as the best baseline based on ROC-AUC.
- The difference between Random Forest and Logistic Regression ROC-AUC was small, so recall remains an important business consideration.

## Files Saved

- `models/churn_model.pkl`
- `models/logistic_regression.pkl`
- `models/random_forest.pkl`
- `models/xgboost_model.pkl`
- `reports/model_comparison.csv`
- `reports/model_metrics.json`
- `reports/figures/08_model_comparison.png`
- `reports/figures/09_confusion_matrices.png`

## Next Steps

- Hyperparameter tuning
- Cross-validation
- SHAP explainability
- Model interpretations