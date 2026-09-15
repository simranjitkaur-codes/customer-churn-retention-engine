# Day 8 — Model Explainability and Retention Recommendations

## Objective

This stage explains the churn model predictions and connects them
to practical customer retention actions.

## Explainability Outputs

The following outputs were generated:

- `reports/figures/13_feature_importance.png`
- `reports/figures/14_shap_summary.png`
- `reports/figures/15_shap_individual_explanation.png`
- `reports/feature_importance.csv`
- `reports/shap_global_importance.csv`
- `reports/day8_individual_shap_explanation.csv`

## Model Explanation

The selected model is the Random Forest classifier.

The model uses 50 processed features after preprocessing and
one-hot encoding.

Feature importance identifies the features that contribute most
to the Random Forest's predictions.

SHAP provides both global and individual explanations:

- Global SHAP explains the overall influence of features.
- Individual SHAP explains the strongest influences for one customer.

## Individual Customer Explanation

The highest-risk customer from the test dataset was selected
using the model's predicted churn probability.

The customer's probability, risk category, recommendations,
and top SHAP features were saved in:

`reports/day8_example_customer_explanation.json`

## Retention Recommendation Engine

The rule-based recommendation engine is implemented in:

`src/recommendations.py`

The engine considers:

- Contract type
- Internet service
- Monthly charges
- Tenure
- Technical support
- Online security
- Payment method

The engine returns practical retention recommendations based
on customer characteristics.

## Testing

All automated tests passed:

- 31 tests passed

The tests cover preprocessing, feature engineering, model training,
model evaluation, risk levels, recommendation rules, and customer
report generation.

## Limitation

The recommendations are rule-based and should be treated as
decision-support suggestions rather than guaranteed causal solutions.