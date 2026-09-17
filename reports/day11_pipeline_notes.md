# Day 11 — Unified Pipeline and Evaluation Review

## Pipeline

Customer fields
→ engineered features
→ imputation and encoding/scaling
→ classifier

Saved artifact: models/churn_pipeline.joblib

## Classifier

Actual classifier: RandomForestClassifier

The existing fitted classifier was packaged with its preprocessing.
This step did not select a new best model.

## Verification

- Original workflow and combined pipeline predictions matched: Yes
- Predictions matched after saving and loading: Yes
- Pipeline tests passed: Yes

## Development Cross-Validation

Source: data/processed/train_churn_data.csv

Preprocessing was fitted inside each training fold.

| Metric | Mean | Standard deviation |
|---|---:|---:|
| ROC-AUC | 0.8440 | 0.0191 |
| Average precision | 0.6580 | 0.0399 |
| Precision | 0.5702 | 0.0167 |
| Recall | 0.6976 | 0.0301 |
| F1 | 0.6274 | 0.0196 |

## Evaluation Limitations

Earlier experiments used the original test set for model comparison.
Those results are therefore development results, not an independent
final evaluation.

Today's cross-validation fixes fold-level preprocessing, but does
not undo earlier model-selection decisions or dataset limitations.

## Deployment

- App loads one combined pipeline.
- Runtime package versions are recorded and pinned.
- Probability calibration and business risk cutoffs remain unvalidated.
- Retention suggestions are hypotheses, not demonstrated savings.