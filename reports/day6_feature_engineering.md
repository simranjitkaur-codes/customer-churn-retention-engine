# Day 6 — Feature Engineering Summary

## What was done

1. Loaded the stratified train/test split created during Day 4.
2. Separated features (X) and target (y).
3. Created two engineered features:
   - `AvgMonthlyCharge` = TotalCharges / tenure, with safe handling for tenure = 0.
   - `TenureGroup` = binned customer tenure.
4. Identified numeric and categorical features automatically.
5. Built a scikit-learn preprocessing pipeline using `ColumnTransformer`.
6. Numeric features:
   - Median imputation
   - StandardScaler
7. Categorical features:
   - Most-frequent imputation
   - OneHotEncoder
   - `handle_unknown="ignore"`
8. Fitted the preprocessor only on training data.
9. Transformed both training and test data.
10. Saved the fitted preprocessor using joblib.
11. Moved reusable feature engineering logic into `src/feature_engineering.py`.
12. Added unit tests for feature engineering and preprocessing.

## Dataset after feature engineering

- Training rows: 5,616
- Test rows: 1,405
- Engineered features added: 2

## Final feature count

After preprocessing and one-hot encoding:

**50 features**

## Leakage Prevention

The preprocessing pipeline was fitted only on the training dataset.

The test dataset was used only for transformation after the preprocessor had been fitted.

This prevents information from the test set from influencing preprocessing decisions.

## Engineered Features

### AvgMonthlyCharge

Calculated as:

`TotalCharges / tenure`

For customers with zero tenure, `MonthlyCharges` is used to avoid division by zero.

### TenureGroup

Customers are grouped into:

- 0–12 months
- 13–24 months
- 25–48 months
- 49–72 months

## Preprocessing Pipeline

### Numeric features

`SimpleImputer(strategy="median") → StandardScaler`

### Categorical features

`SimpleImputer(strategy="most_frequent") → OneHotEncoder(handle_unknown="ignore")`

## Saved Artifacts

- `models/preprocessor.joblib`
- `reports/final_feature_names.csv`

Processed `.npy` files were generated locally for future model training.

## Testing

Feature engineering tests were added to:

`tests/test_feature_engineering.py`

The complete project test suite passed successfully.

## Conclusion

The project now has a reusable and leakage-free feature engineering and preprocessing pipeline that is ready for model training.
