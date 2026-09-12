# Data Inspection Notes

## Day 2 — Data Inspection

- Dataset shape: 7043 rows × 21 columns.
- TotalCharges was stored as object instead of numeric.
- 11 rows contained blank/whitespace values in TotalCharges.
- Churn distribution:
  - No: 5174
  - Yes: 1869

---

## Day 3 — Cleaning Results

### Steps applied

1. TotalCharges: blank spaces replaced with 0.0, converted to float64.
   Reason: 11 new customers (tenure = 0) had no charges yet.

2. Churn: mapped Yes → 1, No → 0. Now int64.

3. customerID: dropped. Not a predictive feature.

4. Duplicates: 22 duplicate occurrences found and removed.

### Final cleaned dataset

- Shape: (7021, 20)
- Saved to: data/processed/cleaned_churn_data.csv

### Unit tests

- 10 tests written in tests/test_preprocessing.py
- All 10 passed.