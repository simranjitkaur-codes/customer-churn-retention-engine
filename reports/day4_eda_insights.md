# Day 4 — Train/Test Split and Exploratory Data Analysis

## Objective

The cleaned customer churn dataset was divided into training and test sets using an 80/20 stratified split. Exploratory data analysis was performed only on the training dataset to avoid using test-set information during analysis.

## Dataset Split

| Dataset | Rows | Churn Rate |
|---|---:|---:|
| Full | 7021 | 26.45% |
| Training | 5616 | 26.44% |
| Test | 1405 | 26.48% |

The split used `random_state=42` and stratification based on the `Churn` column.

## Key EDA Findings

### 1. Overall Churn Distribution

The training dataset contains:

- 4,131 customers who stayed
- 1,485 customers who churned
- Overall observed churn rate: 26.44%

This shows that churn is present in a substantial portion of the dataset, while customers who stayed form the majority.

### 2. Contract Type and Churn

Observed churn rates by contract type:

| Contract Type | Customers | Churn Rate |
|---|---:|---:|
| Month-to-month | 3078 | 42.66% |
| One year | 1182 | 11.17% |
| Two year | 1356 | 2.95% |

Month-to-month customers have the highest observed churn rate, while two-year customers have the lowest observed churn rate.

These results describe an association in the training data and should not be interpreted as proof that contract type directly causes churn.

### 3. Tenure and Churn

| Customer Status | Customers | Mean Tenure | Median Tenure |
|---|---:|---:|---:|
| Churned | 1485 | 18.35 | 10.0 |
| Stayed | 4131 | 37.50 | 37.0 |

Customers who churned have lower average and median tenure than customers who stayed.

### 4. Monthly Charges and Churn

| Customer Status | Customers | Mean Monthly Charges | Median Monthly Charges |
|---|---:|---:|---:|
| Churned | 1485 | 75.13 | 79.90 |
| Stayed | 4131 | 61.22 | 64.50 |

Customers who churned had higher average and median monthly charges than customers who stayed.

## Visualizations

The following visualizations were created using training data only:

1. `01_churn_distribution.png`
2. `02_contract_churn_rate.png`
3. `03_tenure_vs_churn.png`
4. `04_monthly_charges_vs_churn.png`

## Business-Oriented Observations

The EDA identifies several customer segments that may deserve further investigation:

- Month-to-month customers show substantially higher observed churn.
- Customers with shorter tenure show higher churn levels.
- Customers with higher monthly charges show higher churn levels.

These observations will be investigated further in later stages of the project. They should not be treated as causal conclusions.

## Day 4 Conclusion

The train/test split has been completed reproducibly, and the initial EDA provides several useful patterns for subsequent churn analysis and modeling. The test dataset was kept separate from exploratory analysis.