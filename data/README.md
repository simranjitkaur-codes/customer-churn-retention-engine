# Dataset Documentation

## Dataset Name

Telco Customer Churn (IBM Sample Data)

## Source

Downloaded from Kaggle:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Originally published by IBM as part of their Cognos Analytics
sample datasets.

## License

The dataset is provided by Kaggle for educational and research use.
Please review Kaggle's dataset page for the current license terms
before any commercial use.

## Description

A fictional telecom company that provided home phone and internet
services to customers in California. Each row represents one
customer, and the target column indicates whether the customer
left within the last month.

## Size

- Rows: 7043
- Columns: 21
- Target column: Churn (Yes / No)

## Column Groups

**Demographic information**
- customerID
- gender
- SeniorCitizen
- Partner
- Dependents

**Account information**
- tenure
- Contract
- PaperlessBilling
- PaymentMethod
- MonthlyCharges
- TotalCharges

**Services subscribed**
- PhoneService
- MultipleLines
- InternetService
- OnlineSecurity
- OnlineBackup
- DeviceProtection
- TechSupport
- StreamingTV
- StreamingMovies

**Target**
- Churn

## Download Instructions

1. Create a free account on Kaggle.
2. Open the dataset link above.
3. Click Download and extract the ZIP file.
4. Rename the CSV file to: telco_customer_churn.csv
5. Place it inside: data/raw/

The CSV file is excluded from Git tracking via .gitignore.

## Known Data Issues to Handle Later

- The TotalCharges column is stored as text and contains some
  blank values that must be converted to numbers.
- The Churn column is Yes/No text and must be converted to 1/0
  before model training.
- The customerID column is a unique identifier and should be
  removed before training.
- Class distribution is imbalanced (fewer churners than
  non-churners), which must be considered during evaluation.