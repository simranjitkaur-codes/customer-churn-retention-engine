"""
data_preprocessing.py

Reusable data loading and cleaning functions for the churn project.
"""

import pandas as pd
import numpy as np


def load_raw_data(filepath: str) -> pd.DataFrame:
    """Load the raw Telco Customer Churn CSV file."""
    df = pd.read_csv(filepath)
    return df


def fix_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """Convert TotalCharges to numeric and replace blanks with 0.0."""
    df = df.copy()
    df["TotalCharges"] = df["TotalCharges"].astype(str).str.strip()
    df["TotalCharges"] = df["TotalCharges"].replace("", np.nan)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
    return df


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Churn from Yes/No to 1/0."""
    df = df.copy()
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    return df


def drop_id_column(df: pd.DataFrame) -> pd.DataFrame:
    """Drop customerID because it is an identifier, not a feature."""
    df = df.copy()

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    return df


def clean_data(filepath: str) -> pd.DataFrame:
    """
    Load and clean the raw churn dataset.

    Steps:
    1. Fix TotalCharges
    2. Encode Churn as 0/1
    3. Drop customerID
    4. Remove duplicate rows
    """
    df = load_raw_data(filepath)
    df = fix_total_charges(df)
    df = encode_target(df)
    df = drop_id_column(df)
    df = df.drop_duplicates()

    return df