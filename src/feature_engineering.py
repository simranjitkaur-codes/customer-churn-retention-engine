"""
Reusable feature engineering and preprocessing for the churn project.
"""

from typing import List, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create simple engineered features.
    """
    df = df.copy()

    # Average monthly charge
    df["AvgMonthlyCharge"] = np.where(
        df["tenure"] > 0,
        df["TotalCharges"] / df["tenure"],
        df["MonthlyCharges"]
    )

    # Tenure groups
    df["TenureGroup"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=[
            "0-12 months",
            "13-24 months",
            "25-48 months",
            "49-72 months"
        ]
    )

    return df


def get_feature_lists(
    X: pd.DataFrame
) -> Tuple[List[str], List[str]]:
    """
    Return numeric and categorical feature names.
    """
    numeric_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    return numeric_features, categorical_features


def build_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str]
) -> ColumnTransformer:
    """
    Build the preprocessing pipeline.
    """

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ],
        remainder="drop"
    )

    return preprocessor


def prepare_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = "Churn"
):
    """
    Prepare train and test features using a leakage-free pipeline.

    Returns:
        X_train_processed
        X_test_processed
        y_train
        y_test
        preprocessor
        feature_names
    """

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]

    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    # Engineer features
    X_train = add_engineered_features(X_train)
    X_test = add_engineered_features(X_test)

    # Identify feature types
    numeric_features, categorical_features = get_feature_lists(
        X_train
    )

    # Build preprocessor
    preprocessor = build_preprocessor(
        numeric_features,
        categorical_features
    )

    # IMPORTANT: fit only on training data
    preprocessor.fit(X_train)

    # Transform both datasets
    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()

    return (
        X_train_processed,
        X_test_processed,
        y_train,
        y_test,
        preprocessor,
        feature_names
    )