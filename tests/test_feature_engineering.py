import os
import sys

import pandas as pd

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from src.feature_engineering import (
    add_engineered_features,
    get_feature_lists,
    build_preprocessor
)


def test_add_engineered_features_creates_columns():
    df = pd.DataFrame({
        "tenure": [0, 12, 36],
        "MonthlyCharges": [50.0, 70.0, 90.0],
        "TotalCharges": [0.0, 840.0, 3240.0]
    })

    result = add_engineered_features(df)

    assert "AvgMonthlyCharge" in result.columns
    assert "TenureGroup" in result.columns


def test_avg_monthly_charge_for_zero_tenure():
    df = pd.DataFrame({
        "tenure": [0],
        "MonthlyCharges": [65.0],
        "TotalCharges": [0.0]
    })

    result = add_engineered_features(df)

    assert result["AvgMonthlyCharge"].iloc[0] == 65.0


def test_get_feature_lists():
    df = pd.DataFrame({
        "tenure": [1, 2],
        "MonthlyCharges": [10.0, 20.0],
        "Contract": ["Month-to-month", "One year"],
        "gender": ["Male", "Female"]
    })

    num, cat = get_feature_lists(df)

    assert "tenure" in num
    assert "MonthlyCharges" in num
    assert "Contract" in cat
    assert "gender" in cat


def test_build_preprocessor_runs():
    df = pd.DataFrame({
        "tenure": [5, 10, 15],
        "MonthlyCharges": [50.0, 60.0, 70.0],
        "Contract": [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    })

    num, cat = get_feature_lists(df)

    preprocessor = build_preprocessor(num, cat)

    preprocessor.fit(df)

    transformed = preprocessor.transform(df)

    assert transformed.shape[0] == 3
    assert transformed.shape[1] > 3


def test_preprocessor_handles_unknown_category():
    train = pd.DataFrame({
        "tenure": [5, 10, 15],
        "Contract": [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    })

    test = pd.DataFrame({
        "tenure": [20],
        "Contract": ["New Contract"]
    })

    num, cat = get_feature_lists(train)

    preprocessor = build_preprocessor(num, cat)

    preprocessor.fit(train)

    transformed = preprocessor.transform(test)

    assert transformed.shape[0] == 1