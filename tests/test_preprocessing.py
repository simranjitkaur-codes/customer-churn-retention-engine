"""
test_preprocessing.py

Unit tests for data_preprocessing.py cleaning functions.
"""

import pytest
import pandas as pd
import sys
import os

# Allow imports from the project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_preprocessing import (
    fix_total_charges,
    encode_target,
    drop_id_column,
)


# ── fix_total_charges ──────────────────────────────────────────────────────────

def test_fix_total_charges_converts_to_float():
    """TotalCharges column must be float64 after fixing."""
    df = pd.DataFrame({"TotalCharges": ["100.5", "200.0", " ", "300.0"]})
    result = fix_total_charges(df)
    assert result["TotalCharges"].dtype == float


def test_fix_total_charges_blank_becomes_zero():
    """Blank TotalCharges rows must become 0.0."""
    df = pd.DataFrame({"TotalCharges": [" ", "", "100.0"]})
    result = fix_total_charges(df)
    assert result["TotalCharges"].iloc[0] == 0.0
    assert result["TotalCharges"].iloc[1] == 0.0


def test_fix_total_charges_does_not_change_valid_values():
    """Valid numeric values must not be changed."""
    df = pd.DataFrame({"TotalCharges": ["500.75", "1200.0"]})
    result = fix_total_charges(df)
    assert result["TotalCharges"].iloc[0] == pytest.approx(500.75)
    assert result["TotalCharges"].iloc[1] == pytest.approx(1200.0)


def test_fix_total_charges_no_nulls_remain():
    """No NaN values should remain after fixing."""
    df = pd.DataFrame({"TotalCharges": [" ", "100.0", ""]})
    result = fix_total_charges(df)
    assert result["TotalCharges"].isnull().sum() == 0


# ── encode_target ──────────────────────────────────────────────────────────────

def test_encode_target_yes_becomes_one():
    """Churn = Yes must map to 1."""
    df = pd.DataFrame({"Churn": ["Yes", "No"]})
    result = encode_target(df)
    assert result["Churn"].iloc[0] == 1


def test_encode_target_no_becomes_zero():
    """Churn = No must map to 0."""
    df = pd.DataFrame({"Churn": ["Yes", "No"]})
    result = encode_target(df)
    assert result["Churn"].iloc[1] == 0


def test_encode_target_dtype_is_int():
    """Churn column must be integer after encoding."""
    df = pd.DataFrame({"Churn": ["Yes", "No", "No", "Yes"]})
    result = encode_target(df)
    assert pd.api.types.is_integer_dtype(result["Churn"])


# ── drop_id_column ─────────────────────────────────────────────────────────────

def test_drop_id_column_removes_customerid():
    """customerID column must not exist after dropping."""
    df = pd.DataFrame({"customerID": ["001", "002"], "tenure": [5, 10]})
    result = drop_id_column(df)
    assert "customerID" not in result.columns


def test_drop_id_column_keeps_other_columns():
    """All columns other than customerID must be kept."""
    df = pd.DataFrame({"customerID": ["001"], "tenure": [5], "Churn": [1]})
    result = drop_id_column(df)
    assert "tenure" in result.columns
    assert "Churn" in result.columns


def test_drop_id_column_safe_if_no_customerid():
    """Function must not raise an error if customerID is already absent."""
    df = pd.DataFrame({"tenure": [5, 10], "Churn": [1, 0]})
    result = drop_id_column(df)
    assert result.shape == df.shape