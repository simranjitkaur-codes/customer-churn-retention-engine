import sys
import os

import numpy as np
import pytest

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
        )
    ),
)

from src.train_model import (
    evaluate_model,
    train_logistic_regression,
    train_random_forest,
    train_xgboost,
    train_all_models,
)


@pytest.fixture
def sample_data():
    """Create small synthetic classification data for testing."""

    np.random.seed(42)

    X_train = np.random.rand(100, 5)

    y_train = np.random.choice(
        [0, 1],
        size=100,
        p=[0.7, 0.3],
    )

    X_test = np.random.rand(30, 5)

    y_test = np.random.choice(
        [0, 1],
        size=30,
        p=[0.7, 0.3],
    )

    return X_train, y_train, X_test, y_test


def test_train_logistic_regression_returns_fitted_model(sample_data):

    X_train, y_train, X_test, y_test = sample_data

    model = train_logistic_regression(
        X_train,
        y_train,
    )

    predictions = model.predict(X_test)

    assert len(predictions) == len(X_test)


def test_train_random_forest_returns_fitted_model(sample_data):

    X_train, y_train, X_test, y_test = sample_data

    model = train_random_forest(
        X_train,
        y_train,
    )

    predictions = model.predict(X_test)

    assert len(predictions) == len(X_test)


def test_train_xgboost_returns_fitted_model(sample_data):

    X_train, y_train, X_test, y_test = sample_data

    model = train_xgboost(
        X_train,
        y_train,
    )

    predictions = model.predict(X_test)

    assert len(predictions) == len(X_test)


def test_evaluate_model_returns_expected_keys(sample_data):

    X_train, y_train, X_test, y_test = sample_data

    model = train_logistic_regression(
        X_train,
        y_train,
    )

    metrics = evaluate_model(
        model,
        X_test,
        y_test,
        "Test Model",
    )

    expected_keys = {
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1_Score",
        "ROC_AUC",
    }

    assert expected_keys == set(metrics.keys())


def test_evaluate_model_metrics_are_valid_range(sample_data):

    X_train, y_train, X_test, y_test = sample_data

    model = train_random_forest(
        X_train,
        y_train,
    )

    metrics = evaluate_model(
        model,
        X_test,
        y_test,
        "Test Model",
    )

    for key in [
        "Accuracy",
        "Precision",
        "Recall",
        "F1_Score",
        "ROC_AUC",
    ]:
        assert 0.0 <= metrics[key] <= 1.0


def test_train_all_models_returns_three_results(sample_data):

    X_train, y_train, X_test, y_test = sample_data

    comparison_df, models = train_all_models(
        X_train,
        y_train,
        X_test,
        y_test,
    )

    assert len(comparison_df) == 3

    assert set(models.keys()) == {
        "Logistic Regression",
        "Random Forest",
        "XGBoost",
    }