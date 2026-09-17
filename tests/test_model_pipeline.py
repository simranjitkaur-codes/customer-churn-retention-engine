import joblib
import numpy as np
import pandas as pd
import pytest

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.feature_engineering import (
    add_engineered_features,
    build_preprocessor,
)
from src.model_pipeline import make_churn_pipeline


NUMERIC_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "AvgMonthlyCharge",
]


@pytest.fixture
def fitted_pipeline():

    tenure = [0, 3, 6, 12, 24, 36, 48, 60]

    X = pd.DataFrame(
        {
            "tenure": tenure,
            "MonthlyCharges": [50.0] * 8,
            "TotalCharges": [
                50.0 * value for value in tenure
            ],
            "Contract": [
                "Month-to-month",
                "Month-to-month",
                "Month-to-month",
                "One year",
                "Two year",
                "One year",
                "Two year",
                "Two year",
            ],
        }
    )

    y = np.array([1, 1, 0, 1, 0, 1, 0, 0])

    preprocessor = build_preprocessor(
        NUMERIC_FEATURES,
        ["Contract", "TenureGroup"],
    )

    pipeline = make_churn_pipeline(
        preprocessor,
        LogisticRegression(max_iter=2000),
    )

    pipeline.fit(X, y)

    return pipeline, X, y


def test_predicts_without_manual_feature_engineering(
    fitted_pipeline
):

    pipeline, X, _ = fitted_pipeline

    original = X.copy(deep=True)

    probabilities = pipeline.predict_proba(X)

    assert probabilities.shape == (len(X), 2)

    assert np.isfinite(probabilities).all()

    assert (
        (probabilities >= 0)
        & (probabilities <= 1)
    ).all()

    np.testing.assert_allclose(
        probabilities.sum(axis=1),
        np.ones(len(X)),
    )

    pd.testing.assert_frame_equal(
        X,
        original,
    )


def test_save_load_preserves_predictions(
    fitted_pipeline,
    tmp_path
):

    pipeline, X, _ = fitted_pipeline

    model_path = (
        tmp_path / "pipeline.joblib"
    )

    joblib.dump(
        pipeline,
        model_path,
    )

    restored = joblib.load(
        model_path
    )

    np.testing.assert_allclose(
        pipeline.predict_proba(X),
        restored.predict_proba(X),
    )


def test_unknown_category_does_not_crash(
    fitted_pipeline
):

    pipeline, X, _ = fitted_pipeline

    customer = X.iloc[[0]].copy()

    customer["Contract"] = (
        "New contract option"
    )

    probabilities = (
        pipeline.predict_proba(customer)
    )

    assert probabilities.shape == (1, 2)

    assert np.isfinite(
        probabilities
    ).all()


def test_cv_imputer_uses_only_fold_training_rows(
    fitted_pipeline
):

    pipeline, X, y = fitted_pipeline

    splitter = StratifiedKFold(
        n_splits=2,
        shuffle=True,
        random_state=42,
    )

    folds = list(
        splitter.split(X, y)
    )

    results = cross_validate(
        pipeline,
        X,
        y,
        cv=folds,
        scoring="roc_auc",
        return_estimator=True,
        error_score="raise",
    )

    for estimator, (
        train_indices,
        _,
    ) in zip(
        results["estimator"],
        folds,
    ):

        fold_features = (
            add_engineered_features(
                X.iloc[train_indices]
            )
        )

        expected_medians = (
            fold_features[
                NUMERIC_FEATURES
            ]
            .median()
            .to_numpy()
        )

        fitted_imputer = (
            estimator
            .named_steps["preprocessor"]
            .named_transformers_["num"]
            .named_steps["imputer"]
        )

        np.testing.assert_allclose(
            fitted_imputer.statistics_,
            expected_medians,
        )