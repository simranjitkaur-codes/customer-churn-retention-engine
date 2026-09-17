"""
Package the existing fitted model and evaluate its configuration.

Run from the project root:
    python -m src.export_pipeline
"""

import json
import platform
import warnings
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.exceptions import InconsistentVersionWarning
from sklearn.metrics import (
    make_scorer,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.feature_engineering import add_engineered_features
from src.model_pipeline import make_churn_pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"


def main():

    # Stop if the saved sklearn model was created with
    # an incompatible sklearn version.
    warnings.simplefilter("error", InconsistentVersionWarning)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing fitted components.
    preprocessor = joblib.load(
        MODELS_DIR / "preprocessor.joblib"
    )

    classifier = joblib.load(
        MODELS_DIR / "churn_model.pkl"
    )

    # Combine them into one pipeline.
    pipeline = make_churn_pipeline(
        preprocessor,
        classifier
    )

    # Training data only.
    train_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "train_churn_data.csv"
    )

    if not train_path.exists():
        raise FileNotFoundError(
            f"Training file not found: {train_path}"
        )

    train_df = pd.read_csv(train_path)

    if "Churn" not in train_df.columns:
        raise ValueError(
            "Training data must contain a Churn column."
        )

    X = train_df.drop(columns=["Churn"])
    y = train_df["Churn"]

    if "customerID" in X.columns:
        raise ValueError(
            "Use the processed training file without customerID."
        )

    if y.isna().any():
        raise ValueError(
            "Churn contains missing values."
        )

    if set(y.unique()) != {0, 1}:
        raise ValueError(
            "Churn must contain exactly the classes 0 and 1."
        )

    if y.value_counts().min() < 5:
        raise ValueError(
            "Each class needs at least 5 rows for this CV setup."
        )

    # ========================================================
    # INTEGRATION CHECK
    # ========================================================

    # Compare:
    #
    # Old:
    # raw data → feature engineering → preprocessor → model
    #
    # New:
    # raw data → combined pipeline
    #
    sample = X.head(20).copy()

    old_features = add_engineered_features(sample)

    old_processed = preprocessor.transform(
        old_features
    )

    old_probabilities = classifier.predict_proba(
        old_processed
    )

    new_probabilities = pipeline.predict_proba(
        sample
    )

    np.testing.assert_allclose(
        old_probabilities,
        new_probabilities,
        rtol=1e-7,
        atol=1e-9,
    )

    print(
        "Prediction check passed: "
        "old and combined workflows match."
    )

    # ========================================================
    # CROSS-VALIDATION
    # ========================================================

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    scoring = {
        "accuracy": "accuracy",
        "roc_auc": "roc_auc",
        "average_precision": "average_precision",
        "precision": make_scorer(
            precision_score,
            zero_division=0
        ),
        "recall": make_scorer(
            recall_score,
            zero_division=0
        ),
        "f1": make_scorer(
            f1_score,
            zero_division=0
        ),
    }

    cv_results = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
        error_score="raise",
    )

    fold_scores = pd.DataFrame(
        {
            metric: cv_results[
                f"test_{metric}"
            ]
            for metric in scoring
        }
    )

    summary = (
        fold_scores
        .agg(["mean", "std"])
        .T
        .rename_axis("metric")
        .reset_index()
    )

    fold_scores.insert(
        0,
        "fold",
        range(1, 6)
    )

    # Save fold-level results.
    fold_scores.to_csv(
        REPORTS_DIR / "day11_cv_folds.csv",
        index=False,
    )

    # Save mean/std summary.
    summary.to_csv(
        REPORTS_DIR / "day11_cv_summary.csv",
        index=False,
    )

    print("\nDevelopment CV results:")
    print(
        summary.round(4).to_string(
            index=False
        )
    )

    # ========================================================
    # SAVE COMBINED PIPELINE
    # ========================================================

    output_path = (
        MODELS_DIR
        / "churn_pipeline.joblib"
    )

    joblib.dump(
        pipeline,
        output_path,
        compress=3,
    )

    # Verify that serialization did not change predictions.
    restored_pipeline = joblib.load(
        output_path
    )

    np.testing.assert_allclose(
        old_probabilities,
        restored_pipeline.predict_proba(sample),
        rtol=1e-7,
        atol=1e-9,
    )

    print(
        "\nSave/load prediction check passed."
    )

    # ========================================================
    # RECORD RUNTIME VERSIONS
    # ========================================================

    package_names = [
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "joblib",
        "streamlit",
    ]

    if type(classifier).__module__.startswith(
        "xgboost"
    ):
        package_names.append("xgboost")

    package_versions = {
        name: version(name)
        for name in package_names
    }

    metadata = {
        "classifier": type(classifier).__name__,
        "python_version": platform.python_version(),
        "runtime_packages": package_versions,
        "training_rows": len(train_df),
        "cv_folds": 5,
        "cv_random_state": 42,
        "evaluation_scope": (
            "Development cross-validation; "
            "not independent final-test performance."
        ),
        "model_file": (
            "models/churn_pipeline.joblib"
        ),
    }

    metadata_path = (
        REPORTS_DIR
        / "day11_pipeline_metadata.json"
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2
        ),
        encoding="utf-8",
    )

    # ========================================================
    # APP REQUIREMENTS
    # ========================================================

    app_requirements = (
        PROJECT_ROOT
        / "app"
        / "requirements.txt"
    )

    app_requirements.write_text(
        "".join(
            f"{name}=={installed_version}\n"
            for name, installed_version
            in package_versions.items()
        ),
        encoding="utf-8",
    )

    print(
        "\nSaved pipeline:",
        output_path
    )

    print(
        "Saved metadata:",
        metadata_path
    )

    print(
        "Saved app requirements:",
        app_requirements
    )


if __name__ == "__main__":
    main()