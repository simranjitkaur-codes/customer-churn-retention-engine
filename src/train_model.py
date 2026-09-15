"""
train_model.py

Train and evaluate baseline classification models for churn prediction.
"""

import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """
    Evaluate a trained classifier and return key metrics.
    """

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1_Score": f1_score(y_test, y_pred),
        "ROC_AUC": roc_auc_score(y_test, y_proba),
    }


def train_logistic_regression(X_train, y_train) -> LogisticRegression:
    """Train Logistic Regression baseline."""

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )

    model.fit(X_train, y_train)

    return model


def train_random_forest(X_train, y_train) -> RandomForestClassifier:
    """Train Random Forest baseline."""

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    return model


def train_xgboost(X_train, y_train) -> XGBClassifier:
    """Train XGBoost baseline."""

    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()

    scale_pos_weight = neg_count / pos_count

    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss",
    )

    model.fit(X_train, y_train)

    return model


def train_all_models(X_train, y_train, X_test, y_test):
    """
    Train all three baseline models and return
    the comparison DataFrame and trained models.
    """

    results = []

    log_reg = train_logistic_regression(
        X_train,
        y_train,
    )

    results.append(
        evaluate_model(
            log_reg,
            X_test,
            y_test,
            "Logistic Regression",
        )
    )

    rf_model = train_random_forest(
        X_train,
        y_train,
    )

    results.append(
        evaluate_model(
            rf_model,
            X_test,
            y_test,
            "Random Forest",
        )
    )

    xgb_model = train_xgboost(
        X_train,
        y_train,
    )

    results.append(
        evaluate_model(
            xgb_model,
            X_test,
            y_test,
            "XGBoost",
        )
    )

    comparison_df = (
        pd.DataFrame(results)
        .sort_values("ROC_AUC", ascending=False)
        .reset_index(drop=True)
    )

    models = {
        "Logistic Regression": log_reg,
        "Random Forest": rf_model,
        "XGBoost": xgb_model,
    }

    return comparison_df, models


def save_best_model(
    models: dict,
    comparison_df: pd.DataFrame,
    output_path,
):
    """
    Save the model with the highest ROC-AUC.
    """

    best_model_name = comparison_df.iloc[0]["Model"]

    best_model = models[best_model_name]

    joblib.dump(best_model, output_path)

    return best_model_name