"""
Combine feature engineering, preprocessing, and classification.

The feature-engineering function stays in an importable src module.
"""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from src.feature_engineering import add_engineered_features


def make_churn_pipeline(preprocessor, classifier):
    """
    Build a pipeline from either fitted or unfitted components.

    Input:
        Customer DataFrame before feature engineering.

    Output:
        A pipeline supporting fit(), predict(), and predict_proba().
    """

    if isinstance(classifier, Pipeline):
        raise ValueError(
            "The classifier is already a Pipeline. "
            "These instructions expect the standalone model saved on Day 7. "
            "Do not apply preprocessing twice."
        )

    return Pipeline(
        steps=[
            (
                "features",
                FunctionTransformer(
                    add_engineered_features,
                    validate=False,
                ),
            ),
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )