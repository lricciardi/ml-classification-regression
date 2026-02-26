"""
Model definitions: classification and regression estimators.
"""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.svm import SVC

from .utils import RANDOM_STATE


def get_classification_models() -> dict[str, Any]:
    """Return a dict of named classification models (Iris-ready)."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=500, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE
        ),
        "SVM (RBF)": SVC(kernel="rbf", random_state=RANDOM_STATE),
    }


def get_regression_models() -> dict[str, Any]:
    """Return a dict of named regression models (Diabetes-ready)."""
    return {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, random_state=RANDOM_STATE
        ),
    }
