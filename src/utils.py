"""
Utilities: preprocessing, train/test split, and evaluation metrics.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    r2_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def preprocess_and_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    scale: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler | None]:
    """
    Split data into train/test and optionally scale features.

    Returns:
        X_train, X_test, y_train, y_test, scaler (or None if scale=False)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if _is_classification(y) else None,
    )
    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, scaler


def _is_classification(y: np.ndarray) -> bool:
    """Heuristic: classification targets are typically few unique integers."""
    return len(np.unique(y)) <= 20 and np.issubdtype(y.dtype, np.integer)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_classifier(
    model: Any,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    cv: int = CV_FOLDS,
) -> dict[str, float]:
    """Fit model, predict, and return train/test accuracy and cross-validation F1."""
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    cv_scores = cross_val_score(
        model, X_train, y_train, cv=cv, scoring="f1_weighted", n_jobs=-1
    )

    return {
        "train_accuracy": accuracy_score(y_train, y_pred_train),
        "test_accuracy": accuracy_score(y_test, y_pred_test),
        "test_precision": precision_score(
            y_test, y_pred_test, average="weighted", zero_division=0
        ),
        "test_recall": recall_score(
            y_test, y_pred_test, average="weighted", zero_division=0
        ),
        "test_f1": f1_score(y_test, y_pred_test, average="weighted", zero_division=0),
        "cv_f1_mean": float(np.mean(cv_scores)),
        "cv_f1_std": float(np.std(cv_scores)),
    }


def evaluate_regressor(
    model: Any,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    cv: int = CV_FOLDS,
) -> dict[str, float]:
    """Fit model, predict, and return MSE, MAE, R² and CV R²."""
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    cv_scores = cross_val_score(
        model, X_train, y_train, cv=cv, scoring="r2", n_jobs=-1
    )

    return {
        "train_r2": r2_score(y_train, y_pred_train),
        "test_r2": r2_score(y_test, y_pred_test),
        "test_mse": mean_squared_error(y_test, y_pred_test),
        "test_mae": mean_absolute_error(y_test, y_pred_test),
        "cv_r2_mean": float(np.mean(cv_scores)),
        "cv_r2_std": float(np.std(cv_scores)),
    }


def print_classification_report(
    y_test: np.ndarray, y_pred: np.ndarray, target_names: list[str]
) -> None:
    """Print sklearn classification report."""
    print(classification_report(y_test, y_pred, target_names=target_names))
