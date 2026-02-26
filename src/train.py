"""
Training pipeline: load data, preprocess, train classification and regression models, evaluate.
"""

from __future__ import annotations

import warnings

from sklearn.datasets import load_diabetes, load_iris

from .model import get_classification_models, get_regression_models
from .utils import (
    evaluate_classifier,
    evaluate_regressor,
    preprocess_and_split,
    print_classification_report,
)

warnings.filterwarnings("ignore", category=UserWarning)


def run_classification() -> None:
    """Load Iris, preprocess, train multiple classifiers, and report metrics."""
    data = load_iris()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test, _ = preprocess_and_split(X, y)
    models = get_classification_models()

    print("=" * 60)
    print("CLASSIFICATION (Iris dataset)")
    print("=" * 60)

    for name, model in models.items():
        metrics = evaluate_classifier(model, X_train, X_test, y_train, y_test)
        print(f"\n{name}")
        print(f"  Test Accuracy:  {metrics['test_accuracy']:.4f}")
        print(f"  Test F1:       {metrics['test_f1']:.4f}")
        print(f"  CV F1 (mean):  {metrics['cv_f1_mean']:.4f} (+/- {metrics['cv_f1_std']:.4f})")

    best_name = max(
        models.keys(),
        key=lambda n: evaluate_classifier(
            models[n], X_train, X_test, y_train, y_test
        )["test_f1"],
    )
    best_model = models[best_name]
    best_model.fit(X_train, y_train)
    print(f"\nClassification report ({best_name}):")
    print_classification_report(
        y_test, best_model.predict(X_test), list(data.target_names)
    )


def run_regression() -> None:
    """Load Diabetes, preprocess, train multiple regressors, and report metrics."""
    data = load_diabetes()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test, _ = preprocess_and_split(X, y, scale=True)
    models = get_regression_models()

    print("\n" + "=" * 60)
    print("REGRESSION (Diabetes dataset)")
    print("=" * 60)

    for name, model in models.items():
        metrics = evaluate_regressor(model, X_train, X_test, y_train, y_test)
        print(f"\n{name}")
        print(f"  Test R²:   {metrics['test_r2']:.4f}")
        print(f"  Test MSE:  {metrics['test_mse']:.2f}")
        print(f"  Test MAE:  {metrics['test_mae']:.2f}")
        print(f"  CV R²:     {metrics['cv_r2_mean']:.4f} (+/- {metrics['cv_r2_std']:.4f})")


def main() -> None:
    """Run classification and regression pipelines."""
    run_classification()
    run_regression()
    print("\nDone.")


if __name__ == "__main__":
    main()
