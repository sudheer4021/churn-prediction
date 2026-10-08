import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


PARAMS = {
    "n_estimators": 100,
    "max_depth": 10,
    "random_state": 42,
    "class_weight": "balanced",
}


def train_and_measure(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    model = RandomForestClassifier(**PARAMS)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1_score": f1_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }
    return predictions, metrics


def validate_reproducibility() -> bool:
    X_train = np.load("data/processed/X_train_final.npy")
    X_test = np.load("data/processed/X_test_final.npy")
    y_train = np.load("data/processed/y_train.npy")
    y_test = np.load("data/processed/y_test.npy")

    first_predictions, first_metrics = train_and_measure(
        X_train, y_train, X_test, y_test
    )
    second_predictions, second_metrics = train_and_measure(
        X_train, y_train, X_test, y_test
    )

    predictions_match = np.array_equal(first_predictions, second_predictions)
    metrics_match = all(
        np.isclose(first_metrics[name], second_metrics[name], rtol=0, atol=1e-12)
        for name in first_metrics
    )

    print("Run 1 metrics:")
    for name, value in first_metrics.items():
        print(f"  {name}: {value:.6f}")
    print("Run 2 metrics:")
    for name, value in second_metrics.items():
        print(f"  {name}: {value:.6f}")
    print(f"Predictions identical: {predictions_match}")
    print(f"Metrics identical: {metrics_match}")
    if predictions_match and metrics_match:
        print("[PASS] Reproducibility validated successfully!")
        return True

    print("[FAIL] Reproducibility check failed!")
    return False


if __name__ == "__main__":
    sys.exit(0 if validate_reproducibility() else 1)
