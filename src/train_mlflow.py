import os
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn

from pathlib import Path
from mlflow.tracking import MlflowClient

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)


# ============================================================
# 1. PROJECT PATH CONFIGURATION
# ============================================================

# Absolute path to your actual project
PROJECT_ROOT = Path(
    r"C:\Users\sudhe\OneDrive\Documents\churn-prediction"
)

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

# Artifact directories
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MLFLOW_ARTIFACTS_DIR = ARTIFACTS_DIR / "mlflow"

# Model directory
MODELS_DIR = PROJECT_ROOT / "models"

# MLflow SQLite database
MLFLOW_DB = PROJECT_ROOT / "mlflow.db"


# ============================================================
# 2. CREATE REQUIRED DIRECTORIES
# ============================================================

ARTIFACTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MLFLOW_ARTIFACTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODELS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 3. MLFLOW TRACKING CONFIGURATION
# ============================================================

# Use SQLite database instead of old filesystem MLflow backend
mlflow.set_tracking_uri(
    f"sqlite:///{MLFLOW_DB.resolve().as_posix()}"
)

EXPERIMENT_NAME = "Telco_Churn_Prediction"


# ============================================================
# 4. CREATE MLFLOW EXPERIMENT
# ============================================================

client = MlflowClient()

experiment = client.get_experiment_by_name(
    EXPERIMENT_NAME
)

if experiment is None:

    experiment_id = client.create_experiment(
        name=EXPERIMENT_NAME,
        artifact_location=MLFLOW_ARTIFACTS_DIR.as_uri()
    )

    print("=" * 60)
    print("NEW MLFLOW EXPERIMENT CREATED")
    print("=" * 60)

    print(
        "Experiment ID :",
        experiment_id
    )

    print(
        "Experiment    :",
        EXPERIMENT_NAME
    )

    print(
        "Artifact Root :",
        MLFLOW_ARTIFACTS_DIR
    )

else:

    # This should normally not happen on the first clean run.
    experiment_id = experiment.experiment_id

    print("=" * 60)
    print("EXISTING MLFLOW EXPERIMENT FOUND")
    print("=" * 60)

    print(
        "Experiment ID     :",
        experiment.experiment_id
    )

    print(
        "Experiment Name   :",
        experiment.name
    )

    print(
        "Artifact Location :",
        experiment.artifact_location
    )


# Set experiment
mlflow.set_experiment(
    EXPERIMENT_NAME
)


# ============================================================
# 5. DISPLAY MLFLOW CONFIGURATION
# ============================================================

print("\n" + "=" * 60)
print("MLFLOW CONFIGURATION")
print("=" * 60)

print(
    "Project Root    :",
    PROJECT_ROOT
)

print(
    "MLflow Database :",
    MLFLOW_DB
)

print(
    "Artifact Root   :",
    MLFLOW_ARTIFACTS_DIR
)

print(
    "Tracking URI    :",
    mlflow.get_tracking_uri()
)

print(
    "Experiment      :",
    EXPERIMENT_NAME
)

print("=" * 60)


# ============================================================
# 6. TRAIN AND TRACK FUNCTION
# ============================================================

def train_and_track(
    run_name="RandomForest_Baseline",
    params=None
):

    # --------------------------------------------------------
    # Default Random Forest parameters
    # --------------------------------------------------------

    if params is None:

        params = {
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42,
            "class_weight": "balanced"
        }


    print(
        f"\n--- Starting MLflow Run: {run_name} ---"
    )


    # ========================================================
    # 7. LOAD PROCESSED DATA
    # ========================================================

    X_train_path = (
        PROCESSED_DIR /
        "X_train_final.npy"
    )

    X_test_path = (
        PROCESSED_DIR /
        "X_test_final.npy"
    )

    y_train_path = (
        PROCESSED_DIR /
        "y_train.npy"
    )

    y_test_path = (
        PROCESSED_DIR /
        "y_test.npy"
    )


    # Check files before loading
    required_files = [
        X_train_path,
        X_test_path,
        y_train_path,
        y_test_path
    ]

    for file_path in required_files:

        if not file_path.exists():

            raise FileNotFoundError(
                f"Required file not found:\n{file_path}"
            )


    # Load arrays
    X_train = np.load(
        X_train_path
    )

    X_test = np.load(
        X_test_path
    )

    y_train = np.load(
        y_train_path
    )

    y_test = np.load(
        y_test_path
    )


    print(
        "Training data shape:",
        X_train.shape
    )

    print(
        "Testing data shape :",
        X_test.shape
    )


    # ========================================================
    # 8. START MLFLOW RUN
    # ========================================================

    with mlflow.start_run(
        run_name=run_name
    ):


        # ====================================================
        # 9. LOG PARAMETERS
        # ====================================================

        mlflow.log_params(
            params
        )

        mlflow.log_param(
            "model_family",
            "RandomForest"
        )


        # ====================================================
        # 10. TRAIN RANDOM FOREST
        # ====================================================

        model = RandomForestClassifier(
            **params
        )

        model.fit(
            X_train,
            y_train
        )

        print(
            "Model training completed."
        )


        # ====================================================
        # 11. MAKE PREDICTIONS
        # ====================================================

        y_pred = model.predict(
            X_test
        )

        y_prob = model.predict_proba(
            X_test
        )[:, 1]


        # ====================================================
        # 12. CALCULATE METRICS
        # ====================================================

        metrics = {

            "accuracy": accuracy_score(
                y_test,
                y_pred
            ),

            "precision": precision_score(
                y_test,
                y_pred,
                zero_division=0
            ),

            "recall": recall_score(
                y_test,
                y_pred,
                zero_division=0
            ),

            "f1_score": f1_score(
                y_test,
                y_pred,
                zero_division=0
            ),

            "roc_auc": roc_auc_score(
                y_test,
                y_prob
            )
        }


        # ====================================================
        # 13. LOG METRICS
        # ====================================================

        mlflow.log_metrics(
            metrics
        )


        print(
            f"Metrics logged: "
            f"Accuracy = {metrics['accuracy']:.4f} | "
            f"Precision = {metrics['precision']:.4f} | "
            f"Recall = {metrics['recall']:.4f} | "
            f"F1 = {metrics['f1_score']:.4f} | "
            f"ROC-AUC = {metrics['roc_auc']:.4f}"
        )


        # ====================================================
        # 14. CONFUSION MATRIX
        # ====================================================

        fig_cm, ax_cm = plt.subplots(
            figsize=(6, 5)
        )


        ConfusionMatrixDisplay.from_predictions(
            y_test,
            y_pred,
            ax=ax_cm,
            cmap="Blues"
        )


        ax_cm.set_title(
            f"Confusion Matrix - {run_name}"
        )


        cm_path = (
            ARTIFACTS_DIR /
            "confusion_matrix.png"
        )


        fig_cm.savefig(
            cm_path,
            bbox_inches="tight"
        )


        plt.close(
            fig_cm
        )


        # Log confusion matrix
        mlflow.log_artifact(
            str(cm_path),
            artifact_path="plots"
        )


        print(
            "Confusion matrix logged."
        )


        # ====================================================
        # 15. ROC CURVE
        # ====================================================

        fig_roc, ax_roc = plt.subplots(
            figsize=(6, 5)
        )


        RocCurveDisplay.from_predictions(
            y_test,
            y_prob,
            ax=ax_roc
        )


        ax_roc.set_title(
            f"ROC Curve - {run_name}"
        )


        roc_path = (
            ARTIFACTS_DIR /
            "roc_curve.png"
        )


        fig_roc.savefig(
            roc_path,
            bbox_inches="tight"
        )


        plt.close(
            fig_roc
        )


        # Log ROC curve
        mlflow.log_artifact(
            str(roc_path),
            artifact_path="plots"
        )


        print(
            "ROC curve logged."
        )


        # ====================================================
        # 16. LOG DATASET METADATA
        # ====================================================

        metadata_path = (
            PROCESSED_DIR /
            "dataset_metadata.json"
        )


        if metadata_path.exists():

            mlflow.log_artifact(
                str(metadata_path),
                artifact_path="metadata"
            )

            print(
                "Dataset metadata logged."
            )

        else:

            print(
                "WARNING: dataset_metadata.json not found."
            )


        # ====================================================
        # 17. LOG MODEL TO MLFLOW
        # ====================================================

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model"
        )


        print(
            "Model logged to MLflow."
        )


        # ====================================================
        # 18. SAVE LOCAL MODEL
        # ====================================================

        model_path = (
            MODELS_DIR /
            "random_forest_model.pkl"
        )


        joblib.dump(
            model,
            model_path
        )


        print(
            "Local model saved to:",
            model_path
        )


        # ====================================================
        # 19. DISPLAY RUN INFORMATION
        # ====================================================

        run = mlflow.active_run()

        if run is not None:

            print(
                "MLflow Run ID:",
                run.info.run_id
            )


        # ====================================================
        # 20. SUCCESS
        # ====================================================

        print(
            f"\nRun '{run_name}' successfully tracked!"
        )


# ============================================================
# 21. MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # RUN 1: RANDOM FOREST BASELINE
    # --------------------------------------------------------

    train_and_track(
        run_name="RandomForest"
    )


    # --------------------------------------------------------
    # RUN 2: RANDOM FOREST TUNED
    # --------------------------------------------------------

    # Uncomment this section when you want a second run.

    # tuned_params = {
    #     "n_estimators": 200,
    #     "max_depth": 5,
    #     "random_state": 42,
    #     "class_weight": "balanced"
    # }

    # train_and_track(
    #     run_name="RandomForest_Shallow",
    #     params=tuned_params
    # )


    # --------------------------------------------------------
    # RUN 3: LOGISTIC REGRESSION
    # --------------------------------------------------------

    # This is kept disabled for now.
    # Enable it later if your Lab 4 requires comparison
    # between Random Forest and Logistic Regression.