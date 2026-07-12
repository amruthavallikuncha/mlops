import json
import os
from datetime import datetime, timezone

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/heart-risk-matplotlib")

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from heart_risk.config import (
    FEATURES,
    METADATA_PATH,
    MODEL_DIR,
    MODEL_PATH,
    PLOT_DIR,
    PROCESSED_DATA_PATH,
    RANDOM_STATE,
    TARGET,
)
from heart_risk.modeling import candidate_models


def evaluate(model, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    predicted = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]
    return {
        "test_accuracy": accuracy_score(y_test, predicted),
        "test_precision": precision_score(y_test, predicted, zero_division=0),
        "test_recall": recall_score(y_test, predicted, zero_division=0),
        "test_f1": f1_score(y_test, predicted, zero_division=0),
        "test_roc_auc": roc_auc_score(y_test, probabilities),
    }


def save_evaluation_plots(model, x_test, y_test, model_name: str) -> list[str]:
    paths = []
    specifications = [
        ("confusion_matrix", ConfusionMatrixDisplay, {"cmap": "Blues"}),
        ("roc_curve", RocCurveDisplay, {}),
    ]
    for kind, display_class, kwargs in specifications:
        _, axis = plt.subplots()
        display = display_class.from_estimator(model, x_test, y_test, ax=axis, **kwargs)
        display.ax_.set_title(
            f"{model_name.replace('_', ' ').title()} – {kind.replace('_', ' ').title()}"
        )
        path = PLOT_DIR / f"{model_name}_{kind}.png"
        plt.tight_layout()
        plt.savefig(path, dpi=180)
        plt.close()
        paths.append(str(path))
    return paths


def main() -> None:
    data = pd.read_csv(PROCESSED_DATA_PATH)
    x_train, x_test, y_train, y_test = train_test_split(
        data[FEATURES],
        data[TARGET],
        test_size=0.20,
        stratify=data[TARGET],
        random_state=RANDOM_STATE,
    )
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI", f"sqlite:///{MODEL_DIR.parent.parent / 'mlflow.db'}"
    )
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("uci-heart-disease-risk")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = []
    fitted = {}

    for name, (pipeline, grid) in candidate_models().items():
        search = GridSearchCV(pipeline, grid, scoring="roc_auc", cv=cv, n_jobs=1, refit=True)
        search.fit(x_train, y_train)
        metrics = {
            "cv_roc_auc": float(search.best_score_),
            **evaluate(search.best_estimator_, x_test, y_test),
        }
        plots = save_evaluation_plots(search.best_estimator_, x_test, y_test, name)
        with mlflow.start_run(run_name=name):
            mlflow.log_params(
                {
                    key.replace("classifier__", ""): value
                    for key, value in search.best_params_.items()
                }
            )
            mlflow.log_metrics(metrics)
            for plot in plots:
                mlflow.log_artifact(plot, artifact_path="evaluation")
            mlflow.sklearn.log_model(
                search.best_estimator_, artifact_path="model", input_example=x_test.head(2)
            )
        fitted[name] = search.best_estimator_
        results.append({"model": name, "best_params": search.best_params_, **metrics})

    winner = max(results, key=lambda item: (item["cv_roc_auc"], item["test_roc_auc"]))
    best_model = fitted[winner["model"]]
    joblib.dump(best_model, MODEL_PATH)
    metadata = {
        "model_name": winner["model"],
        "selected_by": "highest 5-fold cross-validated ROC-AUC on the training partition",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_state": RANDOM_STATE,
        "features": FEATURES,
        "training_records": len(x_train),
        "held_out_records": len(x_test),
        "metrics": {k: v for k, v in winner.items() if k.startswith(("cv_", "test_"))},
        "all_results": results,
        "limitations": (
            "Model trained on a small historical cohort - "
            "not for clinical diagnosis."
        ),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n")
    (MODEL_DIR / "model_comparison.csv").write_text(pd.DataFrame(results).to_csv(index=False))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
