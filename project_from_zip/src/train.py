"""Train, cross-validate and tune models; log EVERY run to MLflow; save best pipeline."""
import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    ParameterGrid,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import FEATURES, TARGET
from src.features import build_preprocessor

SCORING = {"accuracy": "accuracy", "precision": "precision", "recall": "recall", "roc_auc": "roc_auc"}
SEED = 42


def candidates():
    """Yield (run_name, params, pipeline) for every model configuration tried."""
    for C in [0.1, 1.0, 10.0]:
        yield (f"logreg_C{C}", {"model": "LogisticRegression", "C": C},
               Pipeline([("pre", build_preprocessor()),
                         ("clf", LogisticRegression(C=C, max_iter=2000, class_weight="balanced",
                                                    random_state=SEED))]))
    grid = {"n_estimators": [200, 400], "max_depth": [None, 8], "min_samples_leaf": [1, 2]}
    for p in ParameterGrid(grid):
        yield (f"rf_n{p['n_estimators']}_d{p['max_depth']}_l{p['min_samples_leaf']}",
               {"model": "RandomForest", **p},
               Pipeline([("pre", build_preprocessor()),
                         ("clf", RandomForestClassifier(class_weight="balanced", random_state=SEED,
                                                        n_jobs=-1, **p))]))


def main(data_path: Path, out_dir: Path, model_dir: Path, mlflow_dir: Path) -> dict:
    df = pd.read_csv(data_path)
    X, y = df[FEATURES], df[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    mlflow_dir.mkdir(parents=True, exist_ok=True)
    exp = "heart-disease-classification"
    server_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if server_uri:  # remote MLflow server (e.g. the docker-compose "mlflow" service)
        mlflow.set_tracking_uri(server_uri)
    else:  # local SQLite store
        mlflow.set_tracking_uri(f"sqlite:///{(mlflow_dir / 'mlflow.db').resolve().as_posix()}")
        if mlflow.get_experiment_by_name(exp) is None:
            mlflow.create_experiment(exp, artifact_location=(mlflow_dir / "artifacts").resolve().as_uri())
    mlflow.set_experiment(exp)

    rows, best = [], None
    for name, params, pipe in candidates():
        with mlflow.start_run(run_name=name):
            mlflow.log_params({**params, "cv_folds": 5, "seed": SEED})
            s = cross_validate(pipe, X_train, y_train, cv=cv, scoring=SCORING, n_jobs=1)
            metrics = {}
            for m in SCORING:
                metrics[f"cv_{m}_mean"] = float(np.mean(s[f"test_{m}"]))
                metrics[f"cv_{m}_std"] = float(np.std(s[f"test_{m}"]))
            mlflow.log_metrics(metrics)
            oof = cross_val_predict(pipe, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            ConfusionMatrixDisplay.from_predictions(y_train, (oof >= 0.5).astype(int), ax=ax[0], cmap="Blues")
            RocCurveDisplay.from_predictions(y_train, oof, ax=ax[1])
            ax[0].set_title(f"{name}: CV confusion matrix")
            ax[1].set_title("CV ROC curve")
            fig.tight_layout()
            mlflow.log_figure(fig, "cv_evaluation.png")
            plt.close(fig)
            rows.append({"run": name, **params, **metrics})
            if best is None or metrics["cv_roc_auc_mean"] > best[0]:
                best = (metrics["cv_roc_auc_mean"], name, params, pipe)

    cv_df = pd.DataFrame(rows).sort_values("cv_roc_auc_mean", ascending=False)
    cv_df.to_csv(out_dir / "cv_results.csv", index=False)

    _, best_name, best_params, best_pipe = best
    best_pipe.fit(X_train, y_train)
    pred = best_pipe.predict(X_test)
    prob = best_pipe.predict_proba(X_test)[:, 1]
    final = {
        "selected_run": best_name, "selected_params": best_params,
        "cv_roc_auc": best[0],
        "test_accuracy": float(accuracy_score(y_test, pred)),
        "test_precision": float(precision_score(y_test, pred, zero_division=0)),
        "test_recall": float(recall_score(y_test, pred, zero_division=0)),
        "test_roc_auc": float(roc_auc_score(y_test, prob)),
        "n_train": len(X_train), "n_test": len(X_test),
    }
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=ax[0], cmap="Blues")
    RocCurveDisplay.from_predictions(y_test, prob, ax=ax[1])
    fig.tight_layout()
    fig.savefig(out_dir / "holdout_evaluation.png", dpi=150)
    plt.close(fig)
    (out_dir / "final_metrics.json").write_text(json.dumps(final, indent=2))

    with mlflow.start_run(run_name=f"FINAL_{best_name}"):
        mlflow.log_params({**best_params, "stage": "final_holdout"})
        mlflow.log_metrics({k: v for k, v in final.items() if k.startswith(("test_", "cv_"))})
        for f in ["cv_results.csv", "holdout_evaluation.png", "final_metrics.json"]:
            mlflow.log_artifact(str(out_dir / f))
        mlflow.sklearn.log_model(best_pipe, name="model", input_example=X_train.head(3),
                                 serialization_format="cloudpickle")

    joblib.dump(best_pipe, model_dir / "heart_disease_pipeline.joblib")
    print(json.dumps(final, indent=2))
    return final


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data/processed/heart_clean.csv"))
    ap.add_argument("--out", default=str(ROOT / "artifacts"))
    ap.add_argument("--model-dir", default=str(ROOT / "model"))
    ap.add_argument("--mlflow-dir", default=str(ROOT / "mlflow_data"))
    a = ap.parse_args()
    main(Path(a.data), Path(a.out), Path(a.model_dir), Path(a.mlflow_dir))
