"""Print a markdown table of all MLflow runs (used for the GitHub Actions job summary)."""
import sys
from pathlib import Path

import mlflow

store = Path(sys.argv[1] if len(sys.argv) > 1 else "mlflow_data")
mlflow.set_tracking_uri(f"sqlite:///{(store / 'mlflow.db').resolve().as_posix()}")
runs = mlflow.search_runs(experiment_names=["heart-disease-classification"],
                          order_by=["metrics.cv_roc_auc_mean DESC"])
cols = ["tags.mlflow.runName", "metrics.cv_accuracy_mean", "metrics.cv_precision_mean",
        "metrics.cv_recall_mean", "metrics.cv_roc_auc_mean"]
t = runs[cols].rename(columns=lambda c: c.split(".")[-1]).round(3)
print(f"### MLflow experiment: {len(runs)} runs\n")
print(t.to_markdown(index=False))
