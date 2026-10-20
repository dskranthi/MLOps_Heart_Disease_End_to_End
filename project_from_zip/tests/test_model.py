
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data import FEATURES, TARGET
from src.features import build_preprocessor


def make_pipe(df):
    pipe = Pipeline([("pre", build_preprocessor()),
                     ("clf", LogisticRegression(max_iter=500))])
    return pipe.fit(df[FEATURES], df[TARGET])


def test_preprocessor_handles_nans_and_output_is_finite(synthetic_clean):
    Xt = build_preprocessor().fit_transform(synthetic_clean[FEATURES])
    assert np.isfinite(Xt.toarray() if hasattr(Xt, "toarray") else Xt).all()


def test_pipeline_predicts_valid_probabilities(synthetic_clean):
    pipe = make_pipe(synthetic_clean)
    proba = pipe.predict_proba(synthetic_clean[FEATURES].head(5))
    assert proba.shape == (5, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_pipeline_ignores_unseen_category(synthetic_clean):
    pipe = make_pipe(synthetic_clean)
    row = synthetic_clean[FEATURES].head(1).copy()
    row["cp"] = 99
    assert pipe.predict(row)[0] in (0, 1)


def test_train_script_end_to_end(synthetic_clean, tmp_path, monkeypatch):
    """Runs the real training script on synthetic data (reduced grid) and checks artifacts."""
    import src.train as t
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)  # may be set by an earlier run in the same shell
    monkeypatch.setattr(t, "candidates", lambda: [
        ("lr", {"model": "LogisticRegression", "C": 1.0},
         Pipeline([("pre", build_preprocessor()), ("clf", LogisticRegression(max_iter=500))]))])
    data = tmp_path / "d.csv"
    synthetic_clean.to_csv(data, index=False)
    res = t.main(data, tmp_path / "art", tmp_path / "model", tmp_path / "mlflow")
    assert 0 <= res["test_roc_auc"] <= 1
    assert (tmp_path / "model" / "heart_disease_pipeline.joblib").exists()
    assert (tmp_path / "art" / "final_metrics.json").exists()
    assert (tmp_path / "mlflow" / "mlflow.db").exists()
