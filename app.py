"""FastAPI model server with request logging and Prometheus metrics."""
import logging
import os
import time
import uuid
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("heart-api")

MODEL_PATH = Path(os.getenv("MODEL_PATH", Path(__file__).parent / "model" / "heart_disease_pipeline.joblib"))
FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
            "exang", "oldpeak", "slope", "ca", "thal"]
model = joblib.load(MODEL_PATH)

REQS = Counter("api_requests_total", "HTTP requests", ["method", "path", "status"])
LATENCY = Histogram("api_request_latency_seconds", "Request latency", ["path"])
PREDS = Counter("api_predictions_total", "Predictions by class", ["label"])

app = FastAPI(title="Heart Disease Prediction API", version="2.0.0")


class Features(BaseModel):
    age: float = Field(..., ge=1, le=120)
    sex: int = Field(..., ge=0, le=1)
    cp: int = Field(..., ge=1, le=4, description="chest pain type")
    trestbps: float = Field(..., ge=50, le=260)
    chol: float = Field(..., ge=50, le=700)
    fbs: int = Field(..., ge=0, le=1)
    restecg: int = Field(..., ge=0, le=2)
    thalach: float = Field(..., ge=40, le=250)
    exang: int = Field(..., ge=0, le=1)
    oldpeak: float = Field(..., ge=0, le=10)
    slope: int = Field(..., ge=1, le=3)
    ca: float = Field(..., ge=0, le=4)
    thal: float = Field(..., ge=3, le=7)


class PredictRequest(BaseModel):
    features: Features


@app.middleware("http")
async def log_requests(request: Request, call_next):
    rid = uuid.uuid4().hex[:8]
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        dur = time.perf_counter() - start
        path = request.url.path
        if path != "/metrics":
            REQS.labels(request.method, path, str(status)).inc()
            LATENCY.labels(path).observe(dur)
        logger.info("req_id=%s method=%s path=%s status=%s latency_ms=%.1f",
                    rid, request.method, path, status, dur * 1000)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        row = pd.DataFrame([req.features.model_dump()], columns=FEATURES)
        pred = int(model.predict(row)[0])
        proba = model.predict_proba(row)[0]
    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=400, detail="Unable to process feature values")
    label = "disease_present" if pred == 1 else "no_disease"
    PREDS.labels(label).inc()
    logger.info("prediction=%s disease_probability=%.3f", label, proba[1])
    return {"prediction": pred, "label": label,
            "confidence": float(max(proba)), "disease_probability": float(proba[1])}
