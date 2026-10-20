# Heart Disease Prediction — MLOps Assignment 01

End-to-end pipeline: data → EDA → training (MLflow) → packaged model → FastAPI → Docker → Kubernetes, with CI, logging and Prometheus/Grafana monitoring. Educational model, not for clinical use.

## Layout
| Path | Purpose |
|---|---|
| `scripts/download_data.py` | Downloads UCI data, writes `data/processed/heart_clean.csv` |
| `src/data.py`, `src/features.py` | Cleaning and preprocessing pipeline |
| `src/train.py` | CV + tuning, logs every run to MLflow, saves `model/*.joblib` |
| `app.py` | FastAPI: `/predict`, `/health`, `/metrics` |
| `tests/` | Pytest: data, model, training, API |
| `.github/workflows/ci.yml` | lint → test → train → docker build + smoke test |
| `Dockerfile`, `docker-compose.yml` | API image; API + Prometheus + Grafana |
| `k8s/deployment.yaml` | Deployment (2 replicas) + LoadBalancer Service |
| `notebooks/` | EDA / training / inference notebook |

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate            # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
python scripts/download_data.py
python src/train.py
pytest -q && ruff check .
uvicorn app:app --port 8000       # http://localhost:8000/docs
```
MLflow UI: `mlflow ui --backend-store-uri sqlite:///mlflow_data/mlflow.db`

## Docker
```bash
docker build -t heart-disease-api:latest .
docker run --rm -p 8000:8000 heart-disease-api:latest
bash scripts/sample_request.sh          # or use /docs
```

## MLflow
- Local: `python src/train.py`, then `mlflow ui --backend-store-uri sqlite:///mlflow_data/mlflow.db --port 5001` run from the project folder, then open http://localhost:5001. The server takes 10-20 seconds to start.
- Docker: `docker compose up -d mlflow` then `docker compose --profile train run --rm train` trains inside Docker and logs every run to the MLflow server (UI at http://localhost:5001) (set `MLFLOW_TRACKING_URI` to log to any server).
- GitHub: each Actions run prints an MLflow run table in the job summary and uploads `mlflow_data/` as an artifact; download it and run the local `mlflow ui` command above to browse the runs.

## Monitoring
`docker compose up --build` → API :8000, Prometheus :9090, Grafana :3000 (add panels on `api_requests_total`, `api_request_latency_seconds_bucket`, `api_predictions_total`). Request logs: `docker logs <container>`.

## Kubernetes
```bash
kubectl apply -f k8s/deployment.yaml
kubectl get pods,svc
kubectl port-forward svc/heart-disease-api 8080:80   # then http://localhost:8080/docs
```
Minikube: `minikube image load heart-disease-api:latest` first, and `minikube tunnel` for the LoadBalancer IP.
