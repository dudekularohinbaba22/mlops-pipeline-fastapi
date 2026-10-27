# MLOps Pipeline

End-to-end ML pipeline: train → track (MLflow) → gate → serve (FastAPI) → containerize (Docker) → CI/CD (GitHub Actions) → monitor (Prometheus + Grafana).

## Architecture
```
train.py ──► MLflow runs ──► promote only if accuracy >= current
                                   │
                              models/model.joblib
                                   │
GitHub Actions: lint → test → train → build image → push to GHCR → deploy
                                   │
                  FastAPI (/predict, /health, /metrics)
                                   │
                        Prometheus ──► Grafana
```

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m src.train            # trains, logs to MLflow, saves model
mlflow ui                      # http://localhost:5000
uvicorn src.app:app --reload   # http://localhost:8000/docs
pytest -q
```

## Run with Docker (API + Prometheus + Grafana)
```bash
python -m src.train
docker compose up --build
```
- API docs: http://localhost:8000/docs
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin) → add Prometheus data source `http://prometheus:9090`

## Example request
```bash
curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"features": [17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189]}'
```

## Grafana queries
- Request rate: `rate(predictions_total[1m])`
- p95 latency: `histogram_quantile(0.95, rate(prediction_latency_seconds_bucket[5m]))`
- Class balance (drift signal): `sum by (label) (predictions_total)`

## Next steps
Kubernetes manifests, DVC for data versioning, Evidently for drift reports.
