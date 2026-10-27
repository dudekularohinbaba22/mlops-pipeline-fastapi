"""FastAPI service that serves the trained model with Prometheus metrics."""
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Histogram, make_asgi_app
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model.joblib"

PREDICTIONS = Counter("predictions_total", "Predictions served", ["label"])
LATENCY = Histogram("prediction_latency_seconds", "Prediction latency")

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if MODEL_PATH.exists():
        state.update(joblib.load(MODEL_PATH))
    yield
    state.clear()


app = FastAPI(title="MLOps Pipeline API", lifespan=lifespan)
app.mount("/metrics", make_asgi_app())


class PredictRequest(BaseModel):
    features: list[float] = Field(..., description="30 numeric features, in training order")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state}


@app.post("/predict")
def predict(req: PredictRequest):
    if "model" not in state:
        raise HTTPException(503, "Model not loaded. Run `python -m src.train` first.")
    expected = len(state["feature_names"])
    if len(req.features) != expected:
        raise HTTPException(422, f"Expected {expected} features, got {len(req.features)}")

    start = time.perf_counter()
    proba = state["model"].predict_proba([req.features])[0]
    idx = int(proba.argmax())
    label = state["target_names"][idx]
    LATENCY.observe(time.perf_counter() - start)
    PREDICTIONS.labels(label=label).inc()
    return {"prediction": label, "confidence": round(float(proba[idx]), 4)}
