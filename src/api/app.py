# src/api/app.py
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="Passos Mágicos - Defasagem API")

_MODEL: Any | None = None


def _load_model() -> Any:
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    model_path = os.getenv("MODEL_PATH") or os.getenv("ARTIFACT_PATH")
    if not model_path:
        raise RuntimeError("MODEL_PATH/ARTIFACT_PATH não definido.")
    _MODEL = joblib.load(model_path)
    return _MODEL


class PredictRequest(BaseModel):
    # aceita ambos: payload flat e payload { "features": {...} }
    features: Optional[Dict[str, Any]] = None
    model_config = {"extra": "allow"}


class PredictResponse(BaseModel):
    prediction: int = Field(..., ge=0, le=1)
    proba: float = Field(..., ge=0.0, le=1.0)


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    model = _load_model()

    payload = req.features if req.features is not None else req.model_dump(exclude={"features"})
    X = pd.DataFrame([payload])

    proba = float(model.predict_proba(X)[:, 1][0])
    pred = int(proba >= 0.5)
    return PredictResponse(prediction=pred, proba=proba)