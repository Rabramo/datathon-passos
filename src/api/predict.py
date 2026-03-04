# src/api/predict.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Request

from src.api.model_loader import load_model, LoadedModel
from src.api.schemas import PredictRequest, PredictResponse

router = APIRouter()


DROP_NON_FEATURE_COLS = {"y", "year_t", "year_t1", "ano"}


def _extract_features(payload: PredictRequest) -> Dict[str, Any]:
    """
    Accepts:
      - {"features": {...}}
      - flat JSON (extra fields) because PredictRequest allows extra keys
    """
    if payload.features is not None:
        if not isinstance(payload.features, dict):
            raise HTTPException(status_code=422, detail="'features' must be an object/dict")
        return dict(payload.features)

    # flat: pydantic stores extra fields in model_dump()
    data = payload.model_dump(exclude={"features"})
    return dict(data)


def _expected_columns(model: Any) -> Optional[list[str]]:
    """
    Try to get the columns the pipeline was fitted on.
    """
    cols = getattr(model, "feature_names_in_", None)
    if cols is not None:
        return list(cols)

    # try pipeline.named_steps["preprocess"].feature_names_in_
    try:
        pre = model.named_steps.get("preprocess")  # type: ignore[attr-defined]
        cols = getattr(pre, "feature_names_in_", None)
        if cols is not None:
            return list(cols)
    except Exception:
        pass

    return None


def _get_loaded_model(request: Request) -> LoadedModel:
    """
    Uses the model cached at startup if available; otherwise loads now.
    """
    # Preferred: app.state.loaded_model_meta set at startup
    if hasattr(request.app.state, "loaded_model") and hasattr(request.app.state, "model_meta"):
        return LoadedModel(model=request.app.state.loaded_model, meta=request.app.state.model_meta)

    # Backwards compatible: only model is cached
    if hasattr(request.app.state, "loaded_model"):
        # reload meta from disk so response can include model_path
        lm = load_model(return_meta=True)  # default tree
        request.app.state.loaded_model = lm.model
        request.app.state.model_meta = lm.meta
        return lm

    # If nothing cached, load now
    lm = load_model(return_meta=True)
    request.app.state.loaded_model = lm.model
    request.app.state.model_meta = lm.meta
    return lm


def _get_threshold(meta: dict, default: float = 0.5) -> float:
    """
    Uses threshold if present in meta; otherwise fallback default.
    (Your latest_tree.json currently has no threshold.)
    """
    thr = meta.get("threshold", None)
    try:
        return float(thr) if thr is not None else float(default)
    except Exception:
        return float(default)


@router.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest, request: Request) -> PredictResponse:
    lm = _get_loaded_model(request)
    model = lm.model
    meta = lm.meta or {}

    feats = _extract_features(payload)

    # drop non-features if client mistakenly sends them
    for c in list(feats.keys()):
        if c in DROP_NON_FEATURE_COLS:
            feats.pop(c, None)

    expected = _expected_columns(model)

    if expected:
        # build row with all expected columns, missing -> None
        row = {c: feats.get(c, None) for c in expected}
        X = pd.DataFrame([row], columns=expected)
    else:
        # fallback: use whatever the client sent
        X = pd.DataFrame([feats])

    # Ensure predict_proba exists
    if not hasattr(model, "predict_proba"):
        raise HTTPException(status_code=500, detail="Loaded model does not support predict_proba")

    try:
        proba = float(model.predict_proba(X)[0][1])
    except Exception as e:
        # return an explicit message to help debugging client payload/columns
        raise HTTPException(status_code=500, detail=f"Prediction failed: {type(e).__name__}: {e}")

    threshold = _get_threshold(meta, default=0.5)
    pred = int(proba >= threshold)

    model_path = str(meta.get("model_path", "unknown"))

    return PredictResponse(
        prediction=pred,
        proba=proba,
        threshold=threshold,
        model_path=model_path,
    )