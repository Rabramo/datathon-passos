from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
from fastapi import HTTPException

from src.api.model_loader import LoadedModel
from src.api.schemas import PredictRequest


def merge_payload(req: PredictRequest) -> Dict[str, Any]:
    """
    Se vier payload flat + features, mescla com precedência para features.
    """
    raw = req.model_dump()
    feats = raw.get("features") or {}
    if not isinstance(feats, dict):
        raise HTTPException(status_code=400, detail="Campo 'features' deve ser um objeto/dict.")

    top_level = {k: v for k, v in raw.items() if k != "features"}
    return {**top_level, **feats}


def to_aligned_dataframe(payload: Dict[str, Any], feature_cols: list[str]) -> pd.DataFrame:
    # fallback (modelo dummy sem preprocess): usa o payload como está
    if not feature_cols:
        return pd.DataFrame([payload])

    row = {c: payload.get(c, np.nan) for c in feature_cols}
    return pd.DataFrame([row])

def predict_one(loaded: LoadedModel, req: PredictRequest) -> tuple[int, float]:
    payload = merge_payload(req)
    X = to_aligned_dataframe(payload, loaded.feature_cols)

    try:
        proba = float(loaded.pipeline.predict_proba(X)[0, 1])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    pred = int(proba >= loaded.threshold)
    return pred, proba