from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    # aceita payload flat e payload {"features": {...}}
    features: Optional[Dict[str, Any]] = None
    model_config = {"extra": "allow"}


class PredictResponse(BaseModel):
    prediction: int = Field(..., ge=0, le=1)
    proba: float = Field(..., ge=0.0, le=1.0)
    threshold: float = Field(..., ge=0.0, le=1.0)
    model_path: str