# src/api/schemas.py

from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Any, Dict, Optional

class PredictRequest(BaseModel):
    features: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Features do aluno (ano t). Se omitido, o payload pode ser flat (campos no topo).",
        examples=[{"ra": "123", "genero": "F", "fase": 5, "ian": 10, "ida": 6.8}],
    )
    model_config = {"extra": "allow"}

class PredictResponse(BaseModel):
    prediction: int = Field(..., ge=0, le=1, description="Classe prevista: 1=risco defasagem, 0=em fase")
    proba: float = Field(..., ge=0.0, le=1.0, description="Probabilidade estimada de y=1 (risco)")
    threshold: float = Field(..., ge=0.0, le=1.0, description="Threshold aplicado para gerar prediction")
    model_path: str = Field(..., description="Caminho do artefato do modelo carregado")