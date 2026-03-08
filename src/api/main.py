# src/api/main.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import Body, FastAPI, Query, Request, Response

from src.api.routers.infra import router as infra_router
from src.api.routers.leaderboard import router as leaderboard_router
from src.api.routers.predict import (
    SwaggerModelKey,
    get_model_info,
    predict,
    predict_batch,
    select_features,
    router as predict_router,
)
from src.api.routers.train import router as train_router
from src.api.schemas import PredictRequest

app = FastAPI(
    title="Passos Mágicos - Defasagem API",
    version="1.0.0",
    description=(
        "API desenvolvida para o Datathon 2026/MLOps - Pós Tech FIAP Machine Learning Engineering\n\n"
        "**Aluno:** Rogerio Abramo A. Pretti | RA 363736 | Grupo 150 | 5MLET\n\n"
        "**Objetivo:** Estimar a probabilidade de um aluno ter defasagem de aprendizado "
        "(y=1) no ano t+1, para que a Associação Primeiro Passos possa tomar medidas preventivas."
    ),
    swagger_ui_parameters={
        "displayRequestDuration": True,
        "docExpansion": "list",
    },
)

app.include_router(infra_router)
app.include_router(leaderboard_router)
app.include_router(train_router)
app.include_router(predict_router)


@app.get("/model", tags=["Compat"], include_in_schema=False)
def model_legacy(
    request: Request,
    model_key: Optional[SwaggerModelKey] = Query(default=None),
):
    return get_model_info(request=request, model_key=model_key)


@app.post("/features/select", tags=["Compat"], include_in_schema=False)
def select_features_legacy(
    request: Request,
    name: str = Query(default="default"),
    features: list[str] = Query(default=[]),
):
    return select_features(request=request, name=name, features=features)


@app.post("/predict", tags=["Compat"], include_in_schema=False)
def predict_legacy(
    request: Request,
    response: Response,
    payload: PredictRequest = Body(...),
    threshold: Optional[float] = Query(default=None, ge=0.0, le=1.0),
    model_key: Optional[SwaggerModelKey] = Query(default=None),
):
    return predict(
        request=request,
        response=response,
        payload=payload,
        threshold=threshold,
        model_key=model_key,
    )


@app.post("/predict/batch", tags=["Compat"], include_in_schema=False)
def predict_batch_legacy(
    request: Request,
    items: List[Dict[str, Any]] = Body(...),
    model_key: Optional[SwaggerModelKey] = Query(default=None),
):
    return predict_batch(
        request=request,
        items=items,
        model_key=model_key,
    )