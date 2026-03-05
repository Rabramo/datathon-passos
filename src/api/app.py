# src/api/app.py
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException

from src.api.model_loader import LoadedModel, load_model
from src.api.predict import router as predict_router


def _load_default_model() -> LoadedModel:
    # Default model is tree via latest_tree.json (handled in loader)
    lm = load_model(return_meta=True)
    if not isinstance(lm, LoadedModel):
        raise RuntimeError("load_model(return_meta=True) deve retornar LoadedModel")
    return lm


def _get_loaded(app: FastAPI) -> LoadedModel:
    loaded: Optional[LoadedModel] = getattr(app.state, "loaded", None)
    if loaded is None:
        loaded = _load_default_model()
        app.state.loaded = loaded
    return loaded


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Preload on normal runtime; tests may skip lifespan, so _get_loaded covers that.
    try:
        app.state.loaded = _load_default_model()
    except Exception:
        app.state.loaded = None
    yield


app = FastAPI(
    title="Passos Mágicos - Defasagem API",
    version="1.0.0",
    description=(
        "**API desenvolvida para o Datathon 2026/MLOps - Pós Tech FIAP Machine Learning Engineering**\n\n"
        "**Aluno**: Rogerio Abramo A. Pretti | RA 363736 | Grupo 150 | 5MLET\n\n"
        "**Objetivo**:\n"
        "Estimar a probabilidade de um aluno ter defasagem de aprendizado (y=1) no ano t+1, para que a Associação Primeiro Passos possa tomar medidas preventivas.\n\n"
        "**Machine Learn Model padrão**:\n"
        "Decision Tree carregado via artifacts/models/latest_tree.json.\n\n"
        "**Documentação**:\n"
        "A apresentação completa da API e regras de negócio estão em /docs."
    ),
    lifespan=lifespan,
)
app.include_router(predict_router)


@app.get("/health")
def health() -> dict[str, str]:
    loaded: Optional[LoadedModel] = getattr(app.state, "loaded", None)
    if loaded is None:
        return {"status": "ok", "model_loaded": "false"}

    meta = loaded.meta or {}
    return {
        "status": "ok",
        "model_loaded": "true",
        "model_path": str(meta.get("model_path", "unknown")),
        "model_name": str(meta.get("model_name", "unknown")),
        "run_id": str(meta.get("run_id", "unknown")),
    }


@app.get("/model")
def model_info() -> dict[str, str]:
    try:
        loaded = _get_loaded(app)
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

    meta = loaded.meta or {}
    return {
        "model_path": str(meta.get("model_path", "unknown")),
        "model_name": str(meta.get("model_name", "unknown")),
        "run_id": str(meta.get("run_id", "unknown")),
    }