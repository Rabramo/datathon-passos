# src/api/app.py
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException

from src.api.model_loader import LoadedModel, load_model
from src.api.predict import router as predict_router


def _load_default_model() -> LoadedModel:
    """
    Loads the default model (tree) via latest_tree.json when available.
    Always returns LoadedModel(model, meta).
    """
    lm = load_model(return_meta=True)  # default model_name="tree" inside loader
    if not isinstance(lm, LoadedModel):
        # Defensive: in case loader is changed to return tuple
        raise RuntimeError("load_model(return_meta=True) must return LoadedModel")
    return lm


def _get_loaded(app: FastAPI) -> LoadedModel:
    """
    Get cached model. If not loaded (e.g. tests not running lifespan),
    loads it on demand.
    """
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
        # Keep API up; /health will show model_loaded=false and /predict should return 503
        app.state.loaded = None
    yield


app = FastAPI(title="Passos Mágicos - Defasagem API", lifespan=lifespan)

# Routes
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
    """
    Optional: returns the model pointer currently loaded.
    Useful for auditing which artifact is serving predictions.
    """
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