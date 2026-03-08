#src/api/routers/infra.py
from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query, Request

from src.api.model_loader import load_model as load_loaded_model, resolve_model_key

router = APIRouter(tags=["infra"])

MODEL_KEY_MAP = {
    "default": "default",
    "tree": "tree",
    "logreg": "logreg",
}


def _normalizar_model_key(model_key: Optional[str]) -> Optional[str]:
    if model_key is None:
        return None
    return MODEL_KEY_MAP.get(model_key, model_key)


def _extrair_features_esperadas(model: Any, meta: dict) -> list[str]:
    for key in ("raw_features", "feature_columns", "input_features"):
        cols = meta.get(key)
        if isinstance(cols, list) and all(isinstance(c, str) for c in cols):
            return cols

    cols = getattr(model, "feature_names_in_", None)
    if cols is not None:
        return [str(c) for c in list(cols)]

    return []


@router.get(
    "/infra/health",
    summary="Liveness probe",
    description="Endpoint simples de liveness probe.",
)
def health() -> Dict[str, str]:
    return {"status": "ok"}


@router.get(
    "/infra/model",
    summary="Exibe metadados do modelo carregado",
)
def get_model_info(
    request: Request,
    model_key: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    internal_key = _normalizar_model_key(model_key)
    resolved_key = resolve_model_key(internal_key)

    cache = getattr(request.app.state, "models_by_key", None)
    if cache is None:
        cache = {}
        request.app.state.models_by_key = cache

    try:
        if resolved_key in cache:
            loaded = cache[resolved_key]
        else:
            loaded = load_loaded_model(model_key=resolved_key, return_meta=True)
            cache[resolved_key] = loaded
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    model = loaded.model
    meta = loaded.meta if isinstance(loaded.meta, dict) else {}
    features_esperadas = _extrair_features_esperadas(model, meta)

    threshold = meta.get("threshold", 0.5)
    try:
        threshold = float(threshold)
    except Exception:
        threshold = 0.5

    return {
        "status": "ok",
        "model_key": resolved_key,
        "model_path": str(meta.get("model_path", "unknown")),
        "threshold": threshold,
        "n_features_esperadas": len(features_esperadas),
        "features_esperadas": features_esperadas,
        "meta_keys": sorted(list(meta.keys())),
    }


@router.get(
    "/infra/smoke",
    summary="Executa verificação rápida da API e do modelo",
)
def smoke(
    request: Request,
    model_key: Optional[str] = Query(default=None),
    dry_run: bool = Query(default=False),
) -> Dict[str, Any]:
    internal_key = _normalizar_model_key(model_key)
    resolved_key = resolve_model_key(internal_key)

    if dry_run:
        return {
            "status": "ok",
            "api": "up",
            "model_ok": True,
            "model_key": resolved_key,
            "model_path": "dry-run",
            "threshold": 0.5,
            "n_features_esperadas": 0,
            "features_esperadas": [],
            "meta_keys": [],
            "dry_run": {"executado": True},
        }

    cache = getattr(request.app.state, "models_by_key", None)
    if cache is None:
        cache = {}
        request.app.state.models_by_key = cache

    try:
        if resolved_key in cache:
            loaded = cache[resolved_key]
        else:
            loaded = load_loaded_model(model_key=resolved_key, return_meta=True)
            cache[resolved_key] = loaded
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    model = loaded.model
    meta = loaded.meta if isinstance(loaded.meta, dict) else {}
    features_esperadas = _extrair_features_esperadas(model, meta)

    threshold = meta.get("threshold", 0.5)
    try:
        threshold = float(threshold)
    except Exception:
        threshold = 0.5

    return {
        "status": "ok",
        "api": "up",
        "model_ok": True,
        "model_key": resolved_key,
        "model_path": str(meta.get("model_path", "unknown")),
        "threshold": threshold,
        "n_features_esperadas": len(features_esperadas),
        "features_esperadas": features_esperadas,
        "meta_keys": sorted(list(meta.keys())),
        "dry_run": {"executado": False},
    }


__all__ = [
    "router",
    "get_model_info",
    "health",
    "smoke",
]