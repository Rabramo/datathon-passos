#src/api/app.py
from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import Query, Request
from fastapi.openapi.utils import get_openapi

from src.api.main import app
from src.api.model_loader import load_model as load_loaded_model, resolve_model_key


def _extrair_features_esperadas(model: Any, meta: dict) -> list[str]:
    for key in ("raw_features", "feature_columns", "input_features"):
        cols = meta.get(key)
        if isinstance(cols, list) and all(isinstance(c, str) for c in cols):
            return cols

    cols = getattr(model, "feature_names_in_", None)
    if cols is not None:
        return [str(c) for c in list(cols)]

    return []


def _custom_openapi() -> dict:
    if app.openapi_schema:
        return app.openapi_schema

    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    paths = schema.setdefault("paths", {})
    paths.setdefault(
        "/docs",
        {
            "get": {
                "summary": "Swagger UI",
                "description": "Documentação interativa da API.",
                "responses": {
                    "200": {
                        "description": "Swagger UI disponível.",
                    }
                },
            }
        },
    )

    app.openapi_schema = schema
    return app.openapi_schema


app.openapi = _custom_openapi


@app.get(
    "/smoke",
    tags=["Compat"],
    summary="Executa verificação rápida da API e do modelo",
)
def smoke_legacy(
    request: Request,
    model_key: Optional[str] = Query(default=None),
    dry_run: bool = Query(default=False),
) -> Dict[str, Any]:
    resolved_key = resolve_model_key(model_key)

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
            "dry_run": {
                "executado": True,
                "estrategia": "zeros",
            },
        }

    cache = getattr(request.app.state, "models_by_key", None)
    if cache is None:
        cache = {}
        request.app.state.models_by_key = cache

    if resolved_key in cache:
        loaded = cache[resolved_key]
    else:
        loaded = load_loaded_model(model_key=resolved_key, return_meta=True)
        cache[resolved_key] = loaded

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
        "dry_run": {
            "executado": False,
            "estrategia": "zeros",
        },
    }


__all__ = ["app", "load_loaded_model"]