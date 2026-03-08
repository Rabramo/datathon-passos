#src/api/routers/infra.py
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request

from src.api.model_loader import load_model as load_loaded_model, resolve_model_key

router = APIRouter(prefix="/infra", tags=["Infra"])


def _extract_expected_features(model: Any, meta: dict[str, Any]) -> list[str]:
    for key in ("raw_features", "feature_columns", "input_features"):
        cols = meta.get(key)
        if isinstance(cols, list) and all(isinstance(col, str) for col in cols):
            return cols

    cols = getattr(model, "feature_names_in_", None)
    if cols is not None:
        return [str(col) for col in list(cols)]

    return []


def _get_model_cache(request: Request) -> dict[str, Any]:
    cache = getattr(request.app.state, "models_by_key", None)
    if cache is None:
        cache = {}
        request.app.state.models_by_key = cache
    return cache


def _load_model_with_cache(request: Request, model_key: str | None) -> tuple[str, Any]:
    cache = _get_model_cache(request)

    try:
        resolved_key = resolve_model_key(model_key)
        if resolved_key in cache:
            loaded = cache[resolved_key]
        else:
            loaded = load_loaded_model(model_key=resolved_key, return_meta=True)
            cache[resolved_key] = loaded
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return resolved_key, loaded


def _build_model_payload(resolved_key: str, loaded: Any, api: str | None = None, dry_run: bool | None = None) -> dict[str, Any]:
    model = loaded.model
    meta = loaded.meta if isinstance(loaded.meta, dict) else {}
    expected_features = _extract_expected_features(model, meta)

    threshold = meta.get("threshold", 0.5)
    try:
        threshold = float(threshold)
    except Exception:
        threshold = 0.5

    payload: dict[str, Any] = {
        "status": "ok",
        "model_key": resolved_key,
        "model_path": str(meta.get("model_path", "unknown")),
        "threshold": threshold,
        "n_features_esperadas": len(expected_features),
        "features_esperadas": expected_features,
        "meta_keys": sorted(list(meta.keys())),
    }

    if api is not None:
        payload["api"] = api
        payload["model_ok"] = True

    if dry_run is not None:
        payload["dry_run"] = {"executado": dry_run, "estrategia": "zeros"}

    return payload


@router.get(
    "/health",
    tags=["Infra"],
    summary="Liveness probe",
    description="Endpoint simples de liveness probe.",
)
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get(
    "/model",
    tags=["Infra"],
    summary="Exibe metadados do modelo carregado",
)
def get_model_info(
    request: Request,
    model_key: str | None = Query(default=None),
) -> dict[str, Any]:
    resolved_key, loaded = _load_model_with_cache(request, model_key)
    return _build_model_payload(resolved_key=resolved_key, loaded=loaded)


@router.get(
    "/smoke",
    tags=["Infra"],
    summary="Executa verificação rápida da API e do modelo",
)
def smoke(
    request: Request,
    model_key: str | None = Query(default=None),
    dry_run: bool = Query(default=False),
) -> dict[str, Any]:
    if dry_run:
        try:
            resolved_key = resolve_model_key(model_key)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
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
            "dry_run": {"executado": True, "estrategia": "zeros"},
        }

    resolved_key, loaded = _load_model_with_cache(request, model_key)
    return _build_model_payload(
        resolved_key=resolved_key,
        loaded=loaded,
        api="up",
        dry_run=False,
    )


__all__ = [
    "router",
    "get_model_info",
    "health",
    "smoke",
]
