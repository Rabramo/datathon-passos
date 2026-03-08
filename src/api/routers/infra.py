from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query, Request

from src.api.model_loader import load_model as load_loaded_model, resolve_model_key

router = APIRouter(
    prefix="/infra",
    tags=["Infra"],
)

MODEL_KEY_MAP = {
    "Dummy": "dummy",
    "Regressão Logística": "logreg",
    "Árvore de Decisão": "tree",
    "CatBoost": "cat",
    "Random Forest": "rf",
    "XGBoost": "xgb",
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
    "/health",
    summary="Verifica se a API está viva",
    description="Endpoint simples de liveness probe.",
)
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get(
    "/smoke",
    summary="Executa verificação rápida da API e do modelo",
    description=(
        "Valida se a API está de pé e tenta carregar o modelo configurado. "
        "Retorna metadados úteis para depuração, incluindo as features esperadas "
        "quando disponíveis."
    ),
    responses={
        200: {
            "description": "Verificação concluída com sucesso",
            "content": {
                "application/json": {
                    "example": {
                        "status": "ok",
                        "api": "up",
                        "model_ok": True,
                        "model_key": "logreg",
                        "model_path": "artifacts/models/model_logreg_20260303_213446.joblib",
                        "threshold": 0.5,
                        "n_features_esperadas": 24,
                        "features_esperadas": [
                            "fase",
                            "turma",
                            "ano_nasc",
                            "genero",
                        ],
                    }
                }
            },
        }
    },
)
def smoke(
    request: Request,
    model_key: Optional[str] = Query(
        default=None,
        description="Chave do modelo a ser validado no smoke test.",
    ),
) -> Dict[str, Any]:
    internal_key = _normalizar_model_key(model_key)

    cache = getattr(request.app.state, "models_by_key", None)
    if cache is None:
        cache = {}
        request.app.state.models_by_key = cache

    resolved_key = resolve_model_key(internal_key)

    try:
        if resolved_key in cache:
            loaded = cache[resolved_key]
        else:
            loaded = load_loaded_model(model_key=resolved_key, return_meta=True)
            cache[resolved_key] = loaded
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Falha no smoke ao carregar modelo '{resolved_key}': {type(exc).__name__}: {exc}",
        ) from exc

    model = loaded.model
    meta = loaded.meta or {}

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
    }