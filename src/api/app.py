# src/api/app.py
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi

from src.api.model_loader import LoadedModel, load_model as load_loaded_model
from src.api.predict import router as predict_router
from src.api.feature_descriptions import router as feature_desc_router
from src.api.feature_descriptions import get_feature_descriptions_map

_START_TIME = time.time()
_OPENAPI_FEATURES_CACHE: Optional[List[str]] = None
_OPENAPI_FEATURES_SOURCE: Optional[str] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


def _uptime_seconds() -> int:
    return int(time.time() - _START_TIME)


def _get_cached_model(request: Request) -> Optional[LoadedModel]:
    loaded = getattr(request.app.state, "loaded", None)
    return loaded if isinstance(loaded, LoadedModel) else None


def _ensure_model_loaded(request: Request) -> LoadedModel:
    cached = _get_cached_model(request)
    if cached is not None:
        return cached
    lm = load_loaded_model(return_meta=True)
    request.app.state.loaded = lm
    return lm


def _extract_features_from_meta(meta: dict) -> Tuple[List[str], str]:
    for key in ("raw_features", "feature_columns", "input_features"):
        cols = meta.get(key)
        if isinstance(cols, list) and cols and all(isinstance(c, str) for c in cols):
            return cols, f"meta.{key}"
    return [], "meta.nao_contem_features"


def _extract_features_from_model(model: Any) -> Tuple[List[str], str]:
    # A) feature_names_in_
    cols = getattr(model, "feature_names_in_", None)
    if cols is not None:
        out = [str(c) for c in list(cols)]
        if out:
            return out, "model.feature_names_in_"

    # B) get_feature_names_out
    g = getattr(model, "get_feature_names_out", None)
    if callable(g):
        try:
            out = [str(c) for c in list(g())]
            if out:
                return out, "model.get_feature_names_out"
        except Exception:
            pass

    # C) pipeline.named_steps -> procurar transformador com get_feature_names_out
    named_steps = getattr(model, "named_steps", None)
    if isinstance(named_steps, dict) and named_steps:
        for step_name, step_obj in named_steps.items():
            g2 = getattr(step_obj, "get_feature_names_out", None)
            if callable(g2):
                try:
                    out = [str(c) for c in list(g2())]
                    if out:
                        return out, f"pipeline.step[{step_name}].get_feature_names_out"
                except Exception:
                    pass

            # ColumnTransformer costuma expor get_feature_names_out no próprio objeto
            g3 = getattr(step_obj, "get_feature_names_out", None)
            if callable(g3):
                try:
                    out = [str(c) for c in list(g3())]
                    if out:
                        return out, f"pipeline.step[{step_name}].ColumnTransformer.get_feature_names_out"
                except Exception:
                    pass

    return [], "model.nao_expoe_features"


def _get_expected_features(lm: LoadedModel) -> Tuple[List[str], str]:
    meta = lm.meta or {}
    model = lm.model

    # Preferência: features "brutas" declaradas no meta (contrato ideal do /predict)
    cols, src = _extract_features_from_meta(meta)
    if cols:
        return cols, src

    # Fallback: features do modelo/pipeline (pode ser pós-transformação)
    cols2, src2 = _extract_features_from_model(model)
    if cols2:
        return cols2, src2

    return [], "nenhuma"


def _build_predict_example_payload(expected_features: List[str]) -> Dict[str, Any]:
    # Placeholder simples: o objetivo é o Swagger renderizar todas as chaves.
    return {f: 0 for f in expected_features}


def _inject_predict_example_into_openapi(app: FastAPI) -> None:

    schema = app.openapi_schema
    if not schema:
        return

    global _OPENAPI_FEATURES_CACHE, _OPENAPI_FEATURES_SOURCE

    if _OPENAPI_FEATURES_CACHE is None or _OPENAPI_FEATURES_SOURCE is None:
        lm = load_loaded_model(return_meta=True)
        feats, src = _get_expected_features(lm)
        _OPENAPI_FEATURES_CACHE = feats
        _OPENAPI_FEATURES_SOURCE = src

    feats = _OPENAPI_FEATURES_CACHE or []
    src = _OPENAPI_FEATURES_SOURCE or "nenhuma"

    if not feats:
        return

    paths = schema.get("paths", {})
    predict_item = paths.get("/predict")
    if not isinstance(predict_item, dict):

        return

    post_op = predict_item.get("post")
    if not isinstance(post_op, dict):
        return

    request_body = post_op.get("requestBody")
    if not isinstance(request_body, dict):
        return

    content = request_body.get("content", {})
    app_json = content.get("application/json")
    if not isinstance(app_json, dict):
        return

    app_json["examples"] = {
        "features_esperadas": {
            "summary": "Payload com todas as features esperadas",
            "description": f"Features extraídas via {src}. Ajuste os valores conforme o aluno.",
            "value": _build_predict_example_payload(feats),
        }
    }

    if "schema" not in app_json or not isinstance(app_json["schema"], dict):
        app_json["schema"] = {"type": "object", "additionalProperties": True}


app = FastAPI(
    title="Passos Mágicos - Defasagem API",
    version="1.0.0",
    description=(
        "**API desenvolvida para o Datathon 2026/MLOps - Pós Tech FIAP Machine Learning Engineering**\n\n"
        "**Aluno**: Rogerio Abramo A. Pretti | RA 363736 | Grupo 150 | 5MLET\n\n"
        "**Objetivo**:\n"
        "Estimar a probabilidade de um aluno ter defasagem de aprendizado (y=1) no ano t+1, "
        "para que a Associação Primeiro Passos possa tomar medidas preventivas.\n\n"
        "**Documentação**:\n"
        "A apresentação completa da API e regras de negócio estão em /docs."
    ),
    lifespan=lifespan,
    openapi_tags=[
        {"name": "Infra", "description": "Liveness/readiness e diagnósticos operacionais"},
        {"name": "Análise de Risco de Defasagem", "description": "Aplicação do modelo preditivo para análise de risco de defasagem de aprendizado"},
    ],
    docs_url=None,
    redoc_url=None,
)

app.include_router(feature_desc_router)

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
        tags=app.openapi_tags,
    )
    app.openapi_schema = schema

    # Injeção do example para o Swagger (não pode derrubar a API)
    try:
        _inject_predict_example_into_openapi(app)
    except Exception:
        pass

    return app.openapi_schema


app.openapi = custom_openapi


@app.get("/docs", include_in_schema=True, tags=["Infra"], summary="Documentação interativa (Swagger UI)")
def custom_docs():
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="Passos Mágicos - Docs",
        swagger_ui_parameters={
            "defaultModelsExpandDepth": -1,
            "docExpansion": "none",
        },
    )


@app.get("/health", tags=["Infra"], summary="Batimentos OK?", description="Endpoint de health check para validar se a API está rodando e se o modelo está em cache."  )
def health(request: Request) -> Dict[str, Any]:
    cached = _get_cached_model(request)
    return {
        "status": "ok",
        "uptime_s": _uptime_seconds(),
        "modelo_em_cache": cached is not None,
    }




@app.get("/smoke", tags=["Infra"], summary="Ligou sem Explodir?", description="Endpoint de smoke test para validar se o modelo carrega e roda predict_proba."  )
def smoke(
    request: Request,
    dry_run: bool = Query(default=False, description="Se true, roda predict_proba com dados sintéticos."),
) -> Dict[str, Any]:
    lm = _ensure_model_loaded(request)
    model = lm.model
    meta = lm.meta or {}

    if not hasattr(model, "predict_proba"):
        raise HTTPException(status_code=500, detail="Modelo não suporta predict_proba")

    expected, source = _get_expected_features(lm)

    dry: Dict[str, Any] = {"executado": False}
    if dry_run:
        try:
            X = pd.DataFrame([{c: 0 for c in expected}], columns=expected) if expected else pd.DataFrame([{}])
            _ = model.predict_proba(X)
            dry = {"executado": True, "ok": True, "estrategia": "zeros"}
        except Exception:
            try:
                X = pd.DataFrame([{c: "" for c in expected}], columns=expected) if expected else pd.DataFrame([{}])
                _ = model.predict_proba(X)
                dry = {"executado": True, "ok": True, "estrategia": "strings_vazias"}
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Falha no dry_run: {type(e).__name__}: {e}")

    threshold_value = float(meta.get("threshold", 0.5))

    return {
        "status": "ok",
        "modelo_carregado": True,
        "suporta_predict_proba": True,
        "model_path": str(meta.get("model_path", "unknown")),
        "threshold": threshold_value,
        "fonte_features_esperadas": source,
        "n_features_esperadas": len(expected),
        "features_esperadas": expected,
        "dry_run": dry,
    }



app.include_router(predict_router)