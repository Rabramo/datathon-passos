# src/api/routers/predict.py
from __future__ import annotations

import re
from typing import Any, Dict, List, Literal, Optional

import pandas as pd
from fastapi import APIRouter, Body, HTTPException, Query, Request, Response

from src.api.model_loader import (
    LoadedModel,
    load_model as load_loaded_model,
    resolve_model_key,
)
from src.api.schemas import (
    FeatureName,
    FeatureSelectionResponse,
    PredictRequest,
    PredictResponse,
)

router = APIRouter(
    prefix="/predict",
    tags=["Análise de Risco de Defasagem"],
)

DROP_NON_FEATURE_COLS = {"y", "year_t", "year_t1", "ano"}


FORBIDDEN_EXACT = {
    "IAN",
    "y",
    "target",
    "defasagem",
    "RISCO_DEFASAGEM",
    "year_t1",
    "year_t+1",
    "ano_t1",
    "ano_t+1",
}


FORBIDDEN_KEY_PATTERNS = [
    r".*t\+?1.*",
    r".*year_?t1.*",
    r".*ano_?t1.*",
]


MODEL_KEY_MAP = {
    "Dummy": "dummy",
    "Regressão Logística": "logreg",
    "Árvore de Decisão": "tree",
    "CatBoost": "cat",
    "Random Forest": "rf",
    "XGBoost": "xgb",
}


SwaggerModelKey = Literal[
    "Dummy",
    "Regressão Logística",
    "Árvore de Decisão",
    "CatBoost",
    "Random Forest",
    "XGBoost",
]


def _extrair_features(payload: PredictRequest) -> Dict[str, Any]:
    """
    Extrai as features do payload.

    Formatos aceitos:
      1) {"features": {...}}
      2) payload flat, com as chaves no topo
    """
    if payload.features is not None:
        if not isinstance(payload.features, dict):
            raise HTTPException(
                status_code=422,
                detail="'features' deve ser um objeto JSON.",
            )
        return dict(payload.features)

    data = payload.model_dump(exclude={"features"})
    # model_key é aceito apenas via query param; ignora no body por compatibilidade.
    data.pop("model_key", None)
    return dict(data)


def _validar_sem_vazamento_ou_target(feats: Dict[str, Any]) -> None:
    """
    Validação anti-vazamento na borda da API.

    Regras:
    - rejeitar chaves do alvo
    - rejeitar chaves do ano t+1 / futuro
    """
    proibidas: list[str] = []

    for chave in feats.keys():
        if chave in FORBIDDEN_EXACT:
            proibidas.append(chave)
            continue

        for padrao in FORBIDDEN_KEY_PATTERNS:
            if re.search(padrao, chave, flags=re.IGNORECASE):
                proibidas.append(chave)
                break

    if proibidas:
        raise HTTPException(
            status_code=422,
            detail={
                "msg": "Payload contém colunas proibidas (possível vazamento t+1/target).",
                "forbidden_keys": sorted(set(proibidas)),
                "hint": (
                    "Remova colunas do alvo e quaisquer campos do ano t+1. "
                    "Consulte GET /predict/model e GET /smoke para inspecionar o contrato."
                ),
            },
        )


def _colunas_esperadas_do_modelo(model: Any) -> Optional[list[str]]:

    cols = getattr(model, "feature_names_in_", None)
    if cols is not None:
        return [str(c) for c in list(cols)]
    return None


def _colunas_esperadas(model: Any, meta: dict) -> Optional[list[str]]:

    for key in ("raw_features", "feature_columns", "input_features"):
        cols = meta.get(key)
        if isinstance(cols, list) and cols and all(isinstance(c, str) for c in cols):
            return cols

    return _colunas_esperadas_do_modelo(model)


def _obter_threshold(meta: dict, default: float = 0.5) -> float:

    thr = meta.get("threshold", None)
    try:
        return float(thr) if thr is not None else float(default)
    except Exception:
        return float(default)


def _normalizar_model_key(model_key: Optional[str]) -> Optional[str]:

    if model_key is None:
        return None
    return MODEL_KEY_MAP.get(model_key, model_key)


def _obter_modelo_carregado(request: Request, model_key: Optional[str] = None) -> LoadedModel:

    key = resolve_model_key(model_key)

    cache = getattr(request.app.state, "models_by_key", None)
    if cache is None:
        cache = {}
        request.app.state.models_by_key = cache

    if key in cache:
        return cache[key]

    try:
        lm = load_loaded_model(model_key=key, return_meta=True)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Falha ao carregar modelo '{key}': {type(exc).__name__}: {exc}",
        ) from exc

    cache[key] = lm
    return lm


@router.get(
    "/model",
    summary="Exibe os metadados do modelo carregado",
    description=(
        "Retorna informações do modelo selecionado. "
        "Útil para depuração, rastreabilidade e reprodutibilidade."
    ),
    responses={
        200: {
            "description": "Metadados retornados com sucesso",
            "content": {
                "application/json": {
                    "example": {
                        "model_key": "default",
                        "model_path": "artifacts/models/model_logreg_20260303_213446.joblib",
                        "threshold": 0.5,
                        "meta_keys": ["model_path", "threshold", "raw_features"],
                    }
                }
            },
        }
    },
)
def get_model_info(
    request: Request,
    model_key: Optional[SwaggerModelKey] = Query(
        default=None,
        description="Seleciona qual modelo consultar.",
    ),
) -> Dict[str, Any]:
    internal_key = _normalizar_model_key(model_key)
    lm = _obter_modelo_carregado(request, model_key=internal_key)
    meta = lm.meta or {}

    return {
        "model_key": resolve_model_key(internal_key),
        "model_path": str(meta.get("model_path", "unknown")),
        "threshold": _obter_threshold(meta, default=0.5),
        "meta_keys": sorted(list(meta.keys())),
    }


@router.post(
    "/features/select",
    response_model=FeatureSelectionResponse,
    summary="Seleciona features via query params",
    description=(
        "Salva em memória um subconjunto de features permitidas para o payload. "
        "Não retreina o modelo; apenas filtra as chaves aceitas na inferência."
    ),
)
def select_features(
    request: Request,
    name: str = Query(default="default"),
    features: List[FeatureName] = Query(default=[]),
) -> FeatureSelectionResponse:
    selected = list(dict.fromkeys(features))

    if not selected:
        raise HTTPException(status_code=422, detail="Selecione ao menos uma feature.")

    _validar_sem_vazamento_ou_target({k: 0 for k in selected})

    if not hasattr(request.app.state, "feature_selections"):
        request.app.state.feature_selections = {}

    request.app.state.feature_selections[name] = selected


    request.app.openapi_schema = None

    return FeatureSelectionResponse(
        status="ok",
        name=name,
        n_selected=len(selected),
        selected_features=selected,
    )


@router.post(
    "",
    response_model=PredictResponse,
    summary="Estima risco de defasagem no ano t+1 com features do ano t",
    description=(
        "Recebe as features do aluno no ano t e retorna:\n"
        "- proba: probabilidade estimada de y=1 no ano t+1\n"
        "- prediction: decisão binária após aplicação do threshold\n\n"
        "Definição do alvo:\n"
        "- y=0 se IAN(t+1) == 10\n"
        "- y=1 se IAN(t+1) != 10\n\n"
        "Regra temporal anti-vazamento:\n"
        "A API deve receber apenas variáveis do ano t."
    ),
    responses={
        200: {
            "description": "Predição realizada com sucesso",
            "content": {
                "application/json": {
                    "example": {"prediction": 1, "proba": 0.73, "threshold": 0.5}
                }
            },
        },
        422: {"description": "Erro de validação do payload"},
        500: {"description": "Erro interno na predição"},
    },
)
def predict(
    request: Request,
    response: Response,
    payload: PredictRequest = Body(
        ...,
        description="Payload de inferência. Envie apenas features do ano t.",
    ),
    threshold: Optional[float] = Query(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Override opcional do threshold. "
            "Se omitido, usa o threshold do metadata do modelo ou 0.5."
        ),
    ),
    model_key: Optional[SwaggerModelKey] = Query(
        default=None,
        description=(
            "Modelos disponíveis: Dummy, Regressão Logística, "
            "Árvore de Decisão, CatBoost, Random Forest, XGBoost."
        ),
    ),
) -> PredictResponse:
    internal_key = _normalizar_model_key(model_key)
    lm = _obter_modelo_carregado(request, model_key=internal_key)
    model = lm.model
    meta = lm.meta or {}

    feats = _extrair_features(payload)

    selection_name = "default"
    selections = getattr(request.app.state, "feature_selections", {}) or {}
    selected = selections.get(selection_name)

    if selected:
        allowed = set(selected)
        feats = {k: v for k, v in feats.items() if k in allowed}

    for c in list(feats.keys()):
        if c in DROP_NON_FEATURE_COLS:
            feats.pop(c, None)

    _validar_sem_vazamento_ou_target(feats)

    expected = _colunas_esperadas(model, meta)

    if expected:
        provided = set(feats.keys())
        expected_set = set(expected)

        if len(provided & expected_set) == 0:
            raise HTTPException(
                status_code=422,
                detail={
                    "msg": "Nenhuma feature reconhecida foi enviada.",
                    "provided_keys": sorted(list(provided))[:80],
                    "expected_keys_sample": sorted(list(expected_set))[:80],
                    "hint": (
                        "Envie as chaves exatamente como o modelo espera. "
                        "Consulte GET /smoke."
                    ),
                },
            )

        row = {c: feats.get(c, None) for c in expected}
        X = pd.DataFrame([row], columns=expected)
    else:
        X = pd.DataFrame([feats])

    if not hasattr(model, "predict_proba"):
        raise HTTPException(
            status_code=500,
            detail="O modelo carregado não suporta predict_proba().",
        )

    try:
        proba = float(model.predict_proba(X)[0][1])
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Falha na predição: {type(exc).__name__}: {exc}",
        ) from exc

    threshold_value = (
        float(threshold)
        if threshold is not None
        else _obter_threshold(meta, default=0.5)
    )
    pred = int(proba >= threshold_value)

    model_path = str(meta.get("model_path", "unknown"))

    response.headers["X-Model-Path"] = model_path
    response.headers["X-Model-Key"] = resolve_model_key(internal_key)
    response.headers["X-Threshold"] = str(threshold_value)

    return PredictResponse(
        prediction=pred,
        proba=proba,
        threshold=threshold_value,
        model_path=model_path,
    )


@router.post(
    "/batch",
    summary="Estima risco de defasagem em lote",
    description=(
        "Recebe uma lista de payloads. Cada item pode ser flat `{...}` "
        "ou aninhado `{ \"features\": {...} }`."
    ),
)
def predict_batch(
    request: Request,
    items: List[Dict[str, Any]] = Body(...),
    model_key: Optional[SwaggerModelKey] = Query(
        default=None,
        description="Seleciona qual modelo usar.",
    ),
) -> Dict[str, Any]:
    if not items:
        raise HTTPException(status_code=422, detail="items não pode ser vazio.")

    internal_key = _normalizar_model_key(model_key)
    lm = _obter_modelo_carregado(request, model_key=internal_key)
    model = lm.model
    meta = lm.meta or {}
    threshold_value = _obter_threshold(meta, default=0.5)

    normalized: List[Dict[str, Any]] = []
    for i, it in enumerate(items):
        if not isinstance(it, dict):
            raise HTTPException(
                status_code=422,
                detail=f"Item {i} deve ser um objeto JSON.",
            )
        if "features" in it and isinstance(it["features"], dict):
            normalized.append(dict(it["features"]))
        else:
            normalized.append(dict(it))

    selection_name = "default"
    selections = getattr(request.app.state, "feature_selections", {}) or {}
    selected = selections.get(selection_name)

    if selected:
        allowed = set(selected)
        normalized = [{k: v for k, v in it.items() if k in allowed} for it in normalized]

    for it in normalized:
        for c in list(it.keys()):
            if c in DROP_NON_FEATURE_COLS:
                it.pop(c, None)
        _validar_sem_vazamento_ou_target(it)

    X = pd.DataFrame(normalized)

    expected = _colunas_esperadas(model, meta)
    if expected:
        X = X.reindex(columns=expected)

    if not hasattr(model, "predict_proba"):
        raise HTTPException(
            status_code=500,
            detail="O modelo carregado não suporta predict_proba().",
        )

    try:
        proba_1 = model.predict_proba(X)[:, 1].tolist()
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Falha na predição em lote: {type(exc).__name__}: {exc}",
        ) from exc

    preds = [1 if float(p) >= threshold_value else 0 for p in proba_1]
    results = [
        {"proba": float(p), "prediction": int(y), "threshold": float(threshold_value)}
        for p, y in zip(proba_1, preds)
    ]

    return {
        "status": "ok",
        "model_key": resolve_model_key(internal_key),
        "model_path": str(meta.get("model_path", "unknown")),
        "threshold": float(threshold_value),
        "n": len(results),
        "items": results,
    }
