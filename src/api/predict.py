# src/api/predict.py
from __future__ import annotations

import re
from typing import Any, Dict, List, Literal, Optional

import pandas as pd
from fastapi import APIRouter, Body, HTTPException, Query, Request, Response

from src.api.model_loader import LoadedModel, load_model as load_loaded_model, resolve_model_key
from src.api.schemas import PredictRequest, PredictResponse, FeatureSelectionRequest, FeatureSelectionResponse

router = APIRouter(tags=["Análise de Risco de Defasagem"])

DROP_NON_FEATURE_COLS = {"y", "year_t", "year_t1", "ano"}

# Bloqueios explícitos: alvo / t+1 / nomes tipicamente usados para target
# Nota: "IAN" (maiúsculo) é frequentemente usado como alvo/derivação do alvo em bases.
# Porém "ian" (minúsculo) aparece no seu modelo como feature (provavelmente IAN do ano t),
# então NÃO bloqueamos "ian". Bloqueamos "IAN" e padrões t+1.
FORBIDDEN_EXACT = {
    "IAN",  # alvo/derivação direta do alvo (muito comum em dados com target)
    "y",
    "target",
    "defasagem",
    "RISCO_DEFASAGEM",
    "year_t1",
    "year_t+1",
    "ano_t1",
    "ano_t+1",
}

# Padrões que sugerem variáveis do ano t+1 / futuro
FORBIDDEN_KEY_PATTERNS = [
    r".*t\+?1.*",      # t1, t+1, etc.
    r".*year_?t1.*",   # year_t1
    r".*ano_?t1.*",    # ano_t1
]

MODEL_KEY_MAP = {
    "Dummy": "dummy",
    "Logistic Regression": "logreg",
    "Decision Tree": "tree",
    "CatBoost": "cat",
    "Random Forest": "rf",
    "XGBoost": "xgb",
}

ModelKey = Literal[
    "Dummy",
    "Logistic Regression",
    "Decision Tree",
    "CatBoost",
    "Random Forest",
    "XGBoost",
]

FeatureName = Literal[
    "fase","turma","ano_nasc","genero","ano_ingresso","instituicao_de_ensino","no_av",
    "rec_av1","rec_av2","rec_av3","rec_av4","iaa","ieg","ips","rec_psicologia","ida",
    "matem","portug","ingles","indicado","atingiu_pv","ipv","ian","fase_ideal","defas",
    "pedra","idade","ipp","tenure","gap_fase","pedra_ord","rec_av_count"
]


def _extract_features(payload: PredictRequest) -> Dict[str, Any]:
    """
    Extrai features do payload.

    Formatos aceitos:
      1) {"features": {...}}
      2) payload "flat" (campos no topo), pois PredictRequest permite chaves extras.
    """
    if payload.features is not None:
        if not isinstance(payload.features, dict):
            raise HTTPException(status_code=422, detail="'features' deve ser um objeto (dict).")
        return dict(payload.features)

    data = payload.model_dump(exclude={"features"})
    return dict(data)


def _validate_no_future_or_target_features(feats: Dict[str, Any]) -> None:
    """
    Validação anti-vazamento no boundary da API.

    Regras:
      - Rejeitar chaves proibidas (target / t+1).
      - Rejeitar chaves com padrão típico de t+1.
    """
    bad: list[str] = []

    for k in feats.keys():
        if k in FORBIDDEN_EXACT:
            bad.append(k)
            continue

        for pat in FORBIDDEN_KEY_PATTERNS:
            if re.search(pat, k, flags=re.IGNORECASE):
                bad.append(k)
                break

    if bad:
        raise HTTPException(
            status_code=422,
            detail={
                "msg": "Payload contém colunas proibidas (possível vazamento t+1/target).",
                "forbidden_keys": sorted(set(bad)),
                "hint": (
                    "Remova colunas do alvo e quaisquer campos do ano t+1. "
                    "Para ver as chaves esperadas do modelo em uso, consulte GET /smoke."
                ),
            },
        )


def _expected_columns_from_model(model: Any) -> Optional[list[str]]:
    """
    Tenta obter as colunas esperadas diretamente do modelo.

    Preferência: model.feature_names_in_ (sklearn).
    """
    cols = getattr(model, "feature_names_in_", None)
    if cols is not None:
        return [str(c) for c in list(cols)]
    return None


def _expected_columns(model: Any, meta: dict) -> Optional[list[str]]:
    """
    Tenta obter as colunas esperadas.

    Preferência:
      - meta.raw_features / meta.feature_columns / meta.input_features (contrato ideal = features brutas)
    Fallback:
      - model.feature_names_in_ (pode refletir pós-transformação)
    """
    for key in ("raw_features", "feature_columns", "input_features"):
        cols = meta.get(key)
        if isinstance(cols, list) and cols and all(isinstance(c, str) for c in cols):
            return cols

    return _expected_columns_from_model(model)


def _get_threshold(meta: dict, default: float = 0.5) -> float:
    thr = meta.get("threshold", None)
    try:
        return float(thr) if thr is not None else float(default)
    except Exception:
        return float(default)


def _get_loaded_model(request: Request, model_key: Optional[str] = None) -> LoadedModel:
    """
    Carrega e cacheia modelos por chave em request.app.state.models_by_key.

    - model_key é normalizada via resolve_model_key (do seu loader).
    - retorna LoadedModel(model, meta).
    """
    key = resolve_model_key(model_key)

    cache = getattr(request.app.state, "models_by_key", None)
    if cache is None:
        cache = {}
        request.app.state.models_by_key = cache

    if key in cache:
        return cache[key]

    try:
        lm = load_loaded_model(model_key=key, return_meta=True)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Falha ao carregar modelo '{key}': {type(e).__name__}: {e}",
        )

    cache[key] = lm
    return lm


# ---------------------------------------------------------------------
# Endpoints auxiliares
# ---------------------------------------------------------------------

@router.get(
    "/model",
    summary="Mostra metadata do modelo carregado",
    description=(
        "Retorna informações do modelo informado.\n"
        "Útil para depuração e reprodutibilidade."
    ),
    responses={
        200: {
            "description": "Metadata retornado com sucesso",
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
    model_key: Optional[ModelKey] = Query(
        default=None,
        description="Seleciona qual modelo consultar (mesma regra do /predict).",
    ),
) -> Dict[str, Any]:
    internal_key = MODEL_KEY_MAP[model_key] if model_key is not None else None
    lm = _get_loaded_model(request, model_key=internal_key)
    meta = lm.meta or {}

    return {
        "model_key": resolve_model_key(model_key),
        "model_path": str(meta.get("model_path", "unknown")),
        "threshold": _get_threshold(meta, default=0.5),
        "meta_keys": sorted(list(meta.keys())),
    }


# ---------------------------------------------------------------------
# Predição
# ---------------------------------------------------------------------
@router.post(
    "/features/select",
    tags=["Análise de Risco de Defasagem"],
    response_model=FeatureSelectionResponse,
    summary="Seleciona features via query (multi-select no Swagger)",
    description=(
        "Salva em memória (app.state) um subconjunto de features permitidas para o payload. "
        "Não retreina o modelo; apenas filtra chaves recebidas e o restante segue como ausente."
    ),
)
def select_features(
    request: Request,
    name: str = Query(default="default"),
    features: List[FeatureName] = Query(default=[]),
):
    selected = list(dict.fromkeys(features))  # remove duplicatas preservando ordem

    if not selected:
        raise HTTPException(status_code=422, detail="Selecione ao menos 1 feature.")

    # valida anti-vazamento (opcional, mas recomendado)
    _validate_no_future_or_target_features({k: 0 for k in selected})

    if not hasattr(request.app.state, "feature_selections"):
        request.app.state.feature_selections = {}

    request.app.state.feature_selections[name] = selected

    # IMPORTANT: force OpenAPI regeneration so Swagger shows the new example
    request.app.openapi_schema = None

    return {"status": "ok", "name": name, "n_selected": len(selected), "selected_features": selected}

@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Estima risco de defasagem (y=1) no ano t+1 usando features do ano t",
    description=(
        "Recebe as **features do aluno no ano t** e retorna:\n"
        "- **proba**: probabilidade estimada de **y=1** (risco) no ano **t+1**\n"
        "- **prediction**: decisão binária (0/1) aplicando o **threshold**\n\n"
        "**Definição do alvo (y)**:\n"
        "- y=0 se **IAN(t+1) == 10** (Sem Defasagem)\n"
        "- y=1 se **IAN(t+1) != 10** (Com Defasagem)\n\n"
        "**Regra anti-vazamento (temporal)**:\n"
        "A API deve receber apenas variáveis do ano **t**.\n\n"
        "**Ajuda**:\n"
        "- Use `GET /smoke` para validar o carregamento do modelo e ver a lista completa de `features_esperadas`.\n"
        "- Use `GET /model` para inspecionar o `model_path`, `threshold` e `meta_keys`.\n"
        "- (Opcional) Use `GET /feature-descriptions` para ver descrições do contrato de entrada.\n\n"
        "**Formatos de payload suportados**:\n"
        "1) Aninhado: `{ \"features\": { ... } }`\n"
        "2) Flat: `{ ... }` (chaves extras são aceitas e tratadas como features)\n\n"
        "**Observação sobre chaves sensíveis**:\n"
        "Se você enviar uma variável equivalente ao alvo no **ano t+1** (futuro), isso é vazamento.\n"
        "O sistema tenta bloquear chaves típicas de t+1 por padrão."
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
        422: {
            "description": "Erro de validação do payload (anti-vazamento ou incompatibilidade de features)",
            "content": {
                "application/json": {
                    "examples": {
                        "forbidden_keys": {
                            "summary": "Colunas proibidas (vazamento/target)",
                            "value": {
                                "detail": {
                                    "msg": "Payload contém colunas proibidas (possível vazamento t+1/target).",
                                    "forbidden_keys": ["IAN"],
                                    "hint": "Remova colunas do alvo/t+1. Para ver as chaves esperadas do modelo em uso, consulte GET /smoke.",
                                }
                            },
                        },
                        "no_recognized_features": {
                            "summary": "Nenhuma feature reconhecida",
                            "value": {
                                "detail": {
                                    "msg": "Nenhuma feature reconhecida foi enviada (não bate com as colunas esperadas do modelo).",
                                    "provided_keys": ["ra", "genero", "fase", "ida"],
                                    "expected_keys_sample": ["fase", "turma", "ano_nasc"],
                                    "hint": "Envie chaves exatamente como o modelo espera. Consulte GET /smoke (features_esperadas).",
                                }
                            },
                        },
                    }
                }
            },
        },
        503: {
            "description": "Modelo indisponível (falha ao carregar artefato)",
            "content": {"application/json": {"example": {"detail": "Model not available"}}},
        },
        500: {
            "description": "Erro ao rodar o modelo (incompatibilidade de formato / falha interna)",
            "content": {"application/json": {"example": {"detail": "Falha na predição: ValueError: ..."}}},
        },
    },
)
def predict(
    request: Request,
    response: Response,
    payload: PredictRequest = Body(
        ...,
        description="Payload de inferência. Envie apenas features do ano t (anti-vazamento).",
    ),
    threshold: Optional[float] = Query(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Override do threshold para converter probabilidade em classe. "
            "Se omitido, usa o threshold salvo no metadata do modelo (ou 0.5)."
        ),
    ),
    model_key: Optional[ModelKey] = Query(
        default=None,
        description="Modelos disponíveis: dummy, logreg, tree (default), cat, rf, xgb.",
    ),
) -> PredictResponse:
    internal_key = MODEL_KEY_MAP[model_key] if model_key is not None else None
    lm = _get_loaded_model(request, model_key=internal_key)
    model = lm.model
    meta = lm.meta or {}

    feats = _extract_features(payload)
    selection_name = "default"
    selections = getattr(request.app.state, "feature_selections", {}) or {}
    selected = selections.get(selection_name)

    if selected:
        allowed = set(selected)
        feats = {k: v for k, v in feats.items() if k in allowed}

    for c in list(feats.keys()):
        if c in DROP_NON_FEATURE_COLS:
            feats.pop(c, None)

    # Anti-vazamento antes de montar o DataFrame
    _validate_no_future_or_target_features(feats)

    expected = _expected_columns(model, meta)

    # Monta X alinhando colunas (quando sabemos o esperado)
    if expected:
        provided = set(feats.keys())
        expected_set = set(expected)

        # Nenhuma interseção => provavelmente payload com chaves erradas
        if len(provided & expected_set) == 0:
            raise HTTPException(
                status_code=422,
                detail={
                    "msg": "Nenhuma feature reconhecida foi enviada (não bate com as colunas esperadas do modelo).",
                    "provided_keys": sorted(list(provided))[:80],
                    "expected_keys_sample": sorted(list(expected_set))[:80],
                    "hint": "Consulte GET /smoke e envie as chaves exatamente como em features_esperadas.",
                },
            )

        row = {c: feats.get(c, None) for c in expected}
        X = pd.DataFrame([row], columns=expected)
    else:
        X = pd.DataFrame([feats])

    if not hasattr(model, "predict_proba"):
        raise HTTPException(status_code=500, detail="Modelo carregado não suporta predict_proba")

    try:
        proba = float(model.predict_proba(X)[0][1])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Falha na predição: {type(e).__name__}: {e}")

    threshold_value = float(threshold) if threshold is not None else _get_threshold(meta, default=0.5)
    pred = int(proba >= threshold_value)

    model_path = str(meta.get("model_path", "unknown"))

    response.headers["X-Model-Path"] = model_path
    response.headers["X-Model-Key"] = resolve_model_key(model_key)
    response.headers["X-Threshold"] = str(threshold_value)

    return PredictResponse(
        prediction=pred,
        proba=proba,
        threshold=threshold_value,
        model_path=model_path,
    )

@router.post(
    "/predict/batch",
    tags=["Análise de Risco de Defasagem"],
    summary="Estima risco de defasagem em lote (batch)",
    description=(
        "Recebe uma lista de payloads (cada item pode ser flat `{...}` ou aninhado `{ \"features\": {...} }`).\n"
        "Retorna uma lista com `proba` e `prediction` para cada item.\n\n"
        "Observação: este endpoint reaproveita a mesma regra de threshold e o mesmo modelo do /predict."
    ),
)
def predict_batch(
    request: Request,
    items: List[Dict[str, Any]],
    model_key: Optional[ModelKey] = Query(
        default=None,
        description="Seleciona qual modelo usar.",
    ),
) -> Dict[str, Any]:
    if not items:
        raise HTTPException(status_code=422, detail="items não pode ser vazio.")

    internal_key = MODEL_KEY_MAP[model_key] if model_key is not None else None
    lm = _get_loaded_model(request, model_key=internal_key)
    model = lm.model
    meta = lm.meta or {}
    threshold_value = _get_threshold(meta, default=0.5)

    # Normaliza: aceita flat ou {"features": {...}}
    normalized: List[Dict[str, Any]] = []
    for i, it in enumerate(items):
        if not isinstance(it, dict):
            raise HTTPException(status_code=422, detail=f"Item {i} deve ser um objeto JSON (dict).")
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
        _validate_no_future_or_target_features(it)

    X = pd.DataFrame(normalized)

    # Reindex se soubermos o esperado
    expected = _expected_columns(model, meta)
    if expected:
        X = X.reindex(columns=expected)

    if not hasattr(model, "predict_proba"):
        raise HTTPException(status_code=500, detail="Modelo não suporta predict_proba")

    try:
        proba_1 = model.predict_proba(X)[:, 1].tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Falha no batch: {type(e).__name__}: {e}")

    preds = [1 if float(p) >= threshold_value else 0 for p in proba_1]

    results = [{"proba": float(p), "prediction": int(y), "threshold": float(threshold_value)} for p, y in zip(proba_1, preds)]

    return {
        "status": "ok",
        "model_key": resolve_model_key(model_key),
        "model_path": str(meta.get("model_path", "unknown")),
        "threshold": float(threshold_value),
        "n": len(results),
        "items": results,
    }
