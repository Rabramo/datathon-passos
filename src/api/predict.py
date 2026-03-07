# src/api/predict.py
from __future__ import annotations

import inspect
import re
from typing import Any, Dict, Optional

import pandas as pd
from fastapi import APIRouter, Body, HTTPException, Query, Request, Response

from src.api.model_loader import LoadedModel, load_model as load_loaded_model, resolve_model_key
from src.api.schemas import PredictRequest, PredictResponse

router = APIRouter(tags=["Análise de Risco de Defasagem"])

# Colunas comuns que podem aparecer no payload, mas nunca devem virar feature
DROP_NON_FEATURE_COLS = {"y", "year_t", "year_t1", "ano"}

# Bloqueios explícitos de colunas-alvo e indicativos de vazamento (t+1)
FORBIDDEN_EXACT = {
    "IAN",  # alvo/derivação direta do alvo
    "y",
    "target",
    "defasagem",
    "RISCO_DEFASAGEM",
    "year_t1",
    "year_t+1",
    "ano_t1",
    "ano_t+1",
}

# Padrões que sugerem variáveis do ano t+1 ou "futuro"
FORBIDDEN_KEY_PATTERNS = [
    r".*t\+?1.*",      # t1, t+1, etc.
    r".*year_?t1.*",   # year_t1
    r".*ano_?t1.*",    # ano_t1
]


def _extract_features(payload: PredictRequest) -> Dict[str, Any]:
    """
    Extrai as features do payload.

    Formatos aceitos:
      1) {"features": {...}}
      2) Payload flat (campos no topo), pois PredictRequest permite chaves extras.
    """
    if payload.features is not None:
        if not isinstance(payload.features, dict):
            raise HTTPException(status_code=422, detail="'features' deve ser um objeto/dict")
        return dict(payload.features)

    data = payload.model_dump(exclude={"features"})
    return dict(data)


def _validate_no_future_or_target_features(feats: Dict[str, Any]) -> None:
    """
    Valida anti-vazamento no boundary da API.

    Regras:
      - Rejeitar colunas de alvo (ex.: IAN) e quaisquer chaves com padrão típico de t+1.
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
                "hint": "Remova colunas do alvo (ex.: IAN) e quaisquer campos do ano t+1.",
            },
        )


def _expected_columns_from_model(model: Any) -> Optional[list[str]]:
    """
    Tenta obter as colunas esperadas pelo modelo/pipeline.

    Preferência:
      - model.feature_names_in_
      - model.named_steps["preprocess"].feature_names_in_ (quando for Pipeline)
    """
    cols = getattr(model, "feature_names_in_", None)
    if cols is not None:
        return list(cols)

    try:
        pre = model.named_steps.get("preprocess")  # type: ignore[attr-defined]
        cols = getattr(pre, "feature_names_in_", None)
        if cols is not None:
            return list(cols)
    except Exception:
        pass

    return None

def _resolve_model_key(model_key: Optional[str]) -> str:
    # normalize and default
    key = (model_key or "default").strip()
    return key

def _get_loaded_model(request: Request, model_key: Optional[str] = None) -> LoadedModel:
    """
    Carrega e cacheia modelos por model_key.

    Motivo:
    - Evitar carregar sempre o 'default' quando model_key muda.
    - Evitar NameError (load_model precisa ser importado deste módulo).
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
        # model_key inválida (não está no registry)
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        # arquivo do modelo não existe
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        # qualquer outra falha
        raise HTTPException(
            status_code=500,
            detail=f"Falha ao carregar modelo '{key}': {type(e).__name__}: {e}",
        )

    cache[key] = lm
    return lm

    # Create per-key cache dict
    cache = getattr(request.app.state, "models_by_key", None)
    if cache is None:
        cache = {}
        request.app.state.models_by_key = cache

    # Return cached model for that key
    if key in cache:
        return cache[key]

    # Load the requested model (this must support selecting by key)
    try:
        lm = load_loaded_model(model_key=key, return_meta=True)  # change signature if needed
    except TypeError:
        # If your load_model does not accept model_key, you need a selector layer.
        raise HTTPException(status_code=500, detail="load_model não suporta model_key. Implemente seleção por chave.")
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Modelo '{key}' indisponível: {type(e).__name__}: {e}")

    cache[key] = lm
    return lm

def _expected_columns(model: Any, meta: dict) -> Optional[list[str]]:
    """
    Define a lista de colunas esperadas.

    Preferência:
      - metadata com features brutas do ano t (raw_features/feature_columns/input_features)
      - fallback: introspecção do modelo (pode refletir pós-transformação)
    """
    for key in ("raw_features", "feature_columns", "input_features"):
        cols = meta.get(key)
        if isinstance(cols, list) and all(isinstance(c, str) for c in cols):
            return cols

    return _expected_columns_from_model(model)


def _load_model_with_optional_key(model_key: Optional[str]) -> LoadedModel:
    """
    Carrega modelo tentando respeitar model_key, sem assumir assinatura de load_model.

    Estratégia:
      - Inspeciona os parâmetros de load_model e só passa kwargs suportados.
    """
    sig = inspect.signature(load_model)
    kwargs: dict[str, Any] = {"return_meta": True}

    if model_key:
        if "model_key" in sig.parameters:
            kwargs["model_key"] = model_key
        elif "latest_key" in sig.parameters:
            kwargs["latest_key"] = model_key
        elif "latest_name" in sig.parameters:
            kwargs["latest_name"] = model_key
        # Se o loader só aceita caminho de JSON, não forçamos aqui sem mapeamento explícito.

    return load_model(**kwargs)  # type: ignore[arg-type]





def _get_threshold(meta: dict, default: float = 0.5) -> float:
    thr = meta.get("threshold", None)
    try:
        return float(thr) if thr is not None else float(default)
    except Exception:
        return float(default)


# ----------------------------
# Endpoints auxiliares
# ----------------------------


@router.get(
    "/model",
    summary="Mostra metadata do modelo carregado",
    description=(
        "Retorna informações do modelo atual (caminho, threshold, chaves do metadata).\n"
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
    model_key: Optional[str] = Query(
        default=None,
        description="Seleciona qual modelo consultar (mesma regra do /predict).",
    ),
) -> Dict[str, Any]:
    lm = _get_loaded_model(request, model_key=model_key)
    meta = lm.meta or {}

    return {
        "model_key": model_key or "default",
        "model_path": str(meta.get("model_path", "unknown")),
        "threshold": _get_threshold(meta, default=0.5),
        "meta_keys": sorted(list(meta.keys())),
    }


# ----------------------------
# Analise de risco de defasagem 
# ----------------------------

@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Estima risco de defasagem (y=1) no ano t+1 usando features do ano t",
    description=(
        "Recebe as **features do aluno no ano t** e retorna:\n"
        "- **proba**: probabilidade estimada de **y=1** (risco) no ano **t+1**\n"
        "- **prediction**: decisão binária (0/1) aplicando o **threshold**\n\n"
        "**Definição do alvo (y)**:\n"
        "- y=0 se **IAN == 10** (Sem Defasagem)\n"
        "- y=1 se **IAN != 10** (Com Defasagem)\n\n"
        "**Regra anti-vazamento**:\n"
        "A API deve receber apenas features do aluno do ano **t**.\n\n"
        "**Ajuda**:\n"
        "- Use `GET /features` para ver as chaves esperadas pelo modelo.\n"
        "- Use `GET /model` para inspecionar o modelo/threshold em uso.\n\n"
        "**Formatos de payload suportados**:\n"
        "1) Aninhado: `{ \"features\": { ... } }`\n"
        "2) Flat: `{ ... }` (chaves extras são aceitas e tratadas como features)\n"
    ),
    responses={
        200: {
            "description": "Predição realizada com sucesso",
            "content": {
                "application/json": {
                    "example": {
                        "prediction": 1,
                        "proba": 0.73,
                        "threshold": 0.5,
                        "model_path": "artifacts/models/model_tree_20260303_213447.joblib",
                    }
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
                                    "hint": "Remova colunas do alvo (ex.: IAN) e quaisquer campos do ano t+1.",
                                }
                            },
                        },
                        "no_recognized_features": {
                            "summary": "Nenhuma feature reconhecida",
                            "value": {
                                "detail": {
                                    "msg": "Nenhuma feature reconhecida foi enviada (não bate com as colunas esperadas do modelo).",
                                    "provided_keys": ["ra", "genero", "fase", "ida"],
                                    "expected_keys_sample": ["QTD_FALTAS", "IDADE", "INDE", "IPV", "IAA"],
                                    "hint": "Envie chaves exatamente como o modelo espera. Use GET /features para ver as chaves esperadas.",
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
            "description": "Erro interno de inferência",
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
    model_key: Optional[str] = Query(
        default=None,
        description=(
            "Seleciona qual modelo carregar conforme último treino:\n\n "
            "'latest_tree', 'latest_cat', 'latest_dummy', 'latest_logreg', 'latest_rf', 'latest_xgb'. \n\n"
            "Se omitido, usa o modelo Árvore de Decisão, que foi considerado o melhor modelo para as necessidades da Primeiros Passos. \n "
        ),
    ),
) -> PredictResponse:
    # Carrega (ou reutiliza) o modelo
    try:
        lm = _get_loaded_model(request, model_key=model_key)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model not available: {type(e).__name__}: {e}")

    model = lm.model
    meta = lm.meta or {}

    feats = _extract_features(payload)

    # Remove colunas que não devem entrar como feature se vierem por engano
    for c in list(feats.keys()):
        if c in DROP_NON_FEATURE_COLS:
            feats.pop(c, None)

    # Validação anti-vazamento antes de montar o DataFrame
    _validate_no_future_or_target_features(feats)

    expected = _expected_columns(model, meta)

    if expected:
        provided = set(feats.keys())
        expected_set = set(expected)

        if len(provided & expected_set) == 0:
            raise HTTPException(
                status_code=422,
                detail={
                    "msg": "Nenhuma feature reconhecida foi enviada (não bate com as colunas esperadas do modelo).",
                    "provided_keys": sorted(list(provided))[:80],
                    "expected_keys_sample": sorted(list(expected_set))[:80],
                    "hint": "Use GET /features para obter a lista de chaves esperadas e monte o payload com essas chaves.",
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

    # Threshold efetivo: query param > metadata > default
    threshold_value = float(threshold) if threshold is not None else _get_threshold(meta, default=0.5)
    pred = int(proba >= threshold_value)

    model_path = str(meta.get("model_path", "unknown"))

    # Headers úteis para debug sem mudar o schema de resposta
    response.headers["X-Model-Path"] = model_path
    response.headers["X-Model-Key"] = str(model_key or "default")
    response.headers["X-Threshold"] = str(threshold_value)

    return PredictResponse(
        prediction=pred,
        proba=proba,
        threshold=threshold_value,
        model_path=model_path,
    )