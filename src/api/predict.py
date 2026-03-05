# src/api/predict.py
from __future__ import annotations

import re
from typing import Any, Dict, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Request

from src.api.model_loader import LoadedModel, load_model
from src.api.schemas import PredictRequest, PredictResponse

router = APIRouter(tags=["Inferência"])

# Colunas comuns que podem aparecer no payload, mas nunca devem virar feature
DROP_NON_FEATURE_COLS = {"y", "year_t", "year_t1", "ano"}

# Bloqueios explícitos de colunas-alvo e indicativos de vazamento (t+1)
# Observação: ajuste conforme o seu dicionário de dados e convenções do pipeline.
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
    # r".*2024.*",      # opcional: habilite apenas se quiser bloquear chaves com ano explícito
]


def _extract_features(payload: PredictRequest) -> Dict[str, Any]:
    """
    Extrai as features do payload.

    Formatos aceitos:
      1) {"features": {...}}
      2) Payload "flat" (campos no topo), pois PredictRequest permite chaves extras.
    """
    if payload.features is not None:
        if not isinstance(payload.features, dict):
            raise HTTPException(status_code=422, detail="'features' deve ser um objeto/dict")
        return dict(payload.features)

    # Quando não vem "features", tratamos o payload inteiro (menos "features") como features
    data = payload.model_dump(exclude={"features"})
    return dict(data)


def _validate_no_future_or_target_features(feats: Dict[str, Any]) -> None:
    """
    Valida anti-vazamento no boundary da API.

    Regra:
      - Rejeitar colunas de alvo (ex.: IAN) e quaisquer chaves com padrão típico de t+1.
      - Isso previne uso acidental de variáveis do futuro/target na inferência.

    Retorna:
      - Levanta HTTPException(422) se encontrar chaves proibidas.
    """
    bad: list[str] = []
    for k in feats.keys():
        # Bloqueio por nome exato
        if k in FORBIDDEN_EXACT:
            bad.append(k)
            continue

        # Bloqueio por padrão
        for pat in FORBIDDEN_KEY_PATTERNS:
            if re.match(pat, k, flags=re.IGNORECASE):
                bad.append(k)
                break

    if bad:
        raise HTTPException(
            status_code=422,
            detail=f"Payload contém colunas proibidas (possível vazamento t+1/target): {sorted(set(bad))}",
        )


def _expected_columns(model: Any) -> Optional[list[str]]:
    """
    Tenta obter as colunas esperadas pelo pipeline treinado.

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
        # Não assume que é Pipeline; apenas segue sem colunas esperadas
        pass

    return None


def _get_loaded_model(request: Request) -> LoadedModel:
    """
    Obtém o modelo carregado para inferência.

    Prioridade:
      1) Usa `request.app.state.loaded` (LoadedModel) pré-carregado no lifespan/startup.
      2) Se não existir, carrega sob demanda via `load_model(return_meta=True)` e
         armazena em `request.app.state.loaded` para reaproveitar nas próximas chamadas.

    Motivação:
      - Evitar carregar o modelo múltiplas vezes.
      - Manter consistência com `src/api/app.py`, que cacheia em `app.state.loaded`.
    """
    loaded = getattr(request.app.state, "loaded", None)
    if isinstance(loaded, LoadedModel):
        return loaded

    # Fallback: carrega agora e cacheia no mesmo atributo usado pelo app.py
    lm = load_model(return_meta=True)
    request.app.state.loaded = lm
    return lm


def _get_threshold(meta: dict, default: float = 0.5) -> float:
    """
    Usa threshold do meta se existir; senão usa default.
    """
    thr = meta.get("threshold", None)
    try:
        return float(thr) if thr is not None else float(default)
    except Exception:
        return float(default)


@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Predizer risco de defasagem de aprendizagem (y=1) no ano t+1 usando features do ano t",
    description=(
        "Recebe as **features do aluno no ano t** e retorna:\n"
        "- **proba**: probabilidade estimada de **y=1** (risco de defasagem) no ano **t+1**\n"
        "- **prediction**: decisão binária (0/1) aplicando o **threshold**\n\n"
        "**Definição do alvo (y)**:\n"
        "- y=0 se **IAN == 10** (Sem Defasagem)\n"
        "- y=1 se **IAN != 10** (Com Defasagem)\n\n" 
        "IAN é o *Índice de Aprendizagem do Aluno* no ano t+1, derivado do histórico escolar.\n\n"
        "**Regra anti-vazamento**:\n"
        "A API deve receber apenas features do aluno do ano **t**. \n\n"
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
            "description": "Erro de validação do payload",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Payload contém colunas proibidas (possível vazamento t+1/target): ['IAN']"
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
def predict(payload: PredictRequest, request: Request) -> PredictResponse:
    """
    Endpoint de inferência.

    Regras:
      - Aceita apenas variáveis do ano t.
      - Bloqueia chaves com padrão de t+1 e colunas-alvo (ex.: IAN).
      - Se o modelo expõe um conjunto fixo de features esperadas, preenche faltantes com None
        e ignora chaves desconhecidas (fora do conjunto esperado).
    """
    lm = _get_loaded_model(request)
    model = lm.model
    meta = lm.meta or {}

    feats = _extract_features(payload)

    # Remove colunas que não devem entrar como feature se vierem por engano
    for c in list(feats.keys()):
        if c in DROP_NON_FEATURE_COLS:
            feats.pop(c, None)

    # Validação anti-vazamento antes de montar o DataFrame
    _validate_no_future_or_target_features(feats)

    expected = _expected_columns(model)

    if expected:
        # Garante que pelo menos alguma feature reconhecida foi enviada
        provided = set(feats.keys())
        expected_set = set(expected)
        if len(provided & expected_set) == 0:
            raise HTTPException(
                status_code=422,
                detail="Nenhuma feature reconhecida foi enviada (não bate com as colunas esperadas do modelo).",
            )

        # Monta linha com todas as colunas esperadas, preenchendo faltantes com None
        row = {c: feats.get(c, None) for c in expected}
        X = pd.DataFrame([row], columns=expected)
    else:
        # Caso não seja possível inferir colunas esperadas, usa tudo que veio como entrada
        X = pd.DataFrame([feats])

    if not hasattr(model, "predict_proba"):
        raise HTTPException(status_code=500, detail="Modelo carregado não suporta predict_proba")

    try:
        proba = float(model.predict_proba(X)[0][1])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Falha na predição: {type(e).__name__}: {e}")

    threshold = _get_threshold(meta, default=0.5)
    pred = int(proba >= threshold)

    model_path = str(meta.get("model_path", "unknown"))

    return PredictResponse(
        prediction=pred,
        proba=proba,
        threshold=threshold,
        model_path=model_path,
    )