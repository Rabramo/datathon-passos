# src/api/schemas.py
from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional, Literal
from pydantic import BaseModel, Field, ConfigDict

# Shared enums (single source of truth for Swagger)
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

FeatureSetName = Literal["all", "selected", "custom"]

class TrainRequest(BaseModel):
    model_key: ModelKey = Field(..., description="Modelo a treinar (dropdown).")
    feature_set: FeatureSetName = Field(
        default="all",
        description="all=todas, selected=selecionadas via /features/select, custom=usa variables.",
    )
    variables: Optional[List[FeatureName]] = Field(
        default=None,
        description="Se informado, sobrescreve feature_set e usa exatamente essas variáveis (multi-select).",
    )
    threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Threshold para métricas de classe.")
    random_seed: int = Field(default=42, description="Seed para reprodutibilidade.")

class TrainResponse(BaseModel):
    status: str = "ok"
    run_id: str
    model_key: ModelKey
    internal_model_key: str
    n_features: int
    features_used: List[str]
    metrics: Dict[str, Any]
    artifacts: Dict[str, str]

class FeatureSelectionRequest(BaseModel):
    name: Optional[str] = Field(
        default="default",
        description="Nome do conjunto de features (ex.: default, ablation_1).",
    )
    features: List[str] = Field(
        ...,
        description="Lista de nomes de features que o usuário deseja usar/enviar.",
        min_length=1,
    )

class FeatureSelectionResponse(BaseModel):
    status: str = Field(default="ok")
    name: str
    n_selected: int
    selected_features: List[str]
    
class PredictRequest(BaseModel):
    """
    Payload de inferência.

    Formatos aceitos:
    1) Aninhado:
       {
         "features": { ... }
       }

    2) Flat (campos no topo):
       {
         "ra": "123",
         "genero": "F",
         "fase": 5,
         ...
       }

    Regra anti-vazamento:
    - Envie apenas variáveis do ano t (não inclua colunas do ano t+1).
    """

    features: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Dicionário de features do aluno no ano t. "
            "Se omitido, o payload pode ser flat (chaves no nível raiz)."
        ),
    )

    # Aceita chaves extras no payload (para suportar o formato flat)
    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={
            "examples": [
                {
                    "summary": "Formato aninhado (recomendado)",
                    "description": "As features vão dentro de `features`.",
                    "value": {
                        "features": {
                            "ra": "123",
                            "genero": "F",
                            "fase": 5,
                            "ian": 10,
                            "ida": 6.8,
                        }
                    },
                },
                {
                    "summary": "Formato flat (aceito)",
                    "description": "As features vão no topo do JSON (chaves extras são tratadas como features).",
                    "value": {
                        "ra": "123",
                        "genero": "F",
                        "fase": 5,
                        "ian": 10,
                        "ida": 6.8,
                    },
                },
            ]
        },
    )


class PredictResponse(BaseModel):
    prediction: int = Field(
        ...,
        ge=0,
        le=1,
        description="Classe prevista: 1=risco de defasagem, 0=em fase.",
        examples=[1],
    )
    proba: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probabilidade estimada de y=1 (risco).",
        examples=[0.82],
    )
    threshold: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Threshold aplicado para gerar `prediction` a partir de `proba`.",
        examples=[0.5],
    )
    model_path: str = Field(
        ...,
        description="Caminho do artefato do modelo carregado (dentro do container/ambiente).",
        examples=["artifacts/models/model_logreg_20260303_213446.joblib"],
    )