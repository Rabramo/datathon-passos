#src.api.schemas.py

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ModelKey(str, Enum):
    dummy = "dummy"
    logreg = "logreg"
    tree = "tree"
    rf = "rf"
    xgb = "xgb"
    cat = "cat"


def _normalize_model_key_value(value: Any) -> Any:
    if value is None or isinstance(value, ModelKey):
        return value

    raw = str(value).strip()
    normalized = raw.lower().replace("_", " ").replace("-", " ")
    normalized = " ".join(normalized.split())

    aliases = {
        "dummy": "dummy",
        "logreg": "logreg",
        "logistic regression": "logreg",
        "tree": "tree",
        "decision tree": "tree",
        "rf": "rf",
        "random forest": "rf",
        "xgb": "xgb",
        "xgboost": "xgb",
        "cat": "cat",
        "catboost": "cat",
    }

    return aliases.get(normalized, value)


class FeatureName(str, Enum):
    fase = "fase"
    turma = "turma"
    ano_nasc = "ano_nasc"
    genero = "genero"
    ano_ingresso = "ano_ingresso"
    instituicao_de_ensino = "instituicao_de_ensino"
    no_av = "no_av"
    rec_av1 = "rec_av1"
    rec_av2 = "rec_av2"
    rec_av3 = "rec_av3"
    rec_av4 = "rec_av4"
    iaa = "iaa"
    ieg = "ieg"
    ips = "ips"
    rec_psicologia = "rec_psicologia"
    ida = "ida"
    matem = "matem"
    portug = "portug"
    ingles = "ingles"
    indicado = "indicado"
    atingiu_pv = "atingiu_pv"
    ipv = "ipv"
    ian = "ian"
    fase_ideal = "fase_ideal"
    defas = "defas"
    pedra = "pedra"
    idade = "idade"
    ipp = "ipp"
    tenure = "tenure"
    gap_fase = "gap_fase"
    pedra_ord = "pedra_ord"
    rec_av_count = "rec_av_count"


class FeatureSetName(str, Enum):
    all = "all"
    selected = "selected"
    custom = "custom"


class TrainRequest(BaseModel):
    model_key: ModelKey = Field(..., description="Modelo a treinar")
    feature_set: FeatureSetName = Field(default=FeatureSetName.all)
    variables: list[FeatureName] | None = Field(default=None, min_length=1)
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    random_seed: int = Field(default=42)

    @field_validator("model_key", mode="before")
    @classmethod
    def _normalize_model_key(cls, value: Any) -> Any:
        return _normalize_model_key_value(value)


class TrainResponse(BaseModel):
    status: str = "ok"
    run_id: str
    model_key: ModelKey
    internal_model_key: str
    n_features: int
    features_used: list[str]
    metrics: dict[str, Any]
    artifacts: dict[str, str]

    @field_validator("model_key", mode="before")
    @classmethod
    def _normalize_model_key(cls, value: Any) -> Any:
        return _normalize_model_key_value(value)


class FeatureSelectionRequest(BaseModel):
    name: str | None = Field(default="default")
    features: list[FeatureName] = Field(..., min_length=1)


class FeatureSelectionResponse(BaseModel):
    status: str = Field(default="ok")
    name: str
    n_selected: int
    selected_features: list[str]


class PredictRequest(BaseModel):
    model_key: ModelKey | None = None
    features: dict[str, Any] | None = Field(default=None)

    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={
            "examples": [
                {
                    "summary": "Formato aninhado",
                    "value": {
                        "model_key": "Logistic Regression",
                        "features": {
                            "ra": "123",
                            "genero": "F",
                            "fase": 5,
                            "ian": 10,
                            "ida": 6.8,
                        },
                    },
                },
                {
                    "summary": "Formato flat",
                    "value": {
                        "model_key": "Logistic Regression",
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

    @field_validator("model_key", mode="before")
    @classmethod
    def _normalize_model_key(cls, value: Any) -> Any:
        return _normalize_model_key_value(value)


class PredictResponse(BaseModel):
    prediction: int = Field(..., ge=0, le=1, examples=[1])
    proba: float = Field(..., ge=0.0, le=1.0, examples=[0.82])
    threshold: float = Field(..., ge=0.0, le=1.0, examples=[0.5])
    model_path: str = Field(
        ...,
        examples=["artifacts/models/model_logreg_20260303_213446.joblib"],
    )
