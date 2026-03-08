from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, status

from src.api.schemas import FeatureName, FeatureSetName, ModelKey, TrainRequest, TrainResponse
from src.models.train_api import train_temporal

router = APIRouter(
    prefix="/train",
    tags=["Treinamento"],
)

MODEL_KEY_TO_INTERNAL = {
    ModelKey.dummy: "dummy",
    ModelKey.logreg: "logreg",
    ModelKey.tree: "tree",
    ModelKey.cat: "cat",
    ModelKey.rf: "rf",
    ModelKey.xgb: "xgb",
}


def _resolve_features(payload: TrainRequest) -> list[str]:
    if payload.feature_set == FeatureSetName.all:
        return [feature.value for feature in FeatureName]

    if payload.variables:
        return [variable.value for variable in payload.variables]

    raise ValueError(
        "Para feature_set='custom' ou 'selected', envie ao menos uma variável em 'variables'."
    )


def _build_train_response(
    payload: TrainRequest,
    internal_model_key: str,
    features: list[str],
    result: Any,
) -> TrainResponse:
    if isinstance(result, TrainResponse):
        return result

    if isinstance(result, dict):
        return TrainResponse(**result)

    if (
        hasattr(result, "run_id")
        and hasattr(result, "model_path")
        and hasattr(result, "metrics_path")
    ):
        metrics = result.metrics if isinstance(result.metrics, dict) else {}
        return TrainResponse(
            status="ok",
            run_id=str(result.run_id),
            model_key=payload.model_key,
            internal_model_key=str(metrics.get("internal_model_key", internal_model_key)),
            n_features=int(metrics.get("n_features", len(features))),
            features_used=metrics.get("features", features),
            metrics=metrics,
            artifacts={
                "model_path": str(result.model_path),
                "metrics_path": str(result.metrics_path),
            },
        )

    raise TypeError("train_temporal retornou um tipo de resposta não suportado.")


def _call_train_temporal(
    payload: TrainRequest,
    internal_model_key: str,
    features: list[str],
) -> Any:
    try:
        return train_temporal(
            internal_model_key=internal_model_key,
            features=features,
            seed=payload.random_seed,
            threshold=payload.threshold,
        )
    except TypeError as exc:
        # Compatibilidade com versões legadas/mocks que aceitam payload único.
        if "unexpected keyword argument" in str(exc):
            return train_temporal(payload)
        raise


@router.post(
    "",
    response_model=TrainResponse,
    status_code=status.HTTP_200_OK,
    summary="Executa o treinamento temporal",
    description=(
        "Dispara o treinamento usando apenas variáveis do ano t "
        "para prever risco de defasagem no ano t+1."
    ),
)
def train_endpoint(payload: TrainRequest) -> TrainResponse:
    try:
        internal_model_key = MODEL_KEY_TO_INTERNAL[payload.model_key]
        features = _resolve_features(payload)

        result = _call_train_temporal(
            payload=payload,
            internal_model_key=internal_model_key,
            features=features,
        )

        return _build_train_response(
            payload=payload,
            internal_model_key=internal_model_key,
            features=features,
            result=result,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Falha no treinamento: {exc}",
        ) from exc
