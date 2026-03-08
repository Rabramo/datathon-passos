from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from src.api.schemas import TrainRequest, TrainResponse
from src.models.train_api import train_temporal

router = APIRouter(
    prefix="/train",
    tags=["Treinamento"],
)


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
        result = train_temporal(payload)

        if isinstance(result, TrainResponse):
            return result

        if isinstance(result, dict):
            return TrainResponse(**result)

        raise TypeError("train_temporal retornou um tipo de resposta não suportado.")

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