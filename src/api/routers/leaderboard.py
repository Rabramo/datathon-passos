#src/api/routers/leaderbord.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

router = APIRouter(
    prefix="/leaderboard",
    tags=["Leaderboard"],
)

LEADERBOARD_PATH = Path("artifacts/leaderboard/leaderboard.csv")


DEFAULT_SORT_COLS = (
    "test_roc_auc",
    "test_f1",
    "test_precision",
    "test_recall",
    "threshold",
    "model",
    "run_id",
    "metrics_path",
    "model_path",
)

SortOrder = Literal["asc", "desc"]
OutputFormat = Literal["json", "csv"]


def _carregar_leaderboard() -> pd.DataFrame:

    if not LEADERBOARD_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail=f"leaderboard.csv não encontrado em: {LEADERBOARD_PATH}",
        )

    try:
        df = pd.read_csv(LEADERBOARD_PATH)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Falha ao ler leaderboard.csv: {type(exc).__name__}: {exc}",
        ) from exc

    if df.empty:
        return df


    df.columns = [str(c).strip() for c in df.columns]
    return df


@router.get(
    "",
    summary="Retorna o leaderboard em JSON ou CSV",
    description=(
        "Lê o arquivo artifacts/leaderboard/leaderboard.csv e retorna os resultados. "
        "Permite limitar linhas, ordenar e filtrar por modelo."
    ),
)
def get_leaderboard(
    format: OutputFormat = Query(
        default="json",
        description="Formato de saída: json ou csv.",
    ),
    top_n: int = Query(
        default=50,
        ge=1,
        le=1000,
        description="Número máximo de linhas retornadas no modo JSON.",
    ),
    sort_by: Optional[str] = Query(
        default=None,
        description="Coluna para ordenação.",
    ),
    order: SortOrder = Query(
        default="desc",
        description="Ordem de ordenação: asc ou desc.",
    ),
    model_key: Optional[str] = Query(
        default=None,
        description="Filtra por model_key, se a coluna existir no arquivo.",
    ),
) -> Any:
    df = _carregar_leaderboard()

    if df.empty:
        if format == "csv":
            return FileResponse(
                str(LEADERBOARD_PATH),
                media_type="text/csv",
                filename="leaderboard.csv",
            )
        return {"status": "ok", "n": 0, "items": [], "columns": []}


    if model_key is not None and "model_key" in df.columns:
        df = df[df["model_key"].astype(str) == str(model_key)]


    if sort_by is not None:
        if sort_by not in df.columns:
            raise HTTPException(
                status_code=422,
                detail={
                    "msg": "sort_by inválido",
                    "sort_by": sort_by,
                    "available_columns": list(df.columns),
                },
            )
        ascending = order == "asc"
        df = df.sort_values(by=sort_by, ascending=ascending)
    else:

        for col in DEFAULT_SORT_COLS:
            if col in df.columns:
                df = df.sort_values(by=col, ascending=False)
                break

    df = df.head(top_n)

    if format == "csv":

        return FileResponse(
            str(LEADERBOARD_PATH),
            media_type="text/csv",
            filename="leaderboard.csv",
        )

    records = df.to_dict(orient="records")
    return {
        "status": "ok",
        "n": len(records),
        "items": records,
        "columns": list(df.columns),
    }