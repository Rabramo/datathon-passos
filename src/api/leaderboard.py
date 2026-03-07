from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Literal

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

router = APIRouter(tags=["Leaderboard"])

LEADERBOARD_PATH = Path("artifacts/leaderboard/leaderboard.csv")

# If you know the schema, list allowed columns here to avoid exposing arbitrary columns
DEFAULT_SORT_COLS = (
    "run_id",
    "model",
    "threshold",
    "test_roc_auc",
    "test_f1",
    "test_precision",
    "test_recall",
    "metrics_path",
    "model_path",
)

SortOrder = Literal["asc", "desc"]
OutputFormat = Literal["json", "csv"]


def _load_leaderboard() -> pd.DataFrame:
    if not LEADERBOARD_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail=f"leaderboard.csv não encontrado em: {LEADERBOARD_PATH}",
        )
    try:
        df = pd.read_csv(LEADERBOARD_PATH)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Falha ao ler leaderboard.csv: {type(e).__name__}: {e}")

    if df.empty:
        return df

    # Normalize column names (optional)
    df.columns = [str(c).strip() for c in df.columns]
    return df


@router.get(
    "/leaderboard",
    summary="Retorna o leaderboard (JSON ou CSV)",
    description=(
        "Lê artifacts/leaderboard/leaderboard.csv e retorna os resultados.\n"
        "Parâmetros permitem ordenar, limitar e filtrar por modelo."
    ),
)
def get_leaderboard(
    format: OutputFormat = Query(default="json", description="Formato de saída: json ou csv."),
    top_n: int = Query(default=50, ge=1, le=1000, description="Número máximo de linhas retornadas."),
    sort_by: Optional[str] = Query(default=None, description="Coluna para ordenação (ex.: f1, roc_auc)."),
    order: SortOrder = Query(default="desc", description="Ordem de ordenação: asc ou desc."),
    model_key: Optional[str] = Query(default=None, description="Filtra por model_key (se coluna existir)."),
) -> Any:
    df = _load_leaderboard()

    if df.empty:
        if format == "csv":
            return FileResponse(str(LEADERBOARD_PATH), media_type="text/csv", filename="leaderboard.csv")
        return {"status": "ok", "n": 0, "items": []}

    # Optional filter
    if model_key is not None and "model_key" in df.columns:
        df = df[df["model_key"].astype(str) == str(model_key)]

    # Sorting guard
    if sort_by is not None:
        if sort_by not in df.columns:
            raise HTTPException(
                status_code=422,
                detail={"msg": "sort_by inválido", "sort_by": sort_by, "available_columns": list(df.columns)},
            )
        ascending = order == "asc"
        df = df.sort_values(by=sort_by, ascending=ascending)
    else:
        # If user didn't pass sort_by, try a sensible default if present
        for c in DEFAULT_SORT_COLS:
            if c in df.columns:
                df = df.sort_values(by=c, ascending=False)
                break

    df = df.head(top_n)

    if format == "csv":
        # Return the whole CSV file (not only top_n). If you want only top_n as CSV, write temp file.
        return FileResponse(str(LEADERBOARD_PATH), media_type="text/csv", filename="leaderboard.csv")

    # JSON output
    records = df.to_dict(orient="records")
    return {"status": "ok", "n": len(records), "items": records, "columns": list(df.columns)}