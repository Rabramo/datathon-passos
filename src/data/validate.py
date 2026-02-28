from __future__ import annotations

from typing import Iterable
import pandas as pd


def detect_id_column(df: pd.DataFrame, candidates: Iterable[str] = ("ID", "id", "Id", "RA", "ra")) -> str:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    # fallback heurístico: colunas que parecem identificador
    for c in df.columns:
        if "id" in c.lower() or "ra" == c.lower():
            return c
    raise ValueError(f"ID column not found. Available columns: {list(df.columns)[:30]}...")


def assert_unique_id(df: pd.DataFrame, id_col: str, year: int) -> None:
    if df[id_col].isna().any():
        raise ValueError(f"Found NA in id column {id_col} for year {year}.")
    dup = df[id_col].duplicated().sum()
    if dup > 0:
        raise ValueError(f"Found {dup} duplicated IDs in {id_col} for year {year}.")


def assert_common_ids(df_t: pd.DataFrame, df_t1: pd.DataFrame, id_col: str, year_t: int, year_t1: int) -> None:
    ids_t = set(df_t[id_col].unique())
    ids_t1 = set(df_t1[id_col].unique())
    inter = ids_t.intersection(ids_t1)
    if len(inter) == 0:
        raise ValueError(f"No common IDs between {year_t} and {year_t1} using {id_col}.")