# src/data/validate.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class ValidationReport:
    year_t: int
    year_t1: int
    id_col: str
    n_t: int
    n_t1: int
    n_common: int


def detect_id_column(df: pd.DataFrame, candidates: Iterable[str] = ("ra", "RA", "id", "student_id")) -> str:
    """
    Detecta a coluna de ID.
    Após o preprocess (normalize_columns), o esperado é 'ra'.
    """
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    # fallback: tenta case-insensitive
    lower_map = {c.lower(): c for c in df.columns}
    if "ra" in lower_map:
        return lower_map["ra"]
    raise ValueError("Não foi possível detectar a coluna de ID. Esperado 'ra'.")


def assert_unique_id(df: pd.DataFrame, id_col: str, year: int) -> None:
    if id_col not in df.columns:
        raise ValueError(f"A coluna de ID '{id_col}' não existe no dataframe do ano {year}.")

    if df[id_col].isna().any():
        n = int(df[id_col].isna().sum())
        raise ValueError(f"A coluna de ID '{id_col}' possui {n} valores nulos no ano {year}.")

    dup = df[id_col].duplicated().sum()
    if dup:
        raise ValueError(f"A coluna de ID '{id_col}' possui {int(dup)} duplicatas no ano {year}.")


def assert_common_ids(
    df_t: pd.DataFrame,
    df_t1: pd.DataFrame,
    id_col: str,
    year_t: int,
    year_t1: int,
    *,
    min_common: int = 1,
) -> ValidationReport:
    """
    Garante que existe interseção de IDs suficiente para construir pares.
    Não exige que os conjuntos sejam idênticos (entrada/saída de alunos é normal).
    """
    ids_t = set(df_t[id_col].astype(str))
    ids_t1 = set(df_t1[id_col].astype(str))
    common = ids_t.intersection(ids_t1)

    report = ValidationReport(
        year_t=year_t,
        year_t1=year_t1,
        id_col=id_col,
        n_t=len(ids_t),
        n_t1=len(ids_t1),
        n_common=len(common),
    )

    if len(common) < min_common:
        raise ValueError(
            f"Interseção insuficiente de IDs entre {year_t} e {year_t1}: "
            f"common={len(common)} (min_common={min_common})."
        )

    return report