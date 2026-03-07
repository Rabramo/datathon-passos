# src/features/build_pairs.py
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.data.validate import assert_common_ids, assert_unique_id, detect_id_column


@dataclass(frozen=True)
class PairSpec:
    year_t: int
    year_t1: int


def _make_target_from_ian(df_t1: pd.DataFrame, ian_col: str) -> pd.Series:
    """
    y = 0 se IAN == 10 (Em fase)
    y = 1 caso contrário (Defasagem moderada ou severa)
    """
    return (df_t1[ian_col] != 10).astype(int)


def build_temporal_pair(
    df_t: pd.DataFrame,
    df_t1: pd.DataFrame,
    year_t: int,
    year_t1: int,
    *,
    id_col: str | None = None,
    ian_col: str = "ian",
    drop_feature_cols: tuple[str, ...] = ("ano",),
) -> pd.DataFrame:
    """
    Constrói dataset temporal t -> t+1:
    - Features: SOMENTE colunas do ano t (df_t)
    - Target y: derivado SOMENTE de IAN no ano t+1 (df_t1)
    """
    if id_col is None:
        id_col = detect_id_column(df_t)

    if id_col not in df_t1.columns:
        raise ValueError(f"ID column '{id_col}' not found in year {year_t1} dataframe.")

    # validações
    assert_unique_id(df_t, id_col, year_t)
    assert_unique_id(df_t1, id_col, year_t1)

    # se sua versão de assert_common_ids já tem min_common, você pode passar:
    # report = assert_common_ids(df_t, df_t1, id_col, year_t, year_t1, min_common=1)
    # senão, mantenha sem kwargs:
    assert_common_ids(df_t, df_t1, id_col, year_t, year_t1)

    # inner join por IDs comuns
    common_ids = set(df_t[id_col]).intersection(set(df_t1[id_col]))
    left = df_t[df_t[id_col].isin(common_ids)].copy()

    # alinhar df_t1 na mesma ordem do df_t
    right = df_t1[df_t1[id_col].isin(common_ids)].copy().set_index(id_col)
    right = right.loc[left[id_col].values.tolist()]

    # target vem SOMENTE do t+1
    if ian_col not in right.columns:
        raise ValueError(f"Column '{ian_col}' not found in year {year_t1} dataframe.")
    y = _make_target_from_ian(right, ian_col)

    # features: SOMENTE colunas de t
    X = left.copy()

    # remove colunas operacionais redundantes (ex.: 'ano' do interim)
    cols_to_drop = [c for c in drop_feature_cols if c in X.columns]
    if cols_to_drop:
        X = X.drop(columns=cols_to_drop)

    # dataset final
    out = pd.DataFrame({id_col: left[id_col].values})
    out["year_t"] = int(year_t)
    out["year_t1"] = int(year_t1)
    out["y"] = y.to_numpy()

    for c in X.columns:
        if c == id_col:
            continue
        out[c] = X[c].to_numpy()

    return out


def build_all_pairs(
    years: dict[int, pd.DataFrame],
    *,
    id_col: str | None = None,
    ian_col: str = "ian",
) -> dict[str, pd.DataFrame]:
    p1 = build_temporal_pair(years[2022], years[2023], 2022, 2023, id_col=id_col, ian_col=ian_col)
    p2 = build_temporal_pair(years[2023], years[2024], 2023, 2024, id_col=id_col, ian_col=ian_col)
    return {"2022_2023": p1, "2023_2024": p2}