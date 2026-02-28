from __future__ import annotations

from dataclasses import dataclass
import pandas as pd

from src.data.validate import assert_common_ids, assert_unique_id, detect_id_column


@dataclass(frozen=True)
class PairSpec:
    year_t: int
    year_t1: int


def _make_target_from_ian(df_t1: pd.DataFrame, ian_col: str) -> pd.Series:
    # y = 0 se IAN == 10 (Em fase), y = 1 caso contrário (Defasagem moderada ou severa)
    return (df_t1[ian_col] != 10).astype(int)


def build_temporal_pair(
    df_t: pd.DataFrame,
    df_t1: pd.DataFrame,
    year_t: int,
    year_t1: int,
    *,
    id_col: str | None = None,
    ian_col: str = "IAN",
) -> pd.DataFrame:
    if id_col is None:
        id_col = detect_id_column(df_t)

    # validações
    assert_unique_id(df_t, id_col, year_t)
    assert_unique_id(df_t1, id_col, year_t1)
    assert_common_ids(df_t, df_t1, id_col, year_t, year_t1)

    # manter só alunos presentes nos dois anos
    common = sorted(set(df_t[id_col]).intersection(set(df_t1[id_col])))
    left = df_t[df_t[id_col].isin(common)].copy()
    right = df_t1[df_t1[id_col].isin(common)].copy()

    # target vem SOMENTE do t+1
    if ian_col not in right.columns:
        raise ValueError(f"Column {ian_col} not found in year {year_t1}.")
    y = _make_target_from_ian(right, ian_col)

    # features: SOMENTE colunas de t
    X = left.copy()

    # dataset final: [id, year_t, year_t1, y, features...]
    out = pd.DataFrame({id_col: left[id_col].values})
    out["year_t"] = year_t
    out["year_t1"] = year_t1
    out["y"] = y.values

    # adiciona features (exclui year/target se existirem em t por acidente)
    # e garante que nenhuma coluna do df_t1 entre no dataset
    for c in X.columns:
        if c == id_col:
            continue
        out[c] = X[c].values

    return out


def build_all_pairs(years: dict[int, pd.DataFrame], *, id_col: str | None = None) -> dict[str, pd.DataFrame]:
    p1 = build_temporal_pair(years[2022], years[2023], 2022, 2023, id_col=id_col)
    p2 = build_temporal_pair(years[2023], years[2024], 2023, 2024, id_col=id_col)
    return {"2022_2023": p1, "2023_2024": p2}