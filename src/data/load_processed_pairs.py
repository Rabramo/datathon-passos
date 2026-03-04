from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd


@dataclass(frozen=True)
class PairPaths:
    train_pair_path: Path = Path("data/processed/pair_2022_2023.parquet")
    test_pair_path: Path = Path("data/processed/pair_2023_2024.parquet")


def _split_X_y(
    df: pd.DataFrame,
    *,
    id_col: str = "ra",
    target_col: str = "y",
    drop_cols: Iterable[str] = ("year_t", "year_t1", "ano"),
) -> tuple[pd.DataFrame, pd.Series]:
    cols_to_drop = [id_col, target_col, *list(drop_cols)]
    X = df.drop(columns=cols_to_drop, errors="ignore")
    y = df[target_col].astype(int)
    return X, y


def load_train_test_from_processed(
    paths: PairPaths = PairPaths(),
    *,
    id_col: str = "ra",
    target_col: str = "y",
    drop_cols: Iterable[str] = ("year_t", "year_t1", "ano"),
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    if not paths.train_pair_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {paths.train_pair_path}")
    if not paths.test_pair_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {paths.test_pair_path}")

    train_df = pd.read_parquet(paths.train_pair_path)
    test_df = pd.read_parquet(paths.test_pair_path)

    X_train, y_train = _split_X_y(train_df, id_col=id_col, target_col=target_col, drop_cols=drop_cols)
    X_test, y_test = _split_X_y(test_df, id_col=id_col, target_col=target_col, drop_cols=drop_cols)
    return X_train, y_train, X_test, y_test