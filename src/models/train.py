# src/models/train.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


@dataclass(frozen=True)
class TrainSpec:
    # colunas normalizadas
    id_col: str = "ra"
    target_col: str = "y"
    # colunas operacionais do dataset temporal (não são features)
    drop_cols: tuple[str, ...] = ("year_t", "year_t1", "ano")
    random_state: int = 42
    max_iter: int = 2000
    # solver explícito para estabilidade. liblinear é bom para binário.
    solver: str = "liblinear"


def split_xy(df: pd.DataFrame, spec: TrainSpec) -> tuple[pd.DataFrame, pd.Series]:
    if spec.target_col not in df.columns:
        raise ValueError(f"Target column '{spec.target_col}' not found.")

    drop = [spec.target_col]
    for c in (spec.id_col, *spec.drop_cols):
        if c in df.columns:
            drop.append(c)

    X = df.drop(columns=drop, errors="ignore")
    y = df[spec.target_col].astype(int)
    return X, y


def _drop_all_nan_columns(X: pd.DataFrame) -> pd.DataFrame:
    """
    Remove colunas 100% NaN. Isso evita warnings do sklearn e comportamentos
    implícitos (p.ex. SimpleImputer ignorando features sem valores observados).
    """
    all_nan_cols = [c for c in X.columns if X[c].isna().all()]
    if all_nan_cols:
        X = X.drop(columns=all_nan_cols)
    return X


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Separa colunas em numéricas e categóricas (inclui dtype 'string').

    Ajuste importante:
    - Remove colunas numéricas que são 100% NaN no treino, para evitar warnings do SimpleImputer
      e garantir pipeline estável (ex.: 'ipp' no seu caso).
    """
    cat_cols: list[str] = []
    bool_cols: list[str] = []
    num_cols: list[str] = []

    for c in X.columns:
        dt = X[c].dtype
        dt_str = str(dt)

        if dt_str == "bool" or dt == bool:
            bool_cols.append(c)
        elif dt_str.startswith("category") or dt_str == "object" or dt_str == "string":
            cat_cols.append(c)
        else:
            num_cols.append(c)

    # trata bool como categórica (one-hot)
    cat_cols = cat_cols + bool_cols

    # remove numéricas 100% NaN no dataset de treino
    # (evita: "Skipping features without any observed values")
    num_cols = [c for c in num_cols if X[c].notna().any()]

    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    transformers: list[tuple[str, object, list[str]]] = []
    if num_cols:
        transformers.append(("num", numeric, num_cols))
    if cat_cols:
        transformers.append(("cat", categorical, cat_cols))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )


def train_model(df_train: pd.DataFrame, spec: TrainSpec) -> Pipeline:
    X_train, y_train = split_xy(df_train, spec)

    # robustez: remove features 100% nulas no treino
    X_train = _drop_all_nan_columns(X_train)

    pre = build_preprocessor(X_train)

    clf = LogisticRegression(
        max_iter=spec.max_iter,
        class_weight="balanced",
        random_state=spec.random_state,
        solver=spec.solver,
    )

    pipe = Pipeline([("preprocess", pre), ("model", clf)])
    pipe.fit(X_train, y_train)
    return pipe


def predict_proba_positive(model: Pipeline, df: pd.DataFrame, spec: TrainSpec) -> np.ndarray:
    X, _ = split_xy(df, spec)

    # Obs: o alinhamento de colunas do X para bater com treino deve ocorrer no pipeline (src/pipelines/train.py)
    if not hasattr(model, "predict_proba"):
        raise TypeError("O modelo/pipeline não expõe predict_proba().")

    proba = model.predict_proba(X)

    if proba.ndim != 2 or proba.shape[1] != 2:
        raise ValueError(f"predict_proba retornou shape inesperado: {proba.shape}")
    return proba[:, 1]