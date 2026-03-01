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
    id_col: str = "ID"
    target_col: str = "y"
    drop_cols: tuple[str, ...] = ("year_t", "year_t1")
    random_state: int = 42
    max_iter: int = 2000


def split_xy(df: pd.DataFrame, spec: TrainSpec) -> tuple[pd.DataFrame, pd.Series]:
    if spec.target_col not in df.columns:
        raise ValueError(f"Target column {spec.target_col} not found.")

    drop = [spec.target_col]
    for c in (spec.id_col, *spec.drop_cols):
        if c in df.columns:
            drop.append(c)

    X = df.drop(columns=drop, errors="ignore")
    y = df[spec.target_col].astype(int)
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    cat_cols = [c for c in X.columns if X[c].dtype == "object" or str(X[c].dtype).startswith("category")]
    bool_cols = [c for c in X.columns if X[c].dtype == "bool"]
    num_cols = [c for c in X.columns if c not in cat_cols + bool_cols]
    cat_cols = cat_cols + bool_cols

    numeric = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical = Pipeline(
        [("imputer", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))]
    )

    return ColumnTransformer(
        transformers=[("num", numeric, num_cols), ("cat", categorical, cat_cols)],
        remainder="drop",
    )


def train_model(df_train: pd.DataFrame, spec: TrainSpec) -> Pipeline:
    X_train, y_train = split_xy(df_train, spec)
    pre = build_preprocessor(X_train)

    clf = LogisticRegression(
        max_iter=spec.max_iter,
        class_weight="balanced",
        random_state=spec.random_state,
        n_jobs=None,
    )

    pipe = Pipeline([("preprocess", pre), ("model", clf)])
    pipe.fit(X_train, y_train)
    return pipe


def predict_proba_positive(model: Pipeline, df: pd.DataFrame, spec: TrainSpec) -> np.ndarray:
    X, _ = split_xy(df, spec)
    return model.predict_proba(X)[:, 1]