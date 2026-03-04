from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from src.models.train import build_preprocessor


def build_pipeline(X_sample: pd.DataFrame, clf: BaseEstimator) -> Pipeline:
    """
    Reaproveita o mesmo preprocessamento do treino atual (build_preprocessor),
    adicionando o classificador no passo 'model'.
    """
    pre = build_preprocessor(X_sample)
    return Pipeline([("preprocess", pre), ("model", clf)])