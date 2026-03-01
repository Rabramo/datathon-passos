# src/models/evaluate.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from src.models.train import TrainSpec, split_xy


@dataclass(frozen=True)
class ThresholdSpec:
    # busca thresholds de 0.05 a 0.95
    start: float = 0.05
    stop: float = 0.95
    step: float = 0.01


def _ensure_proba_1d(y_proba: np.ndarray) -> np.ndarray:
    """
    Normaliza y_proba para um vetor 1D com probabilidade da classe positiva (1).

    Aceita:
    - shape (n,) já como probabilidade da classe 1
    - shape (n, 2) como saída padrão de predict_proba
    """
    y_proba = np.asarray(y_proba)

    if y_proba.ndim == 1:
        return y_proba.astype(float)

    if y_proba.ndim == 2 and y_proba.shape[1] == 2:
        return y_proba[:, 1].astype(float)

    raise ValueError(f"y_proba com shape inválido: {y_proba.shape}. Esperado (n,) ou (n,2).")


def choose_threshold_max_f1(y_true: np.ndarray, y_proba: np.ndarray, spec: ThresholdSpec) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_proba_1d = _ensure_proba_1d(y_proba)

    best_t: float = 0.5
    best_f1: float = -1.0

    # garante que stop é incluído (com tolerância)
    t = spec.start
    while t <= spec.stop + 1e-12:
        y_pred = (y_proba_1d >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_t = float(t)
        t += spec.step

    return best_t


def compute_metrics(y_true: np.ndarray, y_proba: np.ndarray, threshold: float) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    y_proba_1d = _ensure_proba_1d(y_proba)

    if y_true.shape[0] != y_proba_1d.shape[0]:
        raise ValueError(
            f"y_true e y_proba têm tamanhos diferentes: {y_true.shape[0]} vs {y_proba_1d.shape[0]}"
        )

    y_pred = (y_proba_1d >= float(threshold)).astype(int)

    metrics: dict[str, Any] = {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    # roc_auc precisa de ambos rótulos presentes
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba_1d))
    except ValueError:
        metrics["roc_auc"] = None

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    metrics["confusion_matrix"] = {"labels": [0, 1], "matrix": cm.tolist()}

    # úteis para debug/monitoramento
    metrics["support"] = {"n": int(y_true.shape[0]), "pos": int(y_true.sum()), "neg": int((y_true == 0).sum())}

    return metrics


def evaluate_dataset(
    df: pd.DataFrame,
    y_proba: np.ndarray,
    threshold: float,
    spec: TrainSpec,
) -> dict[str, Any]:
    """
    Avalia um dataset (train ou test) já no formato 'pair':
    - usa split_xy(df, spec) para obter y_true
    - calcula métricas a partir de y_proba e threshold
    """
    _, y = split_xy(df, spec)
    y_true = y.to_numpy()
    return compute_metrics(y_true, y_proba, threshold)