from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.models.train import TrainSpec, split_xy


@dataclass(frozen=True)
class ThresholdSpec:
    # busca thresholds de 0.05 a 0.95
    start: float = 0.05
    stop: float = 0.95
    step: float = 0.01


def choose_threshold_max_f1(y_true: np.ndarray, y_proba: np.ndarray, spec: ThresholdSpec) -> float:
    best_t = 0.5
    best_f1 = -1.0
    t = spec.start
    while t <= spec.stop + 1e-12:
        y_pred = (y_proba >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
        t += spec.step
    return best_t


def compute_metrics(y_true: np.ndarray, y_proba: np.ndarray, threshold: float) -> dict[str, Any]:
    y_pred = (y_proba >= threshold).astype(int)

    metrics: dict[str, Any] = {}
    metrics["threshold"] = float(threshold)
    metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))

    # roc_auc precisa de ambos rótulos presentes
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
    except ValueError:
        metrics["roc_auc"] = None

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    metrics["confusion_matrix"] = {
        "labels": [0, 1],
        "matrix": cm.tolist(),
    }
    return metrics


def evaluate_dataset(df: pd.DataFrame, y_proba: np.ndarray, threshold: float, spec: TrainSpec) -> dict[str, Any]:
    _, y = split_xy(df, spec)
    y_true = y.to_numpy()
    return compute_metrics(y_true, y_proba, threshold)