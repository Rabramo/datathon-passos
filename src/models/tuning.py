# src/models/tuning.py
# src/models/tuning.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold


def _finite_space_size(param_distributions: Dict[str, Any]) -> Optional[int]:
    """
    Retorna o tamanho exato do espaço quando TODOS os valores são iteráveis finitos com len().
    Se algum valor não tiver len() (ex.: distribuições contínuas), retorna None.
    """
    if not param_distributions:
        return 0

    sizes: list[int] = []
    for v in param_distributions.values():
        try:
            sizes.append(len(v))  # type: ignore[arg-type]
        except Exception:
            return None

    total = 1
    for s in sizes:
        total *= s
    return total


@dataclass(frozen=True)
class TuningConfig:
    n_splits: int = 5
    n_iter: int = 25
    random_state: int = 42
    n_jobs: int = -1
    refit_metric: str = "roc_auc"  # ou "f1"
    threshold_grid: Tuple[float, ...] = tuple(np.round(np.arange(0.05, 0.96, 0.05), 2))


def tune_model(
    estimator: BaseEstimator,
    param_distributions: Dict[str, Any],
    X,
    y,
    cfg: TuningConfig,
):
    cv = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.random_state)

    scoring = {
        "roc_auc": "roc_auc",
        "f1": "f1",
        "precision": "precision",
        "recall": "recall",
    }

    # evita warning quando o espaço é menor que n_iter
    space = _finite_space_size(param_distributions)
    n_iter = cfg.n_iter
    if space is not None and space > 0:
        n_iter = min(cfg.n_iter, space)

    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        refit=cfg.refit_metric,
        cv=cv,
        n_jobs=cfg.n_jobs,
        random_state=cfg.random_state,
        verbose=0,
    )
    search.fit(X, y)
    return search


def pick_threshold(
    y_true,
    y_proba,
    thresholds,
    objective: str = "recall",
    min_precision: Optional[float] = 0.50,
    min_recall: Optional[float] = None,
    tie_tol: float = 1e-6,
):
    """
    Seleciona threshold maximizando 'objective' ('f1' ou 'recall'),
    com restrições opcionais (min_precision, min_recall).

    Tie-break:
    - se o score empatar (diferença <= tie_tol), escolhe o MAIOR threshold
      para reduzir falsos positivos mantendo desempenho semelhante.
    """
    best = {
        "threshold": 0.5,
        "score": -np.inf,
        "precision": None,
        "recall": None,
        "f1": None,
    }

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        if min_precision is not None and p < min_precision:
            continue
        if min_recall is not None and r < min_recall:
            continue

        score = f1 if objective == "f1" else r

        better = score > best["score"] + tie_tol
        tie = abs(score - best["score"]) <= tie_tol

        if better or (tie and float(t) > float(best["threshold"])):
            best = {
                "threshold": float(t),
                "score": float(score),
                "precision": float(p),
                "recall": float(r),
                "f1": float(f1),
            }

    return best


def evaluate_at_threshold(y_true, y_proba, threshold: float):
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }