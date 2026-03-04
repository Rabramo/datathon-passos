# src/models/model_factory.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

ModelName = Literal["dummy", "logreg", "tree", "rf", "xgb", "cat"]


@dataclass(frozen=True)
class ModelConfig:
    name: ModelName
    random_state: int = 42

    # Logistic Regression
    max_iter: int = 2000
    solver: str = "liblinear"

    # Decision Tree
    dt_max_depth: Optional[int] = None
    dt_min_samples_leaf: int = 1
    dt_class_weight: Optional[str] = "balanced"

    # Random Forest
    rf_n_estimators: int = 500
    rf_max_depth: Optional[int] = None
    rf_min_samples_leaf: int = 1
    rf_class_weight: Optional[str] = "balanced"
    rf_n_jobs: int = -1

    # XGBoost
    xgb_n_estimators: int = 500
    xgb_learning_rate: float = 0.05
    xgb_max_depth: int = 4
    xgb_subsample: float = 0.9
    xgb_colsample_bytree: float = 0.9
    xgb_reg_lambda: float = 1.0
    xgb_reg_alpha: float = 0.0

    # CatBoost
    cat_iterations: int = 800
    cat_learning_rate: float = 0.05
    cat_depth: int = 6
    cat_l2_leaf_reg: float = 3.0


def build_model(cfg: ModelConfig):
    if cfg.name == "dummy":
        return DummyClassifier(strategy="most_frequent")

    if cfg.name == "logreg":
        return LogisticRegression(
            random_state=cfg.random_state,
            max_iter=cfg.max_iter,
            solver=cfg.solver,
        )

    if cfg.name == "tree":
        return DecisionTreeClassifier(
            random_state=cfg.random_state,
            max_depth=cfg.dt_max_depth,
            min_samples_leaf=cfg.dt_min_samples_leaf,
            class_weight=cfg.dt_class_weight,
        )

    if cfg.name == "rf":
        return RandomForestClassifier(
            n_estimators=cfg.rf_n_estimators,
            random_state=cfg.random_state,
            max_depth=cfg.rf_max_depth,
            min_samples_leaf=cfg.rf_min_samples_leaf,
            class_weight=cfg.rf_class_weight,
            n_jobs=cfg.rf_n_jobs,
        )

    if cfg.name == "xgb":
        try:
            from xgboost import XGBClassifier
        except Exception as e:
            raise ImportError("xgboost não está instalado. Rode: pip install xgboost") from e

        return XGBClassifier(
            n_estimators=cfg.xgb_n_estimators,
            learning_rate=cfg.xgb_learning_rate,
            max_depth=cfg.xgb_max_depth,
            subsample=cfg.xgb_subsample,
            colsample_bytree=cfg.xgb_colsample_bytree,
            reg_lambda=cfg.xgb_reg_lambda,
            reg_alpha=cfg.xgb_reg_alpha,
            random_state=cfg.random_state,
            n_jobs=-1,
            eval_metric="logloss",
        )

    if cfg.name == "cat":
        try:
            from catboost import CatBoostClassifier
        except Exception as e:
            raise ImportError("catboost não está instalado. Rode: pip install catboost") from e

        return CatBoostClassifier(
            iterations=cfg.cat_iterations,
            learning_rate=cfg.cat_learning_rate,
            depth=cfg.cat_depth,
            l2_leaf_reg=cfg.cat_l2_leaf_reg,
            random_seed=cfg.random_state,
            loss_function="Logloss",
            verbose=False,
        )

    raise ValueError(f"Modelo não suportado: {cfg.name}")