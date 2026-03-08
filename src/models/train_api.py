# src/models/train_api.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Dict

import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

DATA_TRAIN = Path("data/processed/pair_2022_2023.parquet")
DATA_TEST = Path("data/processed/pair_2023_2024.parquet")

ART_DIR = Path("artifacts/models")
MET_DIR = Path("artifacts/metrics")
ART_DIR.mkdir(parents=True, exist_ok=True)
MET_DIR.mkdir(parents=True, exist_ok=True)

@dataclass
class TrainResult:
    run_id: str
    model_path: str
    metrics_path: str
    metrics: Dict[str, Any]

def _load_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset não encontrado: {path}")
    return pd.read_parquet(path)

def _make_model(internal_key: str, seed: int):
    if internal_key == "logreg":
        return LogisticRegression(max_iter=2000, random_state=seed)
    if internal_key == "tree":
        return DecisionTreeClassifier(random_state=seed)
    if internal_key == "rf":
        return RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)

    # Optional (only if installed). If not installed, raise clear error.
    if internal_key == "cat":
        try:
            from catboost import CatBoostClassifier
        except Exception as e:
            raise RuntimeError(f"CatBoost não disponível: {type(e).__name__}: {e}")
        return CatBoostClassifier(verbose=False, random_seed=seed)

    if internal_key == "xgb":
        try:
            from xgboost import XGBClassifier
        except Exception as e:
            raise RuntimeError(f"XGBoost não disponível: {type(e).__name__}: {e}")
        return XGBClassifier(
            random_state=seed,
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
        )

    if internal_key == "dummy":
        # handled separately (no sklearn fitting)
        return None

    raise ValueError(f"Modelo interno inválido: {internal_key}")

def train_temporal(
    internal_model_key: str,
    features: List[str],
    seed: int,
    threshold: float,
) -> TrainResult:
    df_tr = _load_df(DATA_TRAIN)
    df_te = _load_df(DATA_TEST)

    if "y" not in df_tr.columns or "y" not in df_te.columns:
        raise ValueError("Coluna alvo 'y' não encontrada nos datasets processados.")

    X_tr = df_tr.reindex(columns=features)
    y_tr = df_tr["y"].astype(int).values
    X_te = df_te.reindex(columns=features)
    y_te = df_te["y"].astype(int).values

    run_id = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")

    if internal_model_key == "dummy":
        # Baseline: probability = prevalence on train
        p = float(y_tr.mean()) if len(y_tr) else 0.0
        proba = pd.Series([p] * len(y_te)).values
        pred = (proba >= threshold).astype(int)

        auc = float(roc_auc_score(y_te, proba)) if len(set(y_te)) > 1 else 0.5
        f1 = float(f1_score(y_te, pred, zero_division=0))
        precision = float(precision_score(y_te, pred, zero_division=0))
        recall = float(recall_score(y_te, pred, zero_division=0))
        cm = confusion_matrix(y_te, pred).tolist()

        metrics = {
            "test_roc_auc": auc,
            "test_f1": f1,
            "test_precision": precision,
            "test_recall": recall,
            "test_sensitivity": recall,
            "confusion_matrix": cm,
            "threshold": float(threshold),
            "n_features": len(features),
        }

        # Save only metrics (no model artifact)
        metrics_path = MET_DIR / f"metrics_dummy_{run_id}.json"
        metrics_path.write_text(pd.Series(metrics).to_json(), encoding="utf-8")

        return TrainResult(
            run_id=run_id,
            model_path="",
            metrics_path=str(metrics_path),
            metrics=metrics,
        )

    model = _make_model(internal_model_key, seed)
    model.fit(X_tr, y_tr)

    if not hasattr(model, "predict_proba"):
        raise ValueError("Modelo treinado não suporta predict_proba.")

    proba = model.predict_proba(X_te)[:, 1]
    pred = (proba >= threshold).astype(int)

    auc = float(roc_auc_score(y_te, proba)) if len(set(y_te)) > 1 else 0.5
    f1 = float(f1_score(y_te, pred, zero_division=0))
    precision = float(precision_score(y_te, pred, zero_division=0))
    recall = float(recall_score(y_te, pred, zero_division=0))
    cm = confusion_matrix(y_te, pred).tolist()

    model_path = ART_DIR / f"model_{internal_model_key}_{run_id}.joblib"
    metrics_path = MET_DIR / f"metrics_{internal_model_key}_{run_id}.json"

    payload = {
        "run_id": run_id,
        "internal_model_key": internal_model_key,
        "threshold": float(threshold),
        "n_features": len(features),
        "features": features,
        "test_roc_auc": auc,
        "test_f1": f1,
        "test_precision": precision,
        "test_recall": recall,
        "test_sensitivity": recall,
        "confusion_matrix": cm,
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
    }

    joblib.dump(model, model_path)
    metrics_path.write_text(pd.Series(payload).to_json(), encoding="utf-8")

    return TrainResult(
        run_id=run_id,
        model_path=str(model_path),
        metrics_path=str(metrics_path),
        metrics=payload,
    )