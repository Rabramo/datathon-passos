# src/models/compare.py
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd

from src.data.load_processed_pairs import load_train_test_from_processed
from src.models.model_factory import ModelConfig, build_model
from src.models.param_spaces import get_param_distributions
from src.models.pipeline import build_pipeline
from src.models.tuning import TuningConfig, tune_model, pick_threshold, evaluate_at_threshold


def _run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_compare(
    models: Optional[List[str]] = None,
    *,
    do_tuning: bool = True,
    threshold_objective: str = "f1",
    min_precision: Optional[float] = None,
    min_recall: Optional[float] = None,
    artifacts_dir: Path = Path("artifacts/models"),
    update_latest_per_model: bool = True,
    update_global_latest: bool = False,
) -> pd.DataFrame:
    """
    Treina e avalia uma lista de modelos no esquema temporal:
      - treino: pair_2022_2023
      - teste: pair_2023_2024

    Salva artefatos:
      - model_<name>_<run_id>.joblib
      - metrics_<name>_<run_id>.json
      - leaderboard.csv

    Ponteiros (opcional):
      - latest_<name>.json (recomendado)
      - latest.json (global, opcional)
    """
    models = models or ["dummy", "logreg", "tree", "rf", "xgb", "cat"]
    _ensure_dir(artifacts_dir)

    X_train, y_train, X_test, y_test = load_train_test_from_processed()

    rows: list[dict[str, Any]] = []
    global_latest_candidate: dict[str, Any] | None = None

    for name in models:
        run_id = _run_id()

        cfg = ModelConfig(name=name)
        clf = build_model(cfg)
        pipe = build_pipeline(X_train, clf)  # usa X_train para inferir colunas no preprocessor

        tuning_info: Dict[str, Any] | None = None
        if do_tuning and name not in ("dummy",):
            params = get_param_distributions(name)
            search = tune_model(pipe, params, X_train, y_train, TuningConfig())
            best_est = search.best_estimator_
            tuning_info = {
                "n_splits": search.cv.n_splits,
                "n_iter": getattr(search, "n_iter", None),
                "refit_metric": getattr(search, "refit", None),
                "best_score": float(search.best_score_),
                "best_params": search.best_params_,
            }
        else:
            best_est = pipe.fit(X_train, y_train)

        # threshold: selecionado no treino (in-sample)
        # (mais tarde pode evoluir para OOF, mas mantém consistente com o que já existe)
        y_train_proba = best_est.predict_proba(X_train)[:, 1]
        thr_best = pick_threshold(
            y_train,
            y_train_proba,
            thresholds=TuningConfig().threshold_grid,
            objective=threshold_objective,
            min_precision=min_precision,
            min_recall=min_recall,
        )
        thr = float(thr_best["threshold"])

        # métricas treino e teste temporal
        train_metrics = evaluate_at_threshold(y_train, y_train_proba, thr)
        y_test_proba = best_est.predict_proba(X_test)[:, 1]
        test_metrics = evaluate_at_threshold(y_test, y_test_proba, thr)

        # salvar modelo
        model_path = artifacts_dir / f"model_{name}_{run_id}.joblib"
        joblib.dump(best_est, model_path)

        # salvar métricas
        metrics = {
            "run_id": run_id,
            "model_name": name,
            "model_path": str(model_path),
            "threshold": thr,
            "threshold_policy": {
                "objective": threshold_objective,
                "min_precision": min_precision,
                "min_recall": min_recall,
            },
            "train_spec": {
                "id_col": "ra",
                "target_col": "y",
                "drop_cols": ["year_t", "year_t1", "ano"],
                "random_state": cfg.random_state,
            },
            "tuning": tuning_info,
            "train": train_metrics,
            "test": test_metrics,
        }
        metrics_path = artifacts_dir / f"metrics_{name}_{run_id}.json"
        _write_json(metrics_path, metrics)

        # atualizar ponteiro latest_<model>.json
        if update_latest_per_model:
            latest_model_path = artifacts_dir / f"latest_{name}.json"
            latest_payload = {
                "run_id": run_id,
                "model_name": name,
                "model_path": str(model_path),
                "metrics_path": str(metrics_path),
                "threshold": thr,
            }
            _write_json(latest_model_path, latest_payload)

        rows.append(
            {
                "run_id": run_id,
                "model": name,
                "threshold": thr,
                "test_roc_auc": test_metrics["roc_auc"],
                "test_f1": test_metrics["f1"],
                "test_precision": test_metrics["precision"],
                "test_recall": test_metrics["recall"],
                "metrics_path": str(metrics_path),
                "model_path": str(model_path),
            }
        )

    leaderboard = pd.DataFrame(rows).sort_values(["test_roc_auc", "test_f1"], ascending=False)
    leaderboard_path = artifacts_dir / "leaderboard.csv"
    leaderboard.to_csv(leaderboard_path, index=False)

    # opcional: atualizar latest.json global apontando para o melhor do leaderboard
    # (por padrão fica desligado para não quebrar a API sem intenção)
    if update_global_latest and len(leaderboard) > 0:
        top = leaderboard.iloc[0].to_dict()
        latest_global_path = artifacts_dir / "latest.json"
        _write_json(
            latest_global_path,
            {
                "run_id": top["run_id"],
                "model_name": top["model"],
                "model_path": top["model_path"],
                "metrics_path": top["metrics_path"],
                "threshold": top["threshold"],
                "selection_rule": "best_by_test_roc_auc_then_f1",
            },
        )

    return leaderboard


if __name__ == "__main__":
    lb = run_compare(
        models=["dummy", "logreg", "tree", "rf", "xgb", "cat"],
        do_tuning=True,
        threshold_objective="f1",
        min_precision=None,
        min_recall=None,
        update_latest_per_model=True,
        update_global_latest=False,
    )
    print(lb.to_string(index=False))