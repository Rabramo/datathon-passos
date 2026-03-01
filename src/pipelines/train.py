# src/pipelines/train.py
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Tuple

import joblib
import numpy as np
import pandas as pd

from src.models.evaluate import ThresholdSpec, choose_threshold_max_f1, evaluate_dataset
from src.models.train import TrainSpec, predict_proba_positive, split_xy, train_model
from src.utils.io import ensure_dir


def _load_pair(processed_dir: Path, stem: str) -> pd.DataFrame:
    """
    Carrega pares do processed_dir.
    Preferência: parquet, fallback: csv.gz (sep=';')
    """
    p_parquet = processed_dir / f"{stem}.parquet"
    if p_parquet.exists():
        return pd.read_parquet(p_parquet, engine="pyarrow")

    p_csv = processed_dir / f"{stem}.csv.gz"
    if p_csv.exists():
        return pd.read_csv(p_csv, compression="gzip", sep=";", engine="python")

    raise FileNotFoundError(
        f"Missing processed pair file for '{stem}'. Expected {p_parquet.name} or {p_csv.name}."
    )


def _align_pair_to_train_schema(df_train: pd.DataFrame, df_other: pd.DataFrame, spec: TrainSpec) -> pd.DataFrame:
    """
    Alinha o schema de features de df_other ao schema de df_train.

    Motivo: pares 2022_2023 e 2023_2024 podem ter colunas diferentes (drift de schema),
    e o ColumnTransformer exige que as colunas do predict existam como no fit.

    Estratégia:
    - calcula X_train e X_other via split_xy
    - adiciona colunas faltantes em X_other com NaN
    - remove colunas extras em X_other
    - reordena X_other para a mesma ordem de X_train
    - reconstrói df_other preservando meta cols (id/year_t/year_t1/y)
    """
    X_train, _ = split_xy(df_train, spec)
    X_other, y_other = split_xy(df_other, spec)

    # adiciona colunas faltantes
    missing = [c for c in X_train.columns if c not in X_other.columns]
    for c in missing:
        X_other[c] = np.nan

    # remove colunas extras
    extra = [c for c in X_other.columns if c not in X_train.columns]
    if extra:
        X_other = X_other.drop(columns=extra)

    # reordena para bater com treino
    X_other = X_other[X_train.columns]

    # reconstrói df_other mantendo meta cols
    meta_cols = []
    for c in df_other.columns:
        if c == spec.target_col:
            continue
        if c in X_other.columns:
            continue
        meta_cols.append(c)

    out = df_other[meta_cols].copy()
    out[spec.target_col] = y_other.to_numpy()
    for c in X_other.columns:
        out[c] = X_other[c].to_numpy()

    return out


def _schema_diff(df_train: pd.DataFrame, df_test: pd.DataFrame, spec: TrainSpec) -> Tuple[list[str], list[str]]:
    Xtr, _ = split_xy(df_train, spec)
    Xte, _ = split_xy(df_test, spec)
    missing = sorted(set(Xtr.columns) - set(Xte.columns))
    extra = sorted(set(Xte.columns) - set(Xtr.columns))
    return missing, extra


def run_train(
    processed_dir: Path,
    artifacts_dir: Path,
    *,
    id_col: str = "ra",
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Treina baseline temporal:
    - Treino: pair_2022_2023
    - Teste: pair_2023_2024

    Salva:
    - artifacts/models/model_logreg_<run_id>.joblib (pipeline scikit-learn)
    - artifacts/models/metrics_<run_id>.json (métricas + threshold + spec)
    """
    df_train = _load_pair(processed_dir, "pair_2022_2023")
    df_test = _load_pair(processed_dir, "pair_2023_2024")

    # normaliza id_col (colunas do dataset estão normalizadas em minúsculo)
    id_col = id_col.strip().lower()

    spec = TrainSpec(id_col=id_col, random_state=random_state)

    # diagnóstico de schema (antes do alinhamento)
    missing, extra = _schema_diff(df_train, df_test, spec)
    if missing or extra:
        print(f"[train] Schema drift detectado. missing_in_test={len(missing)} extra_in_test={len(extra)}")
        if missing:
            print(f"[train] Exemplos missing_in_test: {missing[:10]}")
        if extra:
            print(f"[train] Exemplos extra_in_test: {extra[:10]}")

    # alinha df_test ao schema do treino para evitar erro do ColumnTransformer
    df_test_aligned = _align_pair_to_train_schema(df_train, df_test, spec)

    # treina modelo (pipeline inclui preprocess)
    model = train_model(df_train, spec)

    # threshold tuning no treino (max F1)
    p_train = predict_proba_positive(model, df_train, spec)
    _, y_train = split_xy(df_train, spec)
    thr = choose_threshold_max_f1(y_train.to_numpy(), p_train, ThresholdSpec())

    # métricas treino e teste
    train_metrics = evaluate_dataset(df_train, p_train, thr, spec)
    df_test_aligned = _align_pair_to_train_schema(df_train, df_test, spec)
    p_test = predict_proba_positive(model, df_test_aligned, spec)
    test_metrics = evaluate_dataset(df_test_aligned, p_test, thr, spec)

    ensure_dir(artifacts_dir)
    run_id = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

    # salvar pipeline diretamente (facilita /predict)
    model_path = artifacts_dir / f"model_logreg_{run_id}.joblib"
    joblib.dump(model, model_path)

    metrics_path = artifacts_dir / f"metrics_{run_id}.json"
    payload: dict[str, Any] = {
        "run_id": run_id,
        "model_path": str(model_path),
        "train_spec": asdict(spec),
        "threshold": float(thr),
        "train": train_metrics,
        "test": test_metrics,
        "schema_drift": {"missing_in_test": missing, "extra_in_test": extra},
    }
    metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train baseline temporally: 2022->2023 train, 2023->2024 test.")
    p.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--artifacts-dir", type=Path, default=Path("artifacts/models"))
    p.add_argument("--id-col", type=str, default="ra")
    p.add_argument("--random-state", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_train(
        processed_dir=args.processed_dir,
        artifacts_dir=args.artifacts_dir,
        id_col=args.id_col,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()