from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from datetime import datetime, UTC
run_id = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.models.evaluate import ThresholdSpec, choose_threshold_max_f1, evaluate_dataset
from src.models.train import TrainSpec, predict_proba_positive, split_xy, train_model
from src.utils.io import ensure_dir


def _load_pair_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, compression="gzip", sep=";")


def run_train(
    processed_dir: Path,
    artifacts_dir: Path,
    *,
    id_col: str = "ID",
    random_state: int = 42,
) -> dict[str, Any]:
    train_path = processed_dir / "pair_2022_2023.csv.gz"
    test_path = processed_dir / "pair_2023_2024.csv.gz"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Missing processed pair files. Run build_dataset pipeline first.")

    df_train = _load_pair_csv(train_path)
    df_test = _load_pair_csv(test_path)

    spec = TrainSpec(id_col=id_col, random_state=random_state)

    model = train_model(df_train, spec)

    # threshold tuning no treino (max F1)
    p_train = predict_proba_positive(model, df_train, spec)
    _, y_train = split_xy(df_train, spec)
    thr = choose_threshold_max_f1(y_train.to_numpy(), p_train, ThresholdSpec())

    # métricas treino e teste
    train_metrics = evaluate_dataset(df_train, p_train, thr, spec)
    p_test = predict_proba_positive(model, df_test, spec)
    test_metrics = evaluate_dataset(df_test, p_test, thr, spec)

    ensure_dir(artifacts_dir)
    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    model_path = artifacts_dir / f"model_logreg_{run_id}.joblib"
    metrics_path = artifacts_dir / f"metrics_{run_id}.json"

    joblib.dump(
        {"model": model, "train_spec": asdict(spec), "threshold": thr},
        model_path,
    )

    payload = {
        "run_id": run_id,
        "model_path": str(model_path),
        "train_spec": asdict(spec),
        "threshold": thr,
        "train": train_metrics,
        "test": test_metrics,
    }
    metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train baseline temporally: 2022->2023 train, 2023->2024 test.")
    p.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--artifacts-dir", type=Path, default=Path("artifacts/models"))
    p.add_argument("--id-col", type=str, default="ID")
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
