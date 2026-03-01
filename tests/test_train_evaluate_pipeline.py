# tests/test_train_evaluate_pipeline.py
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.pipelines.train import run_train


def _write_pair(processed_dir: Path, name: str, df: pd.DataFrame) -> None:
    """
    Escreve o par em parquet se possível (recomendado no pipeline atual),
    caso contrário escreve csv.gz com sep=';'.
    """
    processed_dir.mkdir(parents=True, exist_ok=True)

    # preferir parquet (se pyarrow estiver disponível no ambiente)
    parquet_path = processed_dir / f"{name}.parquet"
    try:
        df.to_parquet(parquet_path, index=False, engine="pyarrow")
        return
    except Exception:
        pass

    csv_path = processed_dir / f"{name}.csv.gz"
    df.to_csv(csv_path, index=False, compression="gzip", sep=";")


def test_run_train_creates_artifacts(tmp_path: Path):
    processed_dir = tmp_path / "data" / "processed"
    artifacts_dir = tmp_path / "artifacts" / "models"

    # dataset processado mínimo (formato pair)
    # colunas normalizadas: 'ra', 'year_t', 'year_t1', 'y', features...
    pair_2022_2023 = pd.DataFrame(
        {
            "ra": ["1", "2", "3", "4"],
            "year_t": [2022] * 4,
            "year_t1": [2023] * 4,
            "y": [0, 1, 0, 1],
            "feat": [10, 20, 30, 40],
        }
    )
    pair_2023_2024 = pd.DataFrame(
        {
            "ra": ["1", "2", "3", "4"],
            "year_t": [2023] * 4,
            "year_t1": [2024] * 4,
            "y": [0, 0, 1, 1],
            "feat": [11, 21, 31, 41],
        }
    )

    _write_pair(processed_dir, "pair_2022_2023", pair_2022_2023)
    _write_pair(processed_dir, "pair_2023_2024", pair_2023_2024)

    # treino deve aceitar id_col='ra'
    out = run_train(processed_dir, artifacts_dir, id_col="ra", random_state=42)

    assert "train" in out and "test" in out
    assert artifacts_dir.exists()

    files = list(artifacts_dir.glob("*"))
    assert any(f.suffix == ".joblib" for f in files)
    assert any(f.suffix == ".json" for f in files)

    metrics_files = [f for f in files if f.suffix == ".json"]
    payload = json.loads(metrics_files[0].read_text(encoding="utf-8"))
    assert "threshold" in payload
    assert "test" in payload
    assert "confusion_matrix" in payload["test"]