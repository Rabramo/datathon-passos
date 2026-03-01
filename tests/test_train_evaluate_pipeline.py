from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.pipelines.train import run_train


def _write_csv_gz(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, compression="gzip", sep=";")


def test_run_train_creates_artifacts(tmp_path: Path):
    processed_dir = tmp_path / "data" / "processed"
    artifacts_dir = tmp_path / "artifacts" / "models"

    # dataset processado mínimo (já no formato do pair)
    pair_2022_2023 = pd.DataFrame(
        {
            "ID": [1, 2, 3, 4],
            "year_t": [2022] * 4,
            "year_t1": [2023] * 4,
            "y": [0, 1, 0, 1],
            "feat": [10, 20, 30, 40],
        }
    )
    pair_2023_2024 = pd.DataFrame(
        {
            "ID": [1, 2, 3, 4],
            "year_t": [2023] * 4,
            "year_t1": [2024] * 4,
            "y": [0, 0, 1, 1],
            "feat": [11, 21, 31, 41],
        }
    )

    _write_csv_gz(processed_dir / "pair_2022_2023.csv.gz", pair_2022_2023)
    _write_csv_gz(processed_dir / "pair_2023_2024.csv.gz", pair_2023_2024)

    out = run_train(processed_dir, artifacts_dir, id_col="ID", random_state=42)

    assert "train" in out and "test" in out
    assert (artifacts_dir).exists()
    # deve ter criado um .joblib e um .json
    files = list(artifacts_dir.glob("*"))
    assert any(f.suffix == ".joblib" for f in files)
    assert any(f.suffix == ".json" for f in files)

    # valida conteúdo básico do json
    metrics_files = [f for f in files if f.suffix == ".json"]
    payload = json.loads(metrics_files[0].read_text(encoding="utf-8"))
    assert "threshold" in payload
    assert "test" in payload
    assert "confusion_matrix" in payload["test"]