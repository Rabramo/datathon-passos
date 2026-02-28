from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.pipelines.build_dataset import run_build_dataset


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, sep=";")


def test_run_build_dataset_creates_outputs_and_stats(tmp_path: Path):
    raw_dir = tmp_path / "data" / "raw"
    processed_dir = tmp_path / "data" / "processed"

    # Dados mínimos coerentes com builder: ID em todos os anos e IAN no t+1
    df_2022 = pd.DataFrame({"ID": [1, 2, 3], "feat": [10, 20, 30]})
    df_2023 = pd.DataFrame({"ID": [1, 2, 3], "feat": [11, 21, 31], "IAN": [10, 9, 10]})
    df_2024 = pd.DataFrame({"ID": [1, 2, 3], "feat": [12, 22, 32], "IAN": [10, 10, 8]})

    _write_csv(raw_dir / "PEDE2022-Table 1.csv", df_2022)
    _write_csv(raw_dir / "PEDE2023-Table 1.csv", df_2023)
    _write_csv(raw_dir / "PEDE2024-Table 1.csv", df_2024)

    stats = run_build_dataset(raw_dir, processed_dir, id_col="ID")

    # arquivos gerados
    out_1 = processed_dir / "pair_2022_2023.csv.gz"
    out_2 = processed_dir / "pair_2023_2024.csv.gz"
    stats_path = processed_dir / "stats.json"

    assert out_1.exists()
    assert out_2.exists()
    assert stats_path.exists()

    # stats coerente
    loaded_stats = json.loads(stats_path.read_text(encoding="utf-8"))
    assert "pairs" in loaded_stats
    assert "2022_2023" in loaded_stats["pairs"]
    assert "2023_2024" in loaded_stats["pairs"]

    # anti-leakage: IAN não pode estar nas features salvas
    df_pair = pd.read_csv(out_1, compression="gzip", sep=";")
    assert "IAN" not in df_pair.columns
    assert "y" in df_pair.columns