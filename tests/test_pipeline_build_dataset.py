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
    interim_dir = tmp_path / "data" / "interim"
    processed_dir = tmp_path / "data" / "processed"

    # Use RA/IAN (o preprocess vai normalizar para ra/ian)
    df_2022 = pd.DataFrame({"ra": ["1", "2", "3"], "feat": [10, 20, 30]})
    df_2023 = pd.DataFrame({"ra": ["1", "2", "3"], "feat": [11, 21, 31], "ian": [10, 9, 10]})
    df_2024 = pd.DataFrame({"ra": ["1", "2", "3"], "feat": [12, 22, 32], "ian": [10, 10, 8]})

    _write_csv(raw_dir / "PEDE2022-Table 1.csv", df_2022)
    _write_csv(raw_dir / "PEDE2023-Table 1.csv", df_2023)
    _write_csv(raw_dir / "PEDE2024-Table 1.csv", df_2024)

    # Se sua assinatura ainda não tem interim_dir, remova o interim_dir da chamada.
    stats = run_build_dataset(raw_dir, interim_dir, processed_dir, id_col="ra")

    out_1 = processed_dir / "pair_2022_2023.parquet"
    out_2 = processed_dir / "pair_2023_2024.parquet"
    stats_path = processed_dir / "stats.json"

    assert out_1.exists()
    assert out_2.exists()
    assert stats_path.exists()

    loaded_stats = json.loads(stats_path.read_text(encoding="utf-8"))
    assert "pairs" in loaded_stats
    assert "2022_2023" in loaded_stats["pairs"]
    assert "2023_2024" in loaded_stats["pairs"]

    df_pair = pd.read_parquet(out_1)

    # sempre deve existir y
    assert "y" in df_pair.columns

    # anti-leakage: o par 2022_2023 NÃO pode carregar ian do t+1 como feature.
    # Como df_2022 não tem IAN, não deve existir 'ian' no par (só o y vem do 2023).
    assert "ian" not in df_pair.columns

    # e nunca deve existir a coluna operacional 'ano' no par
    assert "ano" not in df_pair.columns