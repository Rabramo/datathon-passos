from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.build_pairs import build_all_pairs
from src.data.load import DataPaths, load_years
from src.utils.io import ensure_dir


def compute_basic_stats(df: pd.DataFrame, y_col: str = "y") -> dict[str, Any]:
    out: dict[str, Any] = {}
    out["n_rows"] = int(df.shape[0])
    out["n_cols"] = int(df.shape[1])

    if y_col in df.columns:
        counts = df[y_col].value_counts(dropna=False).to_dict()
        out["y_counts"] = {str(k): int(v) for k, v in counts.items()}
        out["y_rate_1"] = float(df[y_col].mean()) if df.shape[0] > 0 else 0.0

    out["missing_rate_mean"] = float(df.isna().mean().mean()) if df.size > 0 else 0.0
    return out


def save_pair(df: pd.DataFrame, processed_dir: Path, name: str) -> Path:
    # Sem pyarrow por enquanto
    out_path = processed_dir / f"pair_{name}.csv.gz"
    df.to_csv(out_path, index=False, compression="gzip", sep=";")
    return out_path


def run_build_dataset(raw_dir: Path, processed_dir: Path, *, id_col: str | None = None) -> dict[str, Any]:
    ensure_dir(processed_dir)

    # Carrega 2022/2023/2024 do data/raw via src.data.load
    years = load_years(DataPaths(raw_dir=raw_dir))

    # Constrói pares temporais sem leakage
    pairs = build_all_pairs(years, id_col=id_col)

    stats: dict[str, Any] = {"outputs": {}, "pairs": {}}

    for name, df in pairs.items():
        # Assert extra anti-leakage (IAN é do t+1, não pode estar como feature)
        if "IAN" in df.columns:
            raise ValueError("Leakage detected: column 'IAN' present in processed dataset.")

        out_path = save_pair(df, processed_dir, name)
        stats["outputs"][name] = str(out_path)
        stats["pairs"][name] = compute_basic_stats(df)

    stats_path = processed_dir / "stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    return stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build temporal datasets t->t+1 without leakage.")
    p.add_argument("--raw-dir", type=Path, default=Path("data/raw"), help="Directory containing raw CSVs.")
    p.add_argument("--processed-dir", type=Path, default=Path("data/processed"), help="Output directory for processed data.")
    p.add_argument("--id-col", type=str, default=None, help="Optional explicit ID column name.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_build_dataset(args.raw_dir, args.processed_dir, id_col=args.id_col)


if __name__ == "__main__":
    main()