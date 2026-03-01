from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.preprocess import preprocess_year_file
from src.features.build_pairs import build_all_pairs
from src.utils.io import ensure_dir


def compute_basic_stats(df: pd.DataFrame, y_col: str = "y") -> dict[str, Any]:
    out: dict[str, Any] = {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "missing_rate_mean": float(df.isna().mean().mean()) if df.size > 0 else 0.0,
    }

    if y_col in df.columns and df.shape[0] > 0:
        counts = df[y_col].value_counts(dropna=False).to_dict()
        out["y_counts"] = {str(k): int(v) for k, v in counts.items()}
        out["y_rate_1"] = float(df[y_col].mean())

    return out


def save_pair(df: pd.DataFrame, processed_dir: Path, name: str, fmt: str = "parquet") -> Path:
    ensure_dir(processed_dir)

    if fmt == "parquet":
        out_path = processed_dir / f"pair_{name}.parquet"
        df.to_parquet(out_path, index=False, engine="pyarrow")
        return out_path

    # fallback compatível (se você ainda quiser csv.gz)
    out_path = processed_dir / f"pair_{name}.csv.gz"
    df.to_csv(out_path, index=False, compression="gzip", sep=";")
    return out_path


def _read_interim_year(interim_dir: Path, year: int) -> pd.DataFrame:
    path = interim_dir / f"pede_{year}_interim.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Interim file not found: {path}. Run preprocess first.")
    return pd.read_parquet(path, engine="pyarrow")


def run_build_dataset(
    raw_dir: Path,
    interim_dir: Path,
    processed_dir: Path,
    *,
    id_col: str | None = "ra",
    ian_col: str = "ian",
    processed_format: str = "parquet",
) -> dict[str, Any]:
    """
    Pipeline:
    1) raw (csv) -> interim (parquet) via src.data.preprocess
    2) interim -> pares t->t+1 via src.features.build_pairs
    3) pares -> processed (parquet/csv.gz) + stats.json
    """
    ensure_dir(interim_dir)
    ensure_dir(processed_dir)

       # normaliza id_col para o padrão do preprocess (colunas em minúsculo)
    if id_col is None:
        id_col = "ra"
    id_col = id_col.strip().lower()

    # 1) raw -> interim (parquet)
    preprocess_year_file(raw_dir / "PEDE2022-Table 1.csv", 2022, out_dir=interim_dir, out_format="parquet", sep=";")
    preprocess_year_file(raw_dir / "PEDE2023-Table 1.csv", 2023, out_dir=interim_dir, out_format="parquet", sep=";")
    preprocess_year_file(raw_dir / "PEDE2024-Table 1.csv", 2024, out_dir=interim_dir, out_format="parquet", sep=";")

    # 2) carregar interim
    years = {
        2022: _read_interim_year(interim_dir, 2022),
        2023: _read_interim_year(interim_dir, 2023),
        2024: _read_interim_year(interim_dir, 2024),
    }

    # 3) construir pares
    pairs = build_all_pairs(years, id_col=id_col, ian_col=ian_col)

    stats: dict[str, Any] = {"outputs": {}, "pairs": {}}

    for name, df in pairs.items():
        # checks simples (anti-leakage básico)
        if "ano" in df.columns:
            raise ValueError("Leakage/erro: coluna operacional 'ano' não deve estar no dataset de pares.")
        if "y" not in df.columns:
            raise ValueError("Dataset de pares sem coluna target 'y'.")
        if not df["y"].isin([0, 1]).all():
            raise ValueError("Target 'y' deve ser binário (0/1).")

        out_path = save_pair(df, processed_dir, name, fmt=processed_format)
        stats["outputs"][name] = str(out_path)
        stats["pairs"][name] = compute_basic_stats(df)

    stats_path = processed_dir / "stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    return stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build temporal datasets t->t+1 without leakage.")
    p.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    p.add_argument("--interim-dir", type=Path, default=Path("data/interim"))
    p.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--id-col", type=str, default="ra")
    p.add_argument("--ian-col", type=str, default="ian")
    p.add_argument("--processed-format", type=str, default="parquet", choices=["parquet", "csv"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_build_dataset(
        args.raw_dir,
        args.interim_dir,
        args.processed_dir,
        id_col=args.id_col,
        ian_col=args.ian_col,
        processed_format=args.processed_format,
    )


if __name__ == "__main__":
    main()