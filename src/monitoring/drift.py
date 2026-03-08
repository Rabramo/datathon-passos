from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EPS = 1e-6
DEFAULT_EXCLUDE_COLS = {"y", "year_t", "year_t1", "ano"}


@dataclass(frozen=True)
class DriftRow:
    feature: str
    kind: str
    score: float
    status: str
    n_ref: int
    n_cur: int


def load_table(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset não encontrado: {p}")

    suffixes = [s.lower() for s in p.suffixes]
    if suffixes and suffixes[-1] == ".parquet":
        return pd.read_parquet(p)
    if suffixes[-1:] == [".csv"] or suffixes[-2:] == [".csv", ".gz"]:
        return pd.read_csv(p)

    raise ValueError(f"Formato não suportado para dataset: {p}")


def _normalize_probs(values: pd.Series) -> pd.Series:
    total = float(values.sum())
    if total <= 0:
        return pd.Series([1.0 / max(len(values), 1)] * len(values), index=values.index)
    return values / total


def _psi_from_probs(p_ref: np.ndarray, p_cur: np.ndarray) -> float:
    p_ref = np.clip(p_ref, EPS, 1.0)
    p_cur = np.clip(p_cur, EPS, 1.0)
    return float(np.sum((p_cur - p_ref) * np.log(p_cur / p_ref)))


def numeric_psi(s_ref: pd.Series, s_cur: pd.Series, bins: int = 10) -> float:
    ref = pd.to_numeric(s_ref, errors="coerce").dropna()
    cur = pd.to_numeric(s_cur, errors="coerce").dropna()
    if ref.empty or cur.empty:
        return 0.0

    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0

    edges = np.concatenate(([-np.inf], edges[1:-1], [np.inf]))

    ref_bins = pd.cut(ref, bins=edges, include_lowest=True)
    cur_bins = pd.cut(cur, bins=edges, include_lowest=True)

    p_ref = _normalize_probs(ref_bins.value_counts(sort=False)).to_numpy()
    p_cur = _normalize_probs(cur_bins.value_counts(sort=False)).reindex(ref_bins.cat.categories, fill_value=0.0).to_numpy()

    return _psi_from_probs(p_ref, p_cur)


def categorical_psi(s_ref: pd.Series, s_cur: pd.Series) -> float:
    ref = s_ref.astype("string").fillna("__na__")
    cur = s_cur.astype("string").fillna("__na__")

    p_ref = _normalize_probs(ref.value_counts(dropna=False))
    p_cur = _normalize_probs(cur.value_counts(dropna=False))

    idx = p_ref.index.union(p_cur.index)
    p_ref = p_ref.reindex(idx, fill_value=0.0).to_numpy()
    p_cur = p_cur.reindex(idx, fill_value=0.0).to_numpy()

    return _psi_from_probs(p_ref, p_cur)


def classify(score: float, warn_threshold: float, alert_threshold: float) -> str:
    if score >= alert_threshold:
        return "alert"
    if score >= warn_threshold:
        return "warn"
    return "ok"


def compute_drift_rows(
    df_ref: pd.DataFrame,
    df_cur: pd.DataFrame,
    *,
    warn_threshold: float,
    alert_threshold: float,
    exclude_cols: set[str] | None = None,
) -> list[DriftRow]:
    exclude = exclude_cols or set()

    rows: list[DriftRow] = []
    common_cols = [c for c in df_ref.columns if c in df_cur.columns and c not in exclude]

    for col in common_cols:
        s_ref = df_ref[col]
        s_cur = df_cur[col]

        numeric = pd.api.types.is_numeric_dtype(s_ref) and pd.api.types.is_numeric_dtype(s_cur)
        if numeric:
            score = numeric_psi(s_ref, s_cur)
            kind = "numeric_psi"
        else:
            score = categorical_psi(s_ref, s_cur)
            kind = "categorical_psi"

        rows.append(
            DriftRow(
                feature=str(col),
                kind=kind,
                score=float(score),
                status=classify(float(score), warn_threshold, alert_threshold),
                n_ref=int(s_ref.notna().sum()),
                n_cur=int(s_cur.notna().sum()),
            )
        )

    rows.sort(key=lambda r: r.score, reverse=True)
    return rows


def build_summary(
    rows: list[DriftRow],
    *,
    ref_path: Path,
    cur_path: Path,
    warn_threshold: float,
    alert_threshold: float,
) -> dict[str, Any]:
    counts = {"ok": 0, "warn": 0, "alert": 0}
    for row in rows:
        counts[row.status] = counts.get(row.status, 0) + 1

    max_score = max((row.score for row in rows), default=0.0)
    max_feature = rows[0].feature if rows else None
    status = "alert" if counts["alert"] > 0 else ("warn" if counts["warn"] > 0 else "ok")

    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "status": status,
        "reference_path": str(ref_path),
        "current_path": str(cur_path),
        "warn_threshold": float(warn_threshold),
        "alert_threshold": float(alert_threshold),
        "n_features": len(rows),
        "counts": counts,
        "max_score": float(max_score),
        "max_feature": max_feature,
        "features": [asdict(r) for r in rows],
    }


def save_outputs(summary: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

    summary_path = output_dir / f"drift_summary_{ts}.json"
    table_path = output_dir / f"drift_features_{ts}.csv"

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(summary["features"]).to_csv(table_path, index=False)

    latest_summary = output_dir / "drift_summary_latest.json"
    latest_table = output_dir / "drift_features_latest.csv"
    latest_summary.write_text(summary_path.read_text(encoding="utf-8"), encoding="utf-8")
    latest_table.write_text(table_path.read_text(encoding="utf-8"), encoding="utf-8")

    return summary_path, table_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitoramento de drift por PSI.")
    parser.add_argument("--reference", type=Path, default=Path("data/processed/pair_2022_2023.parquet"))
    parser.add_argument("--current", type=Path, default=Path("data/processed/pair_2023_2024.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/monitoring"))
    parser.add_argument("--warn-threshold", type=float, default=0.10)
    parser.add_argument("--alert-threshold", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df_ref = load_table(args.reference)
    df_cur = load_table(args.current)

    rows = compute_drift_rows(
        df_ref,
        df_cur,
        warn_threshold=float(args.warn_threshold),
        alert_threshold=float(args.alert_threshold),
        exclude_cols=DEFAULT_EXCLUDE_COLS,
    )

    summary = build_summary(
        rows,
        ref_path=args.reference,
        cur_path=args.current,
        warn_threshold=float(args.warn_threshold),
        alert_threshold=float(args.alert_threshold),
    )
    summary_path, table_path = save_outputs(summary, args.output_dir)

    print(f"[drift] status={summary['status']} n_features={summary['n_features']} max_score={summary['max_score']:.4f}")
    print(f"[drift] summary={summary_path}")
    print(f"[drift] table={table_path}")

    # exit code 2 sinaliza alerta crítico de drift
    if summary["status"] == "alert":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

