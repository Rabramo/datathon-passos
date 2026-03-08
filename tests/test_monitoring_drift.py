from __future__ import annotations

import json

import pandas as pd

from src.monitoring.drift import (
    build_summary,
    categorical_psi,
    compute_drift_rows,
    numeric_psi,
    save_outputs,
)


def test_numeric_psi_increases_with_distribution_shift():
    ref = pd.Series([0, 0, 1, 1, 2, 2, 3, 3])
    cur_same = pd.Series([0, 0, 1, 1, 2, 2, 3, 3])
    cur_shifted = pd.Series([10, 10, 11, 11, 12, 12, 13, 13])

    psi_same = numeric_psi(ref, cur_same)
    psi_shifted = numeric_psi(ref, cur_shifted)

    assert psi_same < 0.05
    assert psi_shifted > psi_same


def test_categorical_psi_detects_shift():
    ref = pd.Series(["A", "A", "B", "B", "B"])
    cur = pd.Series(["A", "C", "C", "C", "C"])

    score = categorical_psi(ref, cur)
    assert score > 0.1


def test_compute_and_save_outputs(tmp_path):
    df_ref = pd.DataFrame(
        {
            "feat_num": [1, 2, 3, 4, 5, 6],
            "feat_cat": ["A", "A", "B", "B", "B", "C"],
            "y": [0, 1, 0, 1, 0, 1],
        }
    )
    df_cur = pd.DataFrame(
        {
            "feat_num": [10, 11, 12, 13, 14, 15],
            "feat_cat": ["C", "C", "C", "C", "A", "A"],
            "y": [0, 1, 0, 1, 0, 1],
        }
    )

    rows = compute_drift_rows(
        df_ref,
        df_cur,
        warn_threshold=0.10,
        alert_threshold=0.25,
        exclude_cols={"y"},
    )
    summary = build_summary(
        rows,
        ref_path=tmp_path / "ref.parquet",
        cur_path=tmp_path / "cur.parquet",
        warn_threshold=0.10,
        alert_threshold=0.25,
    )
    summary_path, table_path = save_outputs(summary, tmp_path)

    assert summary["n_features"] == 2
    assert summary_path.exists()
    assert table_path.exists()

    loaded = json.loads(summary_path.read_text(encoding="utf-8"))
    assert loaded["status"] in {"ok", "warn", "alert"}
    assert len(loaded["features"]) == 2

