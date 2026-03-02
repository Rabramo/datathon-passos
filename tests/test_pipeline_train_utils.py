# tests/test_pipeline_train_utils.py
from __future__ import annotations

import pandas as pd

from src.models.train import TrainSpec
from src.pipelines.train import _align_pair_to_train_schema, _drop_all_nan_features, _schema_diff


def _mk_df(
    cols: dict[str, list],
) -> pd.DataFrame:
    return pd.DataFrame(cols)


def test_schema_diff_detects_missing_and_extra() -> None:
    spec = TrainSpec(id_col="ra")

    df_train = _mk_df(
        {
            "ra": ["1", "2"],
            "year_t": [2022, 2022],
            "year_t1": [2023, 2023],
            "y": [0, 1],
            "feat_a": [1.0, 2.0],
        }
    )
    df_test = _mk_df(
        {
            "ra": ["1", "2"],
            "year_t": [2023, 2023],
            "year_t1": [2024, 2024],
            "y": [0, 0],
            "feat_a": [3.0, 4.0],
            "feat_extra": [9.0, 9.0],
        }
    )

    missing, extra = _schema_diff(df_train, df_test, spec)
    assert missing == []
    assert extra == ["feat_extra"]


def test_align_pair_to_train_schema_adds_missing_drops_extra_and_orders() -> None:
    spec = TrainSpec(id_col="ra")

    df_train = _mk_df(
        {
            "ra": ["1", "2"],
            "year_t": [2022, 2022],
            "year_t1": [2023, 2023],
            "y": [0, 1],
            "feat_a": [1.0, 2.0],
            "feat_b": [10.0, 20.0],
        }
    )

    # df_other: falta feat_b e tem feat_extra
    df_other = _mk_df(
        {
            "ra": ["1", "2"],
            "year_t": [2023, 2023],
            "year_t1": [2024, 2024],
            "y": [0, 0],
            "feat_a": [3.0, 4.0],
            "feat_extra": [999.0, 999.0],
        }
    )

    out = _align_pair_to_train_schema(df_train, df_other, spec)

    # deve conter feat_a e feat_b; não deve conter feat_extra
    assert "feat_a" in out.columns
    assert "feat_b" in out.columns
    assert "feat_extra" not in out.columns

    # feat_b deve ter sido criado como NaN
    assert out["feat_b"].isna().all()

    # ordem respeita treino (feat_a depois feat_b, mas validamos pelo split_xy indiretamente)
    # aqui garantimos que ambas existem e o conteúdo de feat_a preserva
    assert out["feat_a"].tolist() == [3.0, 4.0]


def test_drop_all_nan_features_removes_from_both_and_does_not_readd_as_meta() -> None:
    spec = TrainSpec(id_col="ra")

    df_train = _mk_df(
        {
            "ra": ["1", "2"],
            "year_t": [2022, 2022],
            "year_t1": [2023, 2023],
            "y": [0, 1],
            "feat_ok": [1.0, 2.0],
            "ipp": [None, None],  # 100% nula no treino
        }
    )

    df_other = _mk_df(
        {
            "ra": ["1", "2"],
            "year_t": [2023, 2023],
            "year_t1": [2024, 2024],
            "y": [0, 0],
            "feat_ok": [3.0, 4.0],
            "ipp": [5.0, 6.0],  # no teste existe, mas deve ser removida por estar all-NaN no treino
        }
    )

    tr, te, removed = _drop_all_nan_features(df_train, df_other, spec)

    assert removed == ["ipp"]
    assert "ipp" not in tr.columns
    assert "ipp" not in te.columns

    # preserva feature válida
    assert "feat_ok" in tr.columns
    assert "feat_ok" in te.columns