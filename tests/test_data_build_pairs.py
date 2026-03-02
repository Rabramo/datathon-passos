# tests/test_data_build_pairs.py
from __future__ import annotations

import pandas as pd
import pytest

from src.data.build_pairs import _make_target_from_ian, build_all_pairs, build_temporal_pair


def test_make_target_from_ian_rule() -> None:
    df_t1 = pd.DataFrame({"ian": [10, 9, 0, 11]})
    y = _make_target_from_ian(df_t1, "ian")
    assert y.tolist() == [0, 1, 1, 1]


def test_build_temporal_pair_inner_join_and_no_leakage() -> None:
    # df_t: features do ano t
    df_t = pd.DataFrame(
        {
            "ra": ["1", "2", "3"],
            "ano": [2022, 2022, 2022],  # deve ser removido (drop_feature_cols)
            "feat_a": [1.0, 2.0, 3.0],
        }
    )

    # df_t1: contém o ian para formar y; inclui um id extra ("4") e falta "2"
    df_t1 = pd.DataFrame(
        {
            "ra": ["1", "3", "4"],
            "ian": [10, 7, 10],
            "qualquer_coisa_t1": [100, 200, 300],  # não pode vazar como feature
        }
    )

    out = build_temporal_pair(df_t, df_t1, 2022, 2023, id_col="ra", ian_col="ian")

    # inner join: apenas IDs comuns {"1","3"} e na ordem do df_t
    assert out["ra"].tolist() == ["1", "3"]
    assert out["year_t"].unique().tolist() == [2022]
    assert out["year_t1"].unique().tolist() == [2023]

    # target: baseado somente no ian do t+1 alinhado
    # ra=1 -> ian=10 => y=0 ; ra=3 -> ian=7 => y=1
    assert out["y"].tolist() == [0, 1]

    # não vazamento: coluna t+1 não deve existir
    assert "qualquer_coisa_t1" not in out.columns

    # coluna operacional 'ano' dropada do feature set
    assert "ano" not in out.columns

    # features do t presentes
    assert "feat_a" in out.columns


def test_build_temporal_pair_raises_if_missing_ian_col() -> None:
    df_t = pd.DataFrame({"ra": ["1"], "feat": [1]})
    df_t1 = pd.DataFrame({"ra": ["1"], "ian_outra": [10]})

    with pytest.raises(ValueError, match="Column 'ian' not found"):
        _ = build_temporal_pair(df_t, df_t1, 2022, 2023, id_col="ra", ian_col="ian")


def test_build_temporal_pair_raises_if_missing_id_in_t1() -> None:
    df_t = pd.DataFrame({"ra": ["1"], "feat": [1]})
    df_t1 = pd.DataFrame({"outra_id": ["1"], "ian": [10]})

    with pytest.raises(ValueError, match="ID column 'ra' not found"):
        _ = build_temporal_pair(df_t, df_t1, 2022, 2023, id_col="ra", ian_col="ian")


def test_build_all_pairs_keys_and_shapes() -> None:
    years = {
        2022: pd.DataFrame({"ra": ["1", "2"], "feat": [1, 2]}),
        2023: pd.DataFrame({"ra": ["1", "2"], "ian": [10, 9], "feat": [3, 4]}),
        2024: pd.DataFrame({"ra": ["1", "2"], "ian": [10, 10], "feat": [5, 6]}),
    }

    out = build_all_pairs(years, id_col="ra", ian_col="ian")
    assert set(out.keys()) == {"2022_2023", "2023_2024"}
    assert out["2022_2023"].shape[0] == 2
    assert out["2023_2024"].shape[0] == 2