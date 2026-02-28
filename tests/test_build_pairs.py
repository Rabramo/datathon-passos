import pandas as pd

from src.data.build_pairs import build_temporal_pair


def test_build_temporal_pair_creates_y_from_t1_only():
    df_t = pd.DataFrame({"ID": [1, 2, 3], "feat_a": [10, 20, 30]})
    df_t1 = pd.DataFrame({"ID": [1, 2, 3], "IAN": [10, 9, 8], "leaky_col": [999, 999, 999]})

    out = build_temporal_pair(df_t, df_t1, 2022, 2023, id_col="ID", ian_col="IAN")

    assert out["y"].tolist() == [0, 1, 1]
    assert "leaky_col" not in out.columns
    assert "IAN" not in out.columns


def test_build_temporal_pair_aligns_target_with_left_order():
    # df_t1 está em ordem diferente; o y precisa seguir a ordem do df_t
    df_t = pd.DataFrame({"ID": [1, 2, 3], "feat_a": [10, 20, 30]})
    df_t1 = pd.DataFrame({"ID": [3, 1, 2], "IAN": [8, 10, 9]})  # y por ID: 1->0, 2->1, 3->1

    out = build_temporal_pair(df_t, df_t1, 2022, 2023, id_col="ID", ian_col="IAN")

    assert out["ID"].tolist() == [1, 2, 3]
    assert out["y"].tolist() == [0, 1, 1]


def test_build_temporal_pair_uses_only_common_ids():
    df_t = pd.DataFrame({"ID": [1, 2, 3], "feat": [1, 2, 3]})
    df_t1 = pd.DataFrame({"ID": [2, 3, 4], "IAN": [10, 9, 10]})

    out = build_temporal_pair(df_t, df_t1, 2022, 2023, id_col="ID")

    assert out["ID"].tolist() == [2, 3]