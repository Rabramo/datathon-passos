from __future__ import annotations

import pandas as pd

from src.features.build_pairs import build_temporal_pair

def test_build_temporal_pair_uses_only_common_ids():
    df_t = pd.DataFrame({"ra": ["1", "2", "3"], "feat": [1, 2, 3], "ano": [2022, 2022, 2022]})
    df_t1 = pd.DataFrame({"ra": ["2", "3", "4"], "ian": [10, 9, 10], "ano": [2023, 2023, 2023]})

    out = build_temporal_pair(df_t, df_t1, 2022, 2023, id_col="ra", ian_col="ian")

    # mantém só IDs comuns (2 e 3)
    assert out["ra"].tolist() == ["2", "3"]
    assert set(["ra", "year_t", "year_t1", "y"]).issubset(out.columns)
    assert out["y"].tolist() == [0, 1]  # 10->0, 9->1
    assert "ano" not in out.columns      # drop_feature_cols

