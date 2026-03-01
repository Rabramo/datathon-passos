# tests/test_seed.py
from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from sklearn.ensemble import RandomForestClassifier


def _make_year_df(year: int) -> pd.DataFrame:
    """
    DF sintético semelhante ao que sai do preprocess:
    - id: 'ra'
    - coluna operacional: 'ano' (deve ser ignorada no par)
    - features numéricas simples
    - 'ian' existe no ano t e t+1 (t+1 para criar y)
    """
    return pd.DataFrame(
        {
            "ra": ["A", "B", "C", "D"],
            "ano": [year] * 4,
            "feat1": [1.0, 2.0, 3.0, 4.0],
            "feat2": [10.0, 20.0, 30.0, 40.0],
            "ian": [10, 9, 10, 8],
        }
    )


def test_build_pairs_is_deterministic():
    # Import local aqui para evitar erro se módulo não existir no momento do discovery
    from src.features.build_pairs import build_temporal_pair

    df_t = _make_year_df(2022)
    df_t1 = _make_year_df(2023)

    out1 = build_temporal_pair(df_t, df_t1, 2022, 2023, id_col="ra", ian_col="ian")
    out2 = build_temporal_pair(df_t, df_t1, 2022, 2023, id_col="ra", ian_col="ian")

    # Mesma saída, inclusive ordem e dtypes
    assert_frame_equal(out1, out2, check_dtype=True, check_like=False)


def test_sklearn_training_is_reproducible_with_fixed_seed():
    """
    Garante que um modelo estocástico do scikit-learn produz
    predições idênticas quando random_state é fixado.
    """
    # Dataset pequeno e determinístico
    X = pd.DataFrame(
        {
            "feat1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "feat2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        }
    )
    y = np.array([0, 1, 0, 1, 0, 1])

    seed = 42

    m1 = RandomForestClassifier(
        n_estimators=50,
        random_state=seed,
        n_jobs=1,  # ajuda a reduzir variação por paralelismo
    )
    m2 = RandomForestClassifier(
        n_estimators=50,
        random_state=seed,
        n_jobs=1,
    )

    m1.fit(X, y)
    m2.fit(X, y)

    p1 = m1.predict_proba(X)
    p2 = m2.predict_proba(X)

    # Igualdade exata costuma funcionar com random_state + n_jobs=1
    assert np.array_equal(p1, p2)