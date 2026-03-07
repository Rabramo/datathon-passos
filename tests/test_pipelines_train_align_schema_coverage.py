import pandas as pd
from pathlib import Path

from src.pipelines.train import _align_pair_to_train_schema, TrainSpec


def test_align_pair_adds_missing_and_drops_extra_columns():
    """
    Cobre:
      - adiciona colunas faltantes em X_other (missing -> set NaN)
      - remove colunas extras em X_other (extra -> drop)
    """

    spec = TrainSpec()  # usa defaults do seu projeto

    # df_train tem features: a, b
    df_train = pd.DataFrame(
        {
            "ra": [1, 2],
            "year_t": [2022, 2022],
            "year_t1": [2023, 2023],
            "y": [0, 1],
            "a": [10.0, 11.0],
            "b": [20.0, 21.0],
        }
    )

    # df_other tem feature 'a' e uma extra 'c' (e não tem 'b')
    df_other = pd.DataFrame(
        {
            "ra": [3],
            "year_t": [2023],
            "year_t1": [2024],
            "y": [1],
            "a": [12.0],
            "c": [99.0],
        }
    )

    out = _align_pair_to_train_schema(df_train=df_train, df_other=df_other, spec=spec)

    # Deve conter 'b' (adicionada) e não conter 'c' (removida)
    assert "b" in out.columns
    assert "c" not in out.columns