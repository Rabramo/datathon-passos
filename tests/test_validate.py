# tests/test_validate.py
from __future__ import annotations

import pandas as pd
import pytest

from src.data.validate import assert_common_ids, assert_unique_id, detect_id_column


def test_detect_id_column_prefers_ra() -> None:
    df = pd.DataFrame({"ra": ["1", "2"], "x": [1, 2]})
    assert detect_id_column(df) == "ra"


def test_detect_id_column_case_insensitive_fallback() -> None:
    df = pd.DataFrame({"RA": ["1", "2"], "x": [1, 2]})
    # candidates incluem "RA", então retorna direto
    assert detect_id_column(df) == "RA"

    df2 = pd.DataFrame({"Ra": ["1", "2"], "x": [1, 2]})
    # não está nos candidates, mas cai no fallback lower_map
    assert detect_id_column(df2) == "Ra"


def test_detect_id_column_raises_when_missing() -> None:
    df = pd.DataFrame({"x": [1, 2]})
    with pytest.raises(ValueError, match="detectar a coluna de ID"):
        _ = detect_id_column(df)


def test_assert_unique_id_raises_when_id_col_missing() -> None:
    df = pd.DataFrame({"x": ["1", "2"]})
    with pytest.raises(ValueError, match="não existe"):
        assert_unique_id(df, id_col="ra", year=2022)


def test_assert_unique_id_raises_when_nulls() -> None:
    df = pd.DataFrame({"ra": ["1", None, "3"]})
    with pytest.raises(ValueError, match="valores nulos"):
        assert_unique_id(df, id_col="ra", year=2022)


def test_assert_unique_id_raises_when_duplicates() -> None:
    df = pd.DataFrame({"ra": ["1", "1", "2"]})
    with pytest.raises(ValueError, match="duplicatas"):
        assert_unique_id(df, id_col="ra", year=2022)


def test_assert_common_ids_ok_and_report() -> None:
    df_t = pd.DataFrame({"ra": ["1", "2", "3"]})
    df_t1 = pd.DataFrame({"ra": ["3", "4"]})

    rep = assert_common_ids(df_t, df_t1, id_col="ra", year_t=2022, year_t1=2023, min_common=1)
    assert rep.n_t == 3
    assert rep.n_t1 == 2
    assert rep.n_common == 1
    assert rep.id_col == "ra"
    assert rep.year_t == 2022
    assert rep.year_t1 == 2023


def test_assert_common_ids_raises_when_insufficient() -> None:
    df_t = pd.DataFrame({"ra": ["1", "2"]})
    df_t1 = pd.DataFrame({"ra": ["3", "4"]})

    with pytest.raises(ValueError, match="Interseção insuficiente"):
        _ = assert_common_ids(df_t, df_t1, id_col="ra", year_t=2022, year_t1=2023, min_common=1)