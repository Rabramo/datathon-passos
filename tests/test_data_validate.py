import pandas as pd
import pytest

from src.data.validate import assert_common_ids, assert_unique_id, detect_id_column


def test_detect_id_column_retorna_ra_quando_presente():
    df = pd.DataFrame({"ra": [1, 2], "nome": ["a", "b"]})

    id_col = detect_id_column(df)

    assert id_col == "ra"


def test_detect_id_column_retorna_coluna_case_insensitive():
    df = pd.DataFrame({"RA": [1, 2], "nome": ["a", "b"]})

    id_col = detect_id_column(df)

    assert id_col == "RA"


def test_detect_id_column_levanta_erro_quando_nao_encontra():
    df = pd.DataFrame({"nome": ["a", "b"]})

    with pytest.raises(ValueError, match="Não foi possível detectar a coluna de ID"):
        detect_id_column(df)


def test_assert_unique_id_ok():
    df = pd.DataFrame({"ra": [1, 2, 3]})

    assert_unique_id(df, id_col="ra", year=2022)


def test_assert_unique_id_levanta_erro_quando_coluna_nao_existe():
    df = pd.DataFrame({"id": [1, 2, 3]})

    with pytest.raises(ValueError, match="não existe"):
        assert_unique_id(df, id_col="ra", year=2022)


def test_assert_unique_id_levanta_erro_quando_tem_nulos():
    df = pd.DataFrame({"ra": [1, None, 3]})

    with pytest.raises(ValueError, match="valores nulos"):
        assert_unique_id(df, id_col="ra", year=2022)


def test_assert_unique_id_levanta_erro_quando_tem_duplicatas():
    df = pd.DataFrame({"ra": [1, 1, 2]})

    with pytest.raises(ValueError, match="duplicatas"):
        assert_unique_id(df, id_col="ra", year=2022)


def test_assert_common_ids_retorna_relatorio():
    df_t = pd.DataFrame({"ra": [1, 2, 3]})
    df_t1 = pd.DataFrame({"ra": [2, 3, 4]})

    report = assert_common_ids(
        df_t=df_t,
        df_t1=df_t1,
        id_col="ra",
        year_t=2022,
        year_t1=2023,
        min_common=1,
    )

    assert report.year_t == 2022
    assert report.year_t1 == 2023
    assert report.id_col == "ra"
    assert report.n_t == 3
    assert report.n_t1 == 3
    assert report.n_common == 2


def test_assert_common_ids_levanta_erro_quando_intersecao_insuficiente():
    df_t = pd.DataFrame({"ra": [1, 2]})
    df_t1 = pd.DataFrame({"ra": [3, 4]})

    with pytest.raises(ValueError, match="Interseção insuficiente de IDs"):
        assert_common_ids(
            df_t=df_t,
            df_t1=df_t1,
            id_col="ra",
            year_t=2022,
            year_t1=2023,
            min_common=1,
        )