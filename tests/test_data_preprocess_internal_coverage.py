import pandas as pd

from src.data.preprocess import _drop_cols_if_exist, _coalesce_columns


def test_drop_cols_if_exist_drops_when_present():
    df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
    out = _drop_cols_if_exist(df, ["b", "x"])
    assert "b" not in out.columns
    assert "a" in out.columns
    assert "c" in out.columns


def test_drop_cols_if_exist_noop_when_none_present():
    df = pd.DataFrame({"a": [1], "c": [3]})
    out = _drop_cols_if_exist(df, ["b", "x"])
    assert out.columns.tolist() == ["a", "c"]


def test_coalesce_columns_creates_target_and_fills_from_candidates():
    df = pd.DataFrame({"cand1": [None, 10], "cand2": [5, None]})

    # target não existe -> cria e faz fillna com candidatas
    out = _coalesce_columns(df, target="target", candidates=["cand1", "cand2"])

    assert "target" in out.columns
    # linha0: cand1 None, cand2 5 -> target 5
    assert out.loc[0, "target"] == 5
    # linha1: cand1 10 -> target 10
    assert out.loc[1, "target"] == 10
    # candidatas devem ser removidas
    assert "cand1" not in out.columns
    assert "cand2" not in out.columns