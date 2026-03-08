from pathlib import Path

import pandas as pd
import pytest

from src.models.train_api import _load_df, _make_model


def test_load_df_reads_parquet(tmp_path: Path):
    path = tmp_path / "sample.parquet"
    pd.DataFrame({"a": [1], "b": [2]}).to_parquet(path, index=False)

    df = _load_df(path)

    assert list(df.columns) == ["a", "b"]
    assert len(df) == 1


def test_load_df_raises_for_missing_file(tmp_path: Path):
    missing = tmp_path / "missing.parquet"

    with pytest.raises(FileNotFoundError):
        _load_df(missing)


def test_load_df_raises_for_unsupported_or_invalid_content(tmp_path: Path):
    path = tmp_path / "sample.txt"
    path.write_text("abc")

    with pytest.raises(Exception):
        _load_df(path)


@pytest.mark.parametrize("internal_key", ["logreg", "tree"])
def test_make_model_supported_keys(internal_key):
    model = _make_model(internal_key, seed=42)
    assert model is not None


def test_make_model_invalid_key_raises():
    with pytest.raises(ValueError):
        _make_model("inexistente", seed=42)