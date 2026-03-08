from pathlib import Path

import pandas as pd
import pytest

from src.data.load import read_csv, read_parquet


def test_read_csv_reads_dataframe(tmp_path: Path):
    path = tmp_path / "sample.csv"
    pd.DataFrame({"id": [1, 2], "x": [10, 20]}).to_csv(path, index=False, sep=";")

    df = read_csv(path)

    assert df.shape == (2, 2)
    assert list(df.columns) == ["id", "x"]


def test_read_csv_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        read_csv(tmp_path / "missing.csv")


def test_read_parquet_reads_dataframe(tmp_path: Path):
    path = tmp_path / "sample.parquet"
    pd.DataFrame({"id": [1], "x": [99]}).to_parquet(path, index=False)

    df = read_parquet(path)

    assert df.shape == (1, 2)
    assert list(df.columns) == ["id", "x"]


def test_read_parquet_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        read_parquet(tmp_path / "missing.parquet")