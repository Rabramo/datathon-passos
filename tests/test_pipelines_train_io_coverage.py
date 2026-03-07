import pandas as pd
import pytest
from pathlib import Path

from src.pipelines.train import _load_pair


def test_load_pair_reads_parquet_when_exists(tmp_path: Path):
    processed = tmp_path
    stem = "pair_2022_2023"

    df = pd.DataFrame({"x": [1, 2]})
    p = processed / f"{stem}.parquet"
    df.to_parquet(p, index=False, engine="pyarrow")

    out = _load_pair(processed, stem)
    assert out.equals(df)


def test_load_pair_reads_csv_gz_when_parquet_missing(tmp_path: Path):
    processed = tmp_path
    stem = "pair_2022_2023"

    df = pd.DataFrame({"x": [1, 2]})
    p = processed / f"{stem}.csv.gz"
    df.to_csv(p, index=False, compression="gzip", sep=";")

    out = _load_pair(processed, stem)
    assert out.equals(df)


def test_load_pair_missing_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        _load_pair(tmp_path, "pair_2099_2100")