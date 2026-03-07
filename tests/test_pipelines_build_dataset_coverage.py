import pandas as pd
import pytest

from pathlib import Path

from src.pipelines.build_dataset import save_pair, _read_interim_year


def test_save_pair_csv_fallback(tmp_path: Path):

    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    out = save_pair(df, processed_dir=tmp_path, name="2022_2023", fmt="csv")

    assert out.exists()
    assert out.name == "pair_2022_2023.csv.gz"

    df2 = pd.read_csv(out, compression="gzip", sep=";")
    assert df2.shape == df.shape


def test_read_interim_year_missing_file_raises(tmp_path: Path):
    """
    Cobre o raise FileNotFoundError em _read_interim_year().
    """
    with pytest.raises(FileNotFoundError):
        _read_interim_year(tmp_path, 2099)