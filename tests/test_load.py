# tests/test_load.py
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data.load import DataPaths, load_years_interim, load_years_raw, read_csv, read_parquet


def test_datapaths_raw_properties(tmp_path: Path) -> None:
    paths = DataPaths(raw_dir=tmp_path)

    assert paths.pede_2022_raw == tmp_path / "PEDE2022-Table 1.csv"
    assert paths.pede_2023_raw == tmp_path / "PEDE2023-Table 1.csv"
    assert paths.pede_2024_raw == tmp_path / "PEDE2024-Table 1.csv"


def test_datapaths_interim_requires_interim_dir(tmp_path: Path) -> None:
    paths = DataPaths(raw_dir=tmp_path, interim_dir=None)

    with pytest.raises(ValueError):
        _ = paths.pede_2022_interim

    with pytest.raises(ValueError):
        _ = paths.pede_2023_interim

    with pytest.raises(ValueError):
        _ = paths.pede_2024_interim


def test_read_csv_default_sep_semicolon(tmp_path: Path) -> None:
    # cria um CSV com ';' (padrão do PEDE)
    p = tmp_path / "x.csv"
    p.write_text("a;b\n1;2\n", encoding="utf-8")

    df = read_csv(p)
    assert list(df.columns) == ["a", "b"]
    assert df.shape == (1, 2)
    assert df.loc[0, "a"] == 1
    assert df.loc[0, "b"] == 2


def test_read_parquet_roundtrip(tmp_path: Path) -> None:
    df0 = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    p = tmp_path / "x.parquet"
    df0.to_parquet(p, index=False, engine="pyarrow")

    df = read_parquet(p)
    assert df.shape == (2, 2)
    assert df["a"].tolist() == [1, 2]
    assert df["b"].tolist() == ["x", "y"]


def test_load_years_raw_reads_expected_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # cria "arquivos esperados" no raw_dir
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    (raw_dir / "PEDE2022-Table 1.csv").write_text("a;b\n1;2\n", encoding="utf-8")
    (raw_dir / "PEDE2023-Table 1.csv").write_text("a;b\n3;4\n", encoding="utf-8")
    (raw_dir / "PEDE2024-Table 1.csv").write_text("a;b\n5;6\n", encoding="utf-8")

    paths = DataPaths(raw_dir=raw_dir)

    out = load_years_raw(paths)
    assert set(out.keys()) == {2022, 2023, 2024}
    assert out[2022].iloc[0].tolist() == [1, 2]
    assert out[2023].iloc[0].tolist() == [3, 4]
    assert out[2024].iloc[0].tolist() == [5, 6]


def test_load_years_interim_reads_expected_files(tmp_path: Path) -> None:
    interim_dir = tmp_path / "interim"
    interim_dir.mkdir()
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    # cria parquets esperados
    pd.DataFrame({"a": [1]}).to_parquet(interim_dir / "pede_2022_interim.parquet", index=False, engine="pyarrow")
    pd.DataFrame({"a": [2]}).to_parquet(interim_dir / "pede_2023_interim.parquet", index=False, engine="pyarrow")
    pd.DataFrame({"a": [3]}).to_parquet(interim_dir / "pede_2024_interim.parquet", index=False, engine="pyarrow")

    paths = DataPaths(raw_dir=raw_dir, interim_dir=interim_dir)

    out = load_years_interim(paths)
    assert set(out.keys()) == {2022, 2023, 2024}
    assert out[2022]["a"].tolist() == [1]
    assert out[2023]["a"].tolist() == [2]
    assert out[2024]["a"].tolist() == [3]