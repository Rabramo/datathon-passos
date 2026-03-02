# tests/test_preprocess_cli_io.py
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.preprocess import _save_df, build_argparser, main, preprocess_year_file


def test_save_df_parquet_and_csv(tmp_path: Path) -> None:
    df = pd.DataFrame({"ra": ["1", "2"], "x": [1, 2]})

    p_parquet = tmp_path / "out.parquet"
    out_pq = _save_df(df, p_parquet)
    assert out_pq.exists()
    df_pq = pd.read_parquet(out_pq, engine="pyarrow")
    assert df_pq.shape == (2, 2)

    p_csv = tmp_path / "out.csv"
    out_csv = _save_df(df, p_csv)
    assert out_csv.exists()
    df_csv = pd.read_csv(out_csv)
    assert df_csv.shape == (2, 2)


def test_preprocess_year_file_writes_output(tmp_path: Path) -> None:
    # CSV com separador ';' e colunas mínimas + variações para cobrir ramos
    raw = tmp_path / "raw.csv"
    raw.write_text(
        "RA;ANO_NASC;PEDRA_23;IPP;DESTAQUE_IPV;AVALIADOR1\n"
        "1;2010;Quartzo;;texto;X\n"
        "2;2009;Topázio;;texto;Y\n",
        encoding="utf-8",
    )

    out_dir = tmp_path / "interim"
    out_path = preprocess_year_file(
        in_path=raw,
        year=2022,
        out_dir=out_dir,
        out_format="parquet",
        encoding="utf-8",
        sep=";",
    )

    assert out_path.exists()
    assert out_path.name == "pede_2022_interim.parquet"

    df_out = pd.read_parquet(out_path, engine="pyarrow")
    # checagens básicas: normalização + drops + features derivadas
    assert "ra" in df_out.columns
    assert "destaque_ipv" not in df_out.columns
    assert "avaliador1" not in df_out.columns
    assert "pedra" in df_out.columns  # unificação pedra_23 -> pedra
    assert "idade" in df_out.columns  # derivada de ano_nasc


def test_build_argparser_parses_args() -> None:
    p = build_argparser()
    args = p.parse_args(
        ["--input", "x.csv", "--year", "2022", "--out-dir", "data/interim", "--out-format", "csv", "--sep", ";"]
    )
    assert args.input == "x.csv"
    assert args.year == 2022
    assert args.out_dir == "data/interim"
    assert args.out_format == "csv"
    assert args.sep == ";"


def test_main_runs_end_to_end(tmp_path: Path, monkeypatch) -> None:
    raw = tmp_path / "raw.csv"
    raw.write_text(
        "ra;ano_nasc;pedra_2023\n"
        "1;2010;Agata\n",
        encoding="utf-8",
    )

    out_dir = tmp_path / "interim"
    out_dir.mkdir()

    # simula chamada CLI
    monkeypatch.setenv("PYTHONIOENCODING", "utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "preprocess",
            "--input",
            str(raw),
            "--year",
            "2023",
            "--out-dir",
            str(out_dir),
            "--out-format",
            "csv",
            "--sep",
            ";",
        ],
    )

    main()

    # verifica que o arquivo foi criado
    out_path = out_dir / "pede_2023_interim.csv"
    assert out_path.exists()