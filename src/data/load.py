from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd


@dataclass(frozen=True)
class DataPaths:
    raw_dir: Path
    interim_dir: Path | None = None

    @property
    def pede_2022_raw(self) -> Path:
        return self.raw_dir / "PEDE2022-Table 1.csv"

    @property
    def pede_2023_raw(self) -> Path:
        return self.raw_dir / "PEDE2023-Table 1.csv"

    @property
    def pede_2024_raw(self) -> Path:
        return self.raw_dir / "PEDE2024-Table 1.csv"

    @property
    def pede_2022_interim(self) -> Path:
        if self.interim_dir is None:
            raise ValueError("interim_dir não configurado em DataPaths")
        return self.interim_dir / "pede_2022_interim.parquet"

    @property
    def pede_2023_interim(self) -> Path:
        if self.interim_dir is None:
            raise ValueError("interim_dir não configurado em DataPaths")
        return self.interim_dir / "pede_2023_interim.parquet"

    @property
    def pede_2024_interim(self) -> Path:
        if self.interim_dir is None:
            raise ValueError("interim_dir não configurado em DataPaths")
        return self.interim_dir / "pede_2024_interim.parquet"


def read_csv(path: Path, sep: str = ";") -> pd.DataFrame:
    # PEDE usa separador ';'
    return pd.read_csv(path, sep=sep, engine="python", encoding="utf-8")


def read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path, engine="pyarrow")


def load_years_raw(paths: DataPaths) -> dict[int, pd.DataFrame]:
    return {
        2022: read_csv(paths.pede_2022_raw),
        2023: read_csv(paths.pede_2023_raw),
        2024: read_csv(paths.pede_2024_raw),
    }


def load_years_interim(paths: DataPaths) -> dict[int, pd.DataFrame]:
    return {
        2022: read_parquet(paths.pede_2022_interim),
        2023: read_parquet(paths.pede_2023_interim),
        2024: read_parquet(paths.pede_2024_interim),
    }