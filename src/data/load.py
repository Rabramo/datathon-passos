from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd


@dataclass(frozen=True)
class DataPaths:
    raw_dir: Path

    @property
    def pede_2022(self) -> Path:
        return self.raw_dir / "PEDE2022-Table 1.csv"

    @property
    def pede_2023(self) -> Path:
        return self.raw_dir / "PEDE2023-Table 1.csv"

    @property
    def pede_2024(self) -> Path:
        return self.raw_dir / "PEDE2024-Table 1.csv"


def read_csv(path: Path) -> pd.DataFrame:
    # PEDE usa separador ';'
    return pd.read_csv(path, sep=";", engine="python", encoding="utf-8")


def load_years(paths: DataPaths) -> dict[int, pd.DataFrame]:
    return {
        2022: read_csv(paths.pede_2022),
        2023: read_csv(paths.pede_2023),
        2024: read_csv(paths.pede_2024),
    }