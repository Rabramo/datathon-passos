# src/data/preprocess.py
from __future__ import annotations

import argparse
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


# -----------------------------
# Regras alinhadas ao README
# -----------------------------

# Excluir:
# - Nome (redundante com RA)
# - INDE* (derivado)
# - Destaque_* (texto livre sem NLP)
# - Avaliador1..Avaliador6 (alta cardinalidade/viés)
# - Cg/Cf/Ct (rankings dependentes do conjunto)
DROP_COLS_EXACT = {
    "nome",
    "cg",
    "cf",
    "ct",
    "destaque_ieg",
    "destaque_ida",
    "destaque_ipv",
}
DROP_PREFIXES = ("inde",)
DROP_AVALIADORES = {f"avaliador{i}" for i in range(1, 7)}

REC_AV_COLS = ("rec_av1", "rec_av2", "rec_av3", "rec_av4")

# Pedra ordinal (normalizada sem acento)
PEDRA_ORD_MAP = {
    "quartzo": 0,
    "agata": 1,
    "ametista": 2,
    "topazio": 3,
}


@dataclass(frozen=True)
class PreprocessConfig:
    keep_ra: bool = True
    create_features: bool = True
    out_format: str = "parquet"  # parquet|csv (csv só para debug)


# -----------------------------
# Helpers
# -----------------------------

def _strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def normalize_colname(col: str) -> str:
    col = str(col).strip()
    col = _strip_accents(col).lower()
    col = col.replace("º", "")
    col = re.sub(r"[\/\-\.\(\)]", " ", col)
    col = re.sub(r"\s+", " ", col).strip()
    col = col.replace(" ", "_")
    col = re.sub(r"[^a-z0-9_]", "", col)
    col = re.sub(r"_+", "_", col).strip("_")
    return col


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [normalize_colname(c) for c in df.columns]
    return df


def clean_string(s: pd.Series) -> pd.Series:
    out = s.astype("string").str.strip()
    out = out.str.replace(r"\s+", " ", regex=True)
    return out


def coerce_numeric(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return s
    s2 = s.astype("string").str.strip()
    s2 = s2.str.replace(",", ".", regex=False)
    s2 = s2.str.replace(r"[^0-9\.\-]+", "", regex=True)
    return pd.to_numeric(s2, errors="coerce")


def _drop_cols_if_exist(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    existing = [c for c in cols if c in df.columns]
    if existing:
        return df.drop(columns=existing, errors="ignore")
    return df


# -----------------------------
# Normalizações de schema (anti-drift)
# -----------------------------

def _coalesce_columns(df: pd.DataFrame, target: str, candidates: list[str]) -> pd.DataFrame:
    """
    Consolida várias colunas candidatas em uma coluna target,
    mantendo valores existentes e preenchendo com alternativas (fillna).
    Remove candidatas após a consolidação.
    """
    df = df.copy()
    if target not in df.columns:
        df[target] = pd.NA

    for c in candidates:
        if c in df.columns and c != target:
            df[target] = df[target].fillna(df[c])
            df = df.drop(columns=[c])
    return df


def _unify_pedra(df: pd.DataFrame) -> pd.DataFrame:
    """
    Garante que só exista 'pedra' (remove pedra_23, pedra_2023, etc).
    """
    df = df.copy()
    pedra_variants = [c for c in df.columns if re.fullmatch(r"pedra_\d{2,4}", c)]
    if pedra_variants:
        # Consolida em 'pedra'
        df = _coalesce_columns(df, target="pedra", candidates=pedra_variants)
    return df


def _drop_idade_sufixada(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove idade_22, idade_23, idade_2022 etc.
    """
    df = df.copy()
    idade_cols = [c for c in df.columns if re.fullmatch(r"idade_\d{2,4}", c)]
    if idade_cols:
        df = df.drop(columns=idade_cols, errors="ignore")
    return df


def _derive_idade(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Idade deve ser derivada: idade = year - ano_nasc (quando possível).
    Se existir 'idade' prévia, ela é sobrescrita pela derivada.
    """
    df = df.copy()
    if "ano_nasc" in df.columns:
        ano_nasc = coerce_numeric(df["ano_nasc"])
        df["idade"] = year - ano_nasc
    return df


# -----------------------------
# Drop conforme README
# -----------------------------

def drop_baseline_columns(df: pd.DataFrame, keep_ra: bool = True) -> pd.DataFrame:
    df = df.copy()

    cols_to_drop = set(DROP_COLS_EXACT) | set(DROP_AVALIADORES)

    # remove prefixos inde*
    for c in df.columns:
        for pfx in DROP_PREFIXES:
            if c == pfx or c.startswith(pfx + "_"):
                cols_to_drop.add(c)

    if keep_ra:
        cols_to_drop.discard("ra")

    existing = [c for c in df.columns if c in cols_to_drop]
    return df.drop(columns=existing, errors="ignore")


# -----------------------------
# Feature engineering (interim)
# -----------------------------

def _pick_pedra_col(df: pd.DataFrame) -> Optional[str]:
    # após _unify_pedra, deve preferir "pedra"
    if "pedra" in df.columns:
        return "pedra"

    # fallback: tenta pedra_YY / pedra_YYYY; escolhe o maior sufixo
    pedra_year = [c for c in df.columns if re.fullmatch(r"pedra_\d{2,4}", c)]
    if pedra_year:

        def suf(c: str) -> int:
            m = re.search(r"(\d{2,4})$", c)
            return int(m.group(1)) if m else -1

        return sorted(pedra_year, key=suf)[-1]

    return None


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # idade = year - ano_nasc já foi derivada em preprocess_year_df
    # tenure = ano - ano_ingresso
    if "ano" in df.columns and "ano_ingresso" in df.columns:
        df["ano_ingresso"] = coerce_numeric(df["ano_ingresso"])
        df["tenure"] = coerce_numeric(df["ano"]) - df["ano_ingresso"]

    # gap_fase = fase - fase_ideal
    if "fase" in df.columns and "fase_ideal" in df.columns:
        df["fase"] = coerce_numeric(df["fase"])
        df["fase_ideal"] = coerce_numeric(df["fase_ideal"])
        df["gap_fase"] = df["fase"] - df["fase_ideal"]

    # pedra_ord
    pedra_col = _pick_pedra_col(df)
    if pedra_col:
        pedra_norm = clean_string(df[pedra_col]).str.lower()
        pedra_norm = pedra_norm.map(lambda x: _strip_accents(x) if isinstance(x, str) else x)
        df["pedra_ord"] = pedra_norm.map(PEDRA_ORD_MAP).astype("Float64")

    # rec_av_count
    present_masks = []
    for c in REC_AV_COLS:
        if c in df.columns:
            s = clean_string(df[c])
            present_masks.append(s.notna() & (s != ""))
    if present_masks:
        df["rec_av_count"] = pd.concat(present_masks, axis=1).sum(axis=1).astype("int64")

    return df


def coerce_common_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    numeric_like = {
        "fase",
        "defas",
        "n_av",
        "iaa",
        "ian",
        "ida",
        "ieg",
        "ips",
        "ipv",
        "ipp",
        "matem",
        "portug",
        "ingles",
        "ano_nasc",
        "ano_ingresso",
        "idade",  # padronizado
        "tenure",
        "gap_fase",
        "pedra_ord",
    }

    # também sufixos _YY/_YYYY (se existirem nos raw)
    for c in df.columns:
        if re.fullmatch(r"(iaa|ian|ida|ieg|ips|ipv|ipp|matem|portug|ingles)_\d{2,4}", c):
            numeric_like.add(c)

    for c in sorted(numeric_like):
        if c in df.columns:
            df[c] = coerce_numeric(df[c])

    # strings
    for c in ["genero", "turma", "instituicao_de_ensino", "fase_ideal", "rec_psicologia", *REC_AV_COLS, "pedra"]:
        if c in df.columns:
            df[c] = clean_string(df[c])

    if "ra" in df.columns:
        df["ra"] = clean_string(df["ra"])

    return df


# -----------------------------
# Pipeline por ano (raw -> interim)
# -----------------------------

def preprocess_year_df(df: pd.DataFrame, year: int, cfg: Optional[PreprocessConfig] = None) -> pd.DataFrame:
    cfg = cfg or PreprocessConfig()

    df = normalize_columns(df)
    df["ano"] = int(year)

    # Harmonização de schema entre anos (sinônimos)
    rename_map = {
        "data_de_nasc": "ano_nasc",
        "defasagem": "defas",
        "mat": "matem",
        "por": "portug",
        "ing": "ingles",
        "nome_anonimizado": "nome",  # cai na regra de drop
        # Se algum ano veio com IPP em variação estranha, normalize aqui:
        "ip_p": "ipp",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Unifica pedra antes de qualquer coisa
    df = _unify_pedra(df)

    # Remove idade_22/idade_23 etc. e deriva idade padrão
    df = _drop_idade_sufixada(df)
    df = _derive_idade(df, year=year)

    # Garantir schema: se ipp não existir no ano, cria coluna vazia
    if "ipp" not in df.columns:
        df["ipp"] = pd.NA

    # Drop por prefixos (captura variações tipo destaque_ipv_1, avaliador5, inde_23 etc.)
    drop_prefixes_extra = ("destaque_", "avaliador", "inde")
    to_drop = [c for c in df.columns if c.startswith(drop_prefixes_extra)]
    if to_drop:
        df = df.drop(columns=to_drop, errors="ignore")

    # Drop baseline conforme README
    df = drop_baseline_columns(df, keep_ra=cfg.keep_ra)

    # Tipos e limpeza simples
    df = coerce_common_types(df)

    # Features derivadas
    if cfg.create_features:
        df = add_engineered_features(df)

    # Garantir schema no final (depois de drops e renomes)
    if "ipp" not in df.columns:
        df["ipp"] = pd.NA

    return df

def _save_df(df: pd.DataFrame, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.suffix.lower() == ".parquet":
        df.to_parquet(out_path, index=False, engine="pyarrow")
        return out_path

    df.to_csv(out_path, index=False)
    return out_path


def preprocess_year_file(
    in_path: Path,
    year: int,
    out_dir: Path = Path("data/interim"),
    out_format: str = "parquet",
    encoding: str = "utf-8",
    sep: str = ";",
) -> Path:
    df = pd.read_csv(in_path, encoding=encoding, sep=sep, engine="python")
    df_clean = preprocess_year_df(df, year=year, cfg=PreprocessConfig(out_format=out_format))

    out_path = out_dir / f"pede_{year}_interim.{out_format}"
    return _save_df(df_clean, out_path)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Pré-processa CSV do PEDE e salva em data/interim (parquet por padrão).")
    p.add_argument("--input", type=str, required=True, help="Caminho do CSV raw")
    p.add_argument("--year", type=int, required=True, help="Ano de origem (ex.: 2022)")
    p.add_argument("--out-dir", type=str, default="data/interim", help="Diretório de saída (default: data/interim)")
    p.add_argument("--out-format", type=str, default="parquet", choices=["parquet", "csv"], help="Formato de saída")
    p.add_argument("--encoding", type=str, default="utf-8", help="Encoding do CSV (default: utf-8)")
    p.add_argument("--sep", type=str, default=";", help="Separador do CSV (default: ;)")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    out_path = preprocess_year_file(
        in_path=Path(args.input),
        year=int(args.year),
        out_dir=Path(args.out_dir),
        out_format=args.out_format,
        encoding=args.encoding,
        sep=args.sep,
    )
    print(str(out_path))


if __name__ == "__main__":
    main()