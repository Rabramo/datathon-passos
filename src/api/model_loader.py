#srf/api/model_loader.py

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

import joblib


@dataclass(frozen=True)
class LoadedModel:
    pipeline: Any
    feature_cols: List[str]
    threshold: float
    model_path: str
    run_id: Optional[str]


def find_latest_model(model_dir: Path) -> Path:
    candidates = sorted(model_dir.glob("model_logreg_*.joblib"))
    if not candidates:
        raise FileNotFoundError(f"Nenhum modelo encontrado em {model_dir}")
    return candidates[-1]


def parse_run_id_from_model_path(p: Path) -> Optional[str]:
    # model_logreg_YYYYMMDD_HHMMSS.joblib
    m = re.match(r"model_logreg_(\d{8}_\d{6})\.joblib$", p.name)
    return m.group(1) if m else None


def extract_feature_columns(pipeline: Any) -> List[str]:
    """
    Extrai colunas esperadas pelo ColumnTransformer no step 'preprocess'.

    Compatibilidade:
    - Se o pipeline NÃO tiver step 'preprocess' (ex.: modelo dummy nos testes),
      retorna [] e o predict faz fallback para DataFrame(payload) sem alinhamento.
    """
    if not hasattr(pipeline, "named_steps") or "preprocess" not in getattr(pipeline, "named_steps", {}):
        return []

    pre = pipeline.named_steps["preprocess"]
    cols: List[str] = []
    for _, _, col_spec in getattr(pre, "transformers", []):
        if col_spec is None:
            continue
        if isinstance(col_spec, (list, tuple)):
            cols.extend(list(col_spec))

    seen = set()
    out: List[str] = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def load_threshold_for_run(model_dir: Path, run_id: Optional[str]) -> Optional[float]:
    """
    Busca threshold no arquivo metrics_<run_id>.json.
    Retorna None se não encontrar ou se run_id for None.
    """
    if not run_id:
        return None
    metrics_path = model_dir / f"metrics_{run_id}.json"
    if not metrics_path.exists():
        return None

    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    thr = data.get("threshold")
    if thr is None:
        return None
    return float(thr)


def load_model(
    *,
    model_dir: Path = Path("artifacts/models"),
    model_path_env: str = "MODEL_PATH",
    artifact_path_env: str = "ARTIFACT_PATH",
    default_threshold: float = 0.5,
) -> LoadedModel:
    """
    Carrega modelo.
    Prioridade:
    1) env MODEL_PATH
    2) env ARTIFACT_PATH
    3) latest em artifacts/models

    Threshold:
    - tenta ler metrics_<run_id>.json
    - fallback para default_threshold
    """
    env_path = os.getenv(model_path_env) or os.getenv(artifact_path_env)
    if env_path:
        p = Path(env_path)
    else:
        p = find_latest_model(model_dir)

    if not p.exists():
        raise FileNotFoundError(f"Modelo não encontrado em: {p}")

    pipeline = joblib.load(p)
    feature_cols = extract_feature_columns(pipeline)
    run_id = parse_run_id_from_model_path(p)

    thr = load_threshold_for_run(model_dir, run_id)
    threshold = float(default_threshold if thr is None else thr)

    return LoadedModel(
        pipeline=pipeline,
        feature_cols=feature_cols,
        threshold=threshold,
        model_path=str(p),
        run_id=run_id,
    )