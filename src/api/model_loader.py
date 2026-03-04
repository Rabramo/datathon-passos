# src/api/model_loader.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_model_dir(model_dir: str | Path) -> Path:
    p = Path(model_dir)
    return p if p.is_absolute() else Path.cwd() / p


def _latest_pointer_path(model_dir: Path, model_name: str) -> Path:
    # default pointer per model
    return model_dir / f"latest_{model_name}.json"


def _legacy_latest_path(model_dir: Path) -> Path:
    # legacy pointer
    return model_dir / "latest.json"


def _find_latest_joblib(model_dir: Path) -> Path:
    """
    Fallback: find newest .joblib inside model_dir.
    Prefer files that look like model artifacts: model_*.joblib
    """
    candidates = sorted(model_dir.glob("model_*.joblib"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        candidates = sorted(model_dir.glob("*.joblib"), key=lambda p: p.stat().st_mtime, reverse=True)

    if not candidates:
        raise FileNotFoundError(f"Nenhum modelo encontrado em {model_dir}")

    return candidates[0]


def _load_from_pointer(pointer_path: Path) -> Tuple[Path, Dict[str, Any]]:
    meta = _read_json(pointer_path)
    if "model_path" not in meta:
        raise KeyError(f"Arquivo {pointer_path} não contém a chave obrigatória 'model_path'")

    model_path = Path(meta["model_path"])
    # allow relative paths in JSON
    if not model_path.is_absolute():
        model_path = Path.cwd() / model_path

    if not model_path.exists():
        raise FileNotFoundError(f"Model path do ponteiro não existe: {model_path}")

    return model_path, meta


def load_model(
    model_dir: str | Path = "artifacts/models",
    *,
    model_name: str = "tree",
    return_meta: bool = False,
):
    """
    Loads a trained model artifact.

    Default behavior:
      1) artifacts/models/latest_tree.json
      2) artifacts/models/latest.json
      3) newest model_*.joblib (fallback)

    Parameters
    ----------
    model_dir:
        Directory containing model artifacts and pointer files.
    model_name:
        Which pointer to prefer (default: tree).
    return_meta:
        If True, returns (model, meta). If False, returns model only (backwards compatible).
    """
    model_dir_path = _resolve_model_dir(model_dir)

    # 1) Prefer latest_<model>.json
    ptr = _latest_pointer_path(model_dir_path, model_name)
    if ptr.exists():
        model_path, meta = _load_from_pointer(ptr)
        model = joblib.load(model_path)
        return (model, meta) if return_meta else model

    # 2) Fallback: latest.json
    legacy = _legacy_latest_path(model_dir_path)
    if legacy.exists():
        model_path, meta = _load_from_pointer(legacy)
        model = joblib.load(model_path)
        return (model, meta) if return_meta else model

    # 3) Last resort: newest joblib
    model_path = _find_latest_joblib(model_dir_path)
    model = joblib.load(model_path)
    meta = {"model_path": str(model_path), "model_name": None, "run_id": None, "source": "fallback_newest_joblib"}
    return (model, meta) if return_meta else model