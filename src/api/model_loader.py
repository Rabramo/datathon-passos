# src/api/model_loader.py
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Dict

import joblib


@dataclass
class LoadedModel:
    model: Any
    meta: dict


# Map keys to artifacts (adjust to your real files)
MODEL_REGISTRY: Dict[str, str] = {
    "default": "artifacts/models/model_tree_20260303_213447.joblib",
    "tree": "artifacts/models/model_tree_20260303_213447.joblib",
    "logreg": "artifacts/models/model_logreg_20260303_213446.joblib",
}


def resolve_model_key(model_key: Optional[str]) -> str:
    return (model_key or "default").strip().lower()


def resolve_model_path(model_key: Optional[str]) -> Path:

    override = os.getenv("PASSOS_MODEL_PATH")
    if override:
        return Path(override)

    key = resolve_model_key(model_key)
    if key not in MODEL_REGISTRY:
        raise KeyError(f"model_key inválida: {key}. Válidas: {sorted(MODEL_REGISTRY.keys())}")

    return Path(MODEL_REGISTRY[key])


def load_model(*, model_key: Optional[str] = None, return_meta: bool = True) -> LoadedModel:

    path = resolve_model_path(model_key)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo do modelo não encontrado: {path}")

    model = joblib.load(path)

    meta = {}
    if return_meta:
        meta = {
            "model_key": resolve_model_key(model_key),
            "model_path": str(path),
            # "threshold": 0.5,  # if you persist it elsewhere, inject here
        }

    return LoadedModel(model=model, meta=meta)