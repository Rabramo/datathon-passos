# src.api.model_loader.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib


@dataclass(frozen=True)
class LoadedModel:
    model: Any
    meta: dict[str, Any]


CANONICAL_MODEL_KEYS = {
    "default",
    "dummy",
    "logreg",
    "tree",
    "rf",
    "xgb",
    "cat",
}

MODEL_KEY_ALIASES = {
    "default": "default",
    "dummy": "dummy",
    "logreg": "logreg",
    "logistic regression": "logreg",
    "logistic_regression": "logreg",
    "logistic-regression": "logreg",
    "tree": "tree",
    "decision tree": "tree",
    "decision_tree": "tree",
    "decision-tree": "tree",
    "rf": "rf",
    "random forest": "rf",
    "random_forest": "rf",
    "random-forest": "rf",
    "xgb": "xgb",
    "xgboost": "xgb",
    "cat": "cat",
    "catboost": "cat",
}

LATEST_POINTERS = {
    "default": "latest_tree.json",
    "dummy": "latest_dummy.json",
    "logreg": "latest_logreg.json",
    "tree": "latest_tree.json",
    "rf": "latest_rf.json",
    "xgb": "latest_xgb.json",
    "cat": "latest_cat.json",
}

MODELS_DIR = Path("artifacts/models")


def _normalize_model_key(model_key: str | None) -> str:
    if model_key is None:
        return "default"

    normalized = str(model_key).strip().lower()
    normalized = normalized.replace("_", " ")
    normalized = normalized.replace("-", " ")
    normalized = " ".join(normalized.split())

    resolved = MODEL_KEY_ALIASES.get(normalized)
    if resolved is None:
        valid = sorted(CANONICAL_MODEL_KEYS)
        raise KeyError(f'model_key inválida: {model_key}. Válidas: {valid}')

    return resolved


def resolve_model_key(model_key: str | None) -> str:
    return _normalize_model_key(model_key)


def _extract_candidate_paths_from_pointer(payload: dict[str, Any]) -> list[str]:
    keys = (
        "model_path",
        "artifact_path",
        "path",
        "model",
        "latest_model_path",
    )

    candidates: list[str] = []
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            candidates.append(value.strip())

    return candidates


def _resolve_pointer_or_artifact(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")

    if path.suffix.lower() != ".json":
        return path

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Pointer JSON must contain an object: {path}")

    candidates = _extract_candidate_paths_from_pointer(payload)
    if not candidates:
        raise ValueError(
            "Pointer JSON does not contain any supported path key. "
            "Expected one of: model_path, artifact_path, path, model, latest_model_path"
        )

    for candidate in candidates:
        raw = Path(candidate)

        possible_paths: list[Path] = []
        if raw.is_absolute():
            possible_paths.append(raw)
        else:
            possible_paths.append(raw)
            possible_paths.append(path.parent / raw)

        for possible in possible_paths:
            resolved = possible.resolve()
            if resolved.exists():
                return resolved

    raise FileNotFoundError(
        f"Resolved model artifact from pointer does not exist. "
        f"Pointer: {path}. Candidates: {candidates}"
    )


def _resolve_model_artifact_path(model_key: str) -> Path:
    pointer_name = LATEST_POINTERS.get(model_key)
    if pointer_name is None:
        valid = sorted(CANONICAL_MODEL_KEYS)
        raise KeyError(f'model_key inválida: {model_key}. Válidas: {valid}')

    pointer_path = MODELS_DIR / pointer_name
    return _resolve_pointer_or_artifact(pointer_path)


def load_model(model_key: str | None = None, return_meta: bool = False) -> LoadedModel | Any:
    resolved_key = resolve_model_key(model_key)
    artifact_path = _resolve_model_artifact_path(resolved_key)

    loaded = joblib.load(artifact_path)

    if isinstance(loaded, dict):
        model = loaded.get("model", loaded)
        meta = dict(loaded.get("meta", {}))
        meta.setdefault("model_path", str(artifact_path))
        meta.setdefault("resolved_model_key", resolved_key)
    else:
        model = loaded
        meta = {
            "model_path": str(artifact_path),
            "resolved_model_key": resolved_key,
        }

    packaged = LoadedModel(model=model, meta=meta)

    if return_meta:
        return packaged

    return packaged.model
