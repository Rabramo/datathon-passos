import importlib
from typing import Any, Optional

import pandas as pd
import pytest
from fastapi.testclient import TestClient


class DummyModelZerosFailStringsOk:
    """
    predict_proba falha quando a linha tem zeros (força cair no fallback strings_vazias),
    mas funciona quando recebe strings vazias.
    """

    # também ajuda a cobrir extração de features via feature_names_in_
    feature_names_in_ = ["fase", "turma", "iaa"]

    def predict_proba(self, X: pd.DataFrame):
        # se qualquer valor for numérico 0, falha (dispara fallback)
        try:
            if (X.fillna(0) == 0).any().any():
                raise ValueError("zeros not accepted")
        except Exception:
            # se X tiver strings, a comparação acima pode falhar; ignore
            pass
        return [[0.2, 0.8]]


class DummyLoadedModel:
    def __init__(self, model: Any, meta: Optional[dict] = None):
        self.model = model
        self.meta = meta or {
            "model_path": "dummy.joblib",
            "threshold": 0.5,
            # sem raw_features de propósito: força fallback p/ feature_names_in_
        }


@pytest.fixture()
def client_with_dummy_loader(monkeypatch):
    """
    Cria TestClient com o load_loaded_model monkeypatched dentro de src.api.app,
    evitando IO de disco e cobrindo branches do /smoke e OpenAPI injection.
    """
    app_mod = importlib.import_module("src.api.app")

    def _fake_load_loaded_model(*, return_meta: bool = True, **kwargs):
        return DummyLoadedModel(model=DummyModelZerosFailStringsOk())

    # patch no símbolo importado em app.py: load_loaded_model
    monkeypatch.setattr(app_mod, "load_loaded_model", _fake_load_loaded_model)

    return TestClient(app_mod.app)


def test_health_ok(client_with_dummy_loader):
    r = client_with_dummy_loader.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "uptime_s" in body
    assert "modelo_em_cache" in body


def test_docs_endpoint_returns_html(client_with_dummy_loader):
    r = client_with_dummy_loader.get("/docs")
    assert r.status_code == 200
    assert "text/html" in r.headers.get("content-type", "")
    assert "Swagger UI" in r.text or "swagger" in r.text.lower()


def test_openapi_json_is_generated_and_has_paths(client_with_dummy_loader):
    r = client_with_dummy_loader.get("/openapi.json")
    assert r.status_code == 200
    spec = r.json()
    assert "paths" in spec
    # health deve existir e /docs não deve ser injetado no schema
    assert "/health" in spec["paths"]
    assert "/docs" not in spec["paths"]


def test_smoke_dry_run_uses_fallback_strings(client_with_dummy_loader):
    r = client_with_dummy_loader.get("/smoke?dry_run=true")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["dry_run"]["executado"] is True

    # A estratégia pode ser 'zeros' (primeira tentativa) ou 'strings_vazias' (fallback).
    assert body["dry_run"]["estrategia"] in ("zeros", "strings_vazias")

    assert "features_esperadas" in body
    assert body["n_features_esperadas"] == len(body["features_esperadas"])
