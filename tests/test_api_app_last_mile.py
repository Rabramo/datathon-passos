from types import SimpleNamespace

from fastapi.testclient import TestClient

from src.api.app import app, _extrair_features_esperadas


def test_infra_smoke_retorna_threshold_default_quando_threshold_invalido(monkeypatch):
    from src.api.routers import infra as infra_module

    class DummyModel:
        feature_names_in_ = ["feature_a", "feature_b"]

    def fake_load_model(model_key=None, return_meta=True):
        return SimpleNamespace(
            model=DummyModel(),
            meta={
                "threshold": "invalido",
                "model_path": "artifacts/models/model.joblib",
                "raw_features": ["feature_a", "feature_b"],
            },
        )

    monkeypatch.setattr(infra_module, "load_loaded_model", fake_load_model)

    client = TestClient(app)
    client.app.state.models_by_key = {}

    r = client.get("/infra/smoke?model_key=logreg")

    assert r.status_code == 200
    body = r.json()
    assert body["threshold"] == 0.5
    assert body["n_features_esperadas"] == 2
    assert body["dry_run"]["executado"] is False

def test_openapi_reutiliza_schema_em_cache():
    schema_1 = app.openapi()
    schema_2 = app.openapi()

    assert schema_1 is schema_2
    assert "/health" in schema_2["paths"]
    assert "/docs" not in schema_2["paths"]

def test_smoke_com_meta_none_retorna_threshold_default():
    client = TestClient(app)

    class DummyModel:
        feature_names_in_ = ["feature_a"]

    client.app.state.models_by_key = {
        "default": SimpleNamespace(
            model=DummyModel(),
            meta=None,
        )
    }

    r = client.get("/smoke")

    assert r.status_code == 200
    body = r.json()
    assert body["threshold"] == 0.5
    assert body["n_features_esperadas"] == 1

def test_extrair_features_esperadas_prefere_meta_raw_features():
    model = SimpleNamespace()
    meta = {"raw_features": ["feature_a", "feature_b"]}

    cols = _extrair_features_esperadas(model, meta)

    assert cols == ["feature_a", "feature_b"]


def test_extrair_features_esperadas_faz_fallback_para_feature_names_in():
    model = SimpleNamespace(feature_names_in_=["feature_x", "feature_y"])
    meta = {}

    cols = _extrair_features_esperadas(model, meta)

    assert cols == ["feature_x", "feature_y"]

def test_extrair_features_esperadas_retorna_lista_vazia_quando_nada_disponivel():
    model = SimpleNamespace()
    meta = {}

    cols = _extrair_features_esperadas(model, meta)

    assert cols == []
