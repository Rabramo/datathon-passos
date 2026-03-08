from types import SimpleNamespace

from fastapi.testclient import TestClient

from src.api.app import app


def test_infra_health_ok():
    client = TestClient(app)
    client.app.state.models_by_key = {}

    r = client.get("/infra/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_infra_model_retorna_404_quando_loader_levanta_keyerror(monkeypatch):
    from src.api.routers import infra as infra_module

    def fake_load_model(model_key=None, return_meta=True):
        raise KeyError("model_key inválida")

    monkeypatch.setattr(infra_module, "load_loaded_model", fake_load_model)

    client = TestClient(app)
    client.app.state.models_by_key = {}

    r = client.get("/infra/model?model_key=inexistente")

    assert r.status_code == 404
    assert "model_key inválida" in r.json()["detail"]


def test_infra_model_retorna_400_quando_loader_levanta_excecao_generica(monkeypatch):
    from src.api.routers import infra as infra_module

    def fake_load_model(model_key=None, return_meta=True):
        raise RuntimeError("falha inesperada")

    monkeypatch.setattr(infra_module, "load_loaded_model", fake_load_model)

    client = TestClient(app)
    client.app.state.models_by_key = {}

    r = client.get("/infra/model?model_key=logreg")

    assert r.status_code == 400
    assert "falha inesperada" in r.json()["detail"]


def test_infra_model_retorna_200_com_loader_mockado(monkeypatch):
    from src.api.routers import infra as infra_module

    class DummyModel:
        feature_names_in_ = ["feature_a", "feature_b"]

    def fake_load_model(model_key=None, return_meta=True):
        return SimpleNamespace(
            model=DummyModel(),
            meta={
                "threshold": 0.55,
                "model_path": "artifacts/models/model_logreg.joblib",
                "raw_features": ["feature_a", "feature_b"],
            },
        )

    monkeypatch.setattr(infra_module, "load_loaded_model", fake_load_model)

    client = TestClient(app)
    client.app.state.models_by_key = {}

    r = client.get("/infra/model?model_key=logreg")

    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_key"] == "logreg"
    assert body["threshold"] == 0.55
    assert body["n_features_esperadas"] == 2


def test_infra_smoke_dry_run_ok():
    client = TestClient(app)
    client.app.state.models_by_key = {}

    r = client.get("/infra/smoke?dry_run=true")

    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["dry_run"]["executado"] is True