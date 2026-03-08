from types import SimpleNamespace

from fastapi.testclient import TestClient

from src.api.app import app


def test_smoke_sem_cache_cria_cache(monkeypatch):
    from src.api import app as app_module

    class DummyModel:
        feature_names_in_ = ["feature_a", "feature_b"]

    def fake_load_model(model_key=None, return_meta=True):
        return SimpleNamespace(
            model=DummyModel(),
            meta={"threshold": 0.5, "raw_features": ["feature_a", "feature_b"]},
        )

    monkeypatch.setattr(app_module, "load_loaded_model", fake_load_model)

    client = TestClient(app)
    r = client.get("/smoke")

    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "dry_run" in body
    assert body["dry_run"]["executado"] is False