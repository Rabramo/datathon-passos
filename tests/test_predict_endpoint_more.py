from types import SimpleNamespace

from fastapi.testclient import TestClient

from src.api.app import app


class DummyModel:
    feature_names_in_ = ["feature_a", "feature_b"]

    def predict_proba(self, X):
        return [[0.2, 0.8] for _ in range(len(X))]


def test_get_model_info_no_422_with_logreg(monkeypatch):
    from src.api.routers import predict as predict_module

    def fake_get_loaded(request, model_key=None):
        return SimpleNamespace(
            model=DummyModel(),
            meta={"threshold": 0.5, "raw_features": ["feature_a", "feature_b"]},
        )

    monkeypatch.setattr(predict_module, "_obter_modelo_carregado", fake_get_loaded)

    client = TestClient(app)
    r = client.get("/model?model_key=logreg")

    assert r.status_code in (200, 400, 404)

