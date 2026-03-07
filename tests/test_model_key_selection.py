from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)


def test_model_default_ok():
    r = client.get("/model")
    assert r.status_code == 200
    data = r.json()
    assert "model_path" in data
    assert "meta_keys" in data


def test_model_tree_ok_or_same_as_default():
    # If you have model_key support, this should return 200
    r = client.get("/model?model_key=tree")
    assert r.status_code in (200, 400, 404)  # depends on your registry
    if r.status_code == 200:
        data = r.json()
        assert "model_path" in data


def test_model_logreg_exists_or_returns_error():
    # This test is intentionally tolerant: it should not crash the API.
    r = client.get("/model?model_key=logreg")
    assert r.status_code in (200, 400, 404)
    if r.status_code == 200:
        data = r.json()
        assert "model_path" in data
        assert "meta_keys" in data