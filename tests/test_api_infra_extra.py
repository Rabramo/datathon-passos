from fastapi.testclient import TestClient

from src.api.app import app

client = TestClient(app)


def test_health_has_expected_keys():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "uptime_s" in body
    assert "modelo_em_cache" in body


def test_openapi_has_health_and_docs():
    r = client.get("/openapi.json")
    assert r.status_code == 200
    spec = r.json()
    assert "paths" in spec
    assert "/health" in spec["paths"]
    assert "/docs" not in spec["paths"]


def test_smoke_default_contract():
    r = client.get("/smoke")
    assert r.status_code in (200, 400, 404)
    if r.status_code == 200:
        body = r.json()
        assert body["status"] == "ok"
        assert "dry_run" in body
        assert body["dry_run"]["executado"] is False
        assert body["dry_run"]["estrategia"] in ("zeros", "strings_vazias")


def test_smoke_dry_run_contract():
    r = client.get("/smoke?dry_run=true")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["dry_run"]["executado"] is True
    assert body["dry_run"]["estrategia"] in ("zeros", "strings_vazias")


def test_model_default_no_422():
    r = client.get("/model")
    assert r.status_code in (200, 400, 404)


def test_model_tree_no_422():
    r = client.get("/model?model_key=tree")
    assert r.status_code in (200, 400, 404)


def test_model_logreg_no_422():
    r = client.get("/model?model_key=logreg")
    assert r.status_code in (200, 400, 404)


def test_model_unknown_no_422():
    r = client.get("/model?model_key=unknown")
    assert r.status_code in (200, 400, 404)
