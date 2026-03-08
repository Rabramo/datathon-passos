from fastapi.testclient import TestClient

from src.api.main import app


def test_openapi_exposes_model_key_enum() -> None:
    client = TestClient(app)
    response = client.get("/openapi.json")
    assert response.status_code == 200

    payload = response.json()
    payload_str = str(payload)

    assert "dummy" in payload_str
    assert "logreg" in payload_str
    assert "tree" in payload_str
    assert "rf" in payload_str
    assert "xgb" in payload_str
    assert "cat" in payload_str