from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routers.predict import router


def create_app() -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    return app


def test_rotas_predict_registradas() -> None:
    app = create_app()
    routes = {(route.path, tuple(route.methods)) for route in app.routes}

    assert any(path == "/predict" and "POST" in methods for path, methods in routes)
    assert any(path == "/predict/batch" and "POST" in methods for path, methods in routes)
    assert any(path == "/predict/model" and "GET" in methods for path, methods in routes)
    assert any(path == "/predict/features/select" and "POST" in methods for path, methods in routes)


def test_openapi_contem_rotas_predict() -> None:
    app = create_app()
    client = TestClient(app)

    response = client.get("/openapi.json")
    assert response.status_code == 200

    data = response.json()
    assert "/predict" in data["paths"]
    assert "/predict/batch" in data["paths"]
    assert "/predict/model" in data["paths"]
    assert "/predict/features/select" in data["paths"]