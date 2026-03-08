from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routers.train import router


def create_app() -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    return app


def test_train_route_is_registered() -> None:
    app = create_app()
    routes = {(route.path, tuple(route.methods)) for route in app.routes}
    assert any(path == "/train" and "POST" in methods for path, methods in routes)


def test_openapi_contains_train_path() -> None:
    app = create_app()
    client = TestClient(app)

    response = client.get("/openapi.json")
    assert response.status_code == 200

    data = response.json()
    assert "/train" in data["paths"]
    assert "post" in data["paths"]["/train"]