from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routers.leaderboard import router


def create_app() -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    return app


def test_rota_leaderboard_registrada() -> None:
    app = create_app()
    routes = {(route.path, tuple(route.methods)) for route in app.routes}

    assert any(path == "/leaderboard" and "GET" in methods for path, methods in routes)


def test_openapi_contem_rota_leaderboard() -> None:
    app = create_app()
    client = TestClient(app)

    response = client.get("/openapi.json")
    assert response.status_code == 200

    data = response.json()
    assert "/leaderboard" in data["paths"]