from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routers.infra import router


def create_app() -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    return app


def test_health_registrado() -> None:
    app = create_app()
    client = TestClient(app)

    response = client.get("/infra/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_openapi_contem_rotas_infra() -> None:
    app = create_app()
    client = TestClient(app)

    response = client.get("/openapi.json")
    assert response.status_code == 200

    data = response.json()
    assert "/infra/health" in data["paths"]
    assert "/infra/smoke" in data["paths"]