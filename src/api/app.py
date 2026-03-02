# src/api/app.py
from __future__ import annotations

from fastapi import FastAPI, HTTPException

from src.api.model_loader import LoadedModel, load_model
from src.api.predict import predict_one
from src.api.schemas import PredictRequest, PredictResponse

app = FastAPI(title="Passos Mágicos - Defasagem API")


def _get_loaded() -> LoadedModel:
    """
    Obtém o modelo carregado. Em ambientes de teste, o evento de startup pode não
    ter sido executado antes do primeiro request; então fazemos lazy-load aqui.
    """
    loaded: LoadedModel | None = getattr(app.state, "loaded_model", None)
    if loaded is None:
        loaded = load_model()
        app.state.loaded_model = loaded
    return loaded


@app.on_event("startup")
def _startup() -> None:
    # Preload em runtime normal; nos testes, _get_loaded() cobre o caso.
    app.state.loaded_model = load_model()


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    try:
        loaded = _get_loaded()
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

    pred, proba = predict_one(loaded, req)
    return PredictResponse(
        prediction=pred,
        proba=proba,
        threshold=float(loaded.threshold),
        model_path=loaded.model_path,
    )

@app.get("/health")
def health() -> dict[str, str]:
    loaded = getattr(app.state, "loaded_model", None)
    if loaded is None:
        return {"status": "ok", "model_loaded": "false"}
    return {
        "status": "ok",
        "model_loaded": "true",
        "model_path": loaded.model_path,
        "run_id": str(loaded.run_id),
    }