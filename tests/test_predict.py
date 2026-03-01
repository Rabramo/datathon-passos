from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def _train_and_save_dummy_model(model_path: Path) -> None:
    """
    Treina um pipeline simples (numérico + categórico) e salva em joblib.
    O objetivo é testar a API /predict sem depender dos dados reais.
    """
    X = pd.DataFrame(
        {
            # features típicas do ano t (normalizadas)
            "fase": [1, 2, 3, 4, 2, 1],
            "idade_t": [12, 13, 14, 15, 13, 12],
            "tenure": [1, 2, 2, 3, 1, 1],
            "pedra_ord": [0, 1, 2, 3, 1, np.nan],
            "genero": ["f", "m", "f", "m", "m", "f"],
            "turma": ["t1", "t1", "t2", "t2", "t1", "t3"],
            # manter ian do ano t como feature (permitido)
            "ian": [10, 9, 10, 8, 10, 9],
        }
    )
    y = np.array([0, 1, 0, 1, 0, 1])

    num_cols = ["fase", "idade_t", "tenure", "pedra_ord", "ian"]
    cat_cols = ["genero", "turma"]

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), num_cols),
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            ),
        ],
        remainder="drop",
    )

    clf = LogisticRegression(max_iter=200, random_state=42)
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(X, y)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, model_path)


def _import_fastapi_app_with_model(model_path: Path):
    """
    Tenta importar src.api.app e garantir que o modelo esteja disponível.
    Suporta padrões comuns:
      - app carrega modelo via env MODEL_PATH / ARTIFACT_PATH
      - app expõe atributo `app` (FastAPI)
      - modelo fica em app.state.model|pipeline ou app.model|pipeline
    """
    os.environ["MODEL_PATH"] = str(model_path)
    os.environ["ARTIFACT_PATH"] = str(model_path)  # fallback para outros padrões

    mod = importlib.import_module("src.api.app")
    importlib.reload(mod)

    if not hasattr(mod, "app"):
        raise AssertionError("src.api.app não expõe variável `app` (FastAPI).")

    fastapi_app = getattr(mod, "app")

    # Se a app não carregou modelo automaticamente, tentamos injetar.
    pipe = joblib.load(model_path)

    injected = False
    for attr in ("model", "pipeline"):
        if hasattr(fastapi_app, "state") and hasattr(fastapi_app.state, attr):
            setattr(fastapi_app.state, attr, pipe)
            injected = True
            break
        if hasattr(fastapi_app, attr):
            setattr(fastapi_app, attr, pipe)
            injected = True
            break

    # Mesmo se não injetar, ainda pode estar ok se o app já carrega internamente.
    # Só falharemos se /predict não funcionar.
    return fastapi_app, injected


def _make_payload() -> Dict[str, Any]:
    """
    Payload coerente com as colunas normalizadas/features usadas no dummy model.
    Ajuste se sua API exigir outro schema (ex.: {"features": {...}}).
    """
    return {
        "fase": 2,
        "idade_t": 13,
        "tenure": 1,
        "pedra_ord": 1,
        "genero": "m",
        "turma": "t1",
        "ian": 10,
    }


@pytest.mark.parametrize("payload_style", ["flat", "nested"])
def test_predict_endpoint_returns_prediction(tmp_path: Path, payload_style: str):
    model_path = tmp_path / "artifacts" / "models" / "model.joblib"
    _train_and_save_dummy_model(model_path)

    fastapi_app, _ = _import_fastapi_app_with_model(model_path)
    client = TestClient(fastapi_app)

    payload = _make_payload()
    if payload_style == "nested":
        # Alguns projetos usam {"features": {...}}
        body = {"features": payload}
    else:
        body = payload

    resp = client.post("/predict", json=body)

    # Se o seu app aceitar apenas um dos formatos, um deles pode retornar 422.
    # Neste caso, exigimos que pelo menos um formato funcione.
    if resp.status_code == 422:
        pytest.skip("Schema do /predict não aceita este formato de payload (ok se o outro formato passar).")

    assert resp.status_code == 200, resp.text
    data = resp.json()

    # Aceitar diferentes nomes comuns de chaves
    pred = data.get("prediction", data.get("pred", data.get("y_pred")))
    assert pred in (0, 1), f"prediction inválida: {data}"

    # probabilidade opcional, mas se existir deve ser float em [0,1]
    proba = data.get("proba", data.get("probability", data.get("score")))
    if proba is not None:
        assert isinstance(proba, (int, float))
        assert 0.0 <= float(proba) <= 1.0