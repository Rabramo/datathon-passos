from fastapi.testclient import TestClient

from src.api.app import app

from src.api.schemas import TrainResponse


def _payload_train():
    return {
        "model_key": "Logistic Regression",
        "feature_set": "custom",
        "variables": ["idade", "ipp"],
        "threshold": 0.5,
        "random_seed": 42,
    }


def test_train_endpoint_retorna_200_quando_train_temporal_retorna_dict(monkeypatch):
    from src.api.routers import train as train_module

    monkeypatch.setattr(
        train_module,
        "train_temporal",
        lambda payload: {
            "status": "ok",
            "run_id": "20260101_000000",
            "model_key": "Logistic Regression",
            "internal_model_key": "logreg",
            "n_features": 2,
            "features_used": ["idade", "ipp"],
            "metrics": {
                "test_f1": 0.5,
                "test_precision": 0.5,
                "test_recall": 0.5,
                "test_roc_auc": 0.5,
                "confusion_matrix": [[1, 0], [0, 1]],
            },
            "artifacts": {
                "model_path": "artifacts/models/model_logreg_20260101_000000.joblib",
                "metrics_path": "artifacts/metrics/metrics_logreg_20260101_000000.json",
            },
        },
    )

    client = TestClient(app)
    r = client.post("/train", json=_payload_train())

    assert r.status_code == 200
    body = r.json()
    assert body["run_id"] == "20260101_000000"
    assert body["internal_model_key"] == "logreg"
    assert body["n_features"] == 2
    assert "metrics" in body
    assert "artifacts" in body


def test_train_endpoint_retorna_400_quando_train_temporal_levanta_value_error(monkeypatch):
    from src.api.routers import train as train_module

    def fake_train(payload):
        raise ValueError("payload inválido")

    monkeypatch.setattr(train_module, "train_temporal", fake_train)

    client = TestClient(app)
    r = client.post("/train", json=_payload_train())

    assert r.status_code == 400
    assert "payload inválido" in r.json()["detail"]


def test_train_endpoint_retorna_404_quando_train_temporal_levanta_file_not_found(monkeypatch):
    from src.api.routers import train as train_module

    def fake_train(payload):
        raise FileNotFoundError("dataset não encontrado")

    monkeypatch.setattr(train_module, "train_temporal", fake_train)

    client = TestClient(app)
    r = client.post("/train", json=_payload_train())

    assert r.status_code == 404
    assert "dataset não encontrado" in r.json()["detail"]


def test_train_endpoint_retorna_500_quando_train_temporal_retorna_tipo_invalido(monkeypatch):
    from src.api.routers import train as train_module

    monkeypatch.setattr(train_module, "train_temporal", lambda payload: 123)

    client = TestClient(app)
    r = client.post("/train", json=_payload_train())

    assert r.status_code == 500
    assert "Falha no treinamento" in r.json()["detail"]


def test_train_endpoint_retorna_500_quando_train_temporal_levanta_excecao_generica(monkeypatch):
    from src.api.routers import train as train_module

    def fake_train(payload):
        raise RuntimeError("erro inesperado")

    monkeypatch.setattr(train_module, "train_temporal", fake_train)

    client = TestClient(app)
    r = client.post("/train", json=_payload_train())

    assert r.status_code == 500
    assert "Falha no treinamento" in r.json()["detail"]

def test_train_endpoint_retorna_200_quando_train_temporal_retorna_trainresponse(monkeypatch):
    from src.api.routers import train as train_module

    monkeypatch.setattr(
        train_module,
        "train_temporal",
        lambda payload: TrainResponse(
            status="ok",
            run_id="20260101_000001",
            model_key="Logistic Regression",
            internal_model_key="logreg",
            n_features=2,
            features_used=["idade", "ipp"],
            metrics={
                "test_f1": 0.5,
                "test_precision": 0.5,
                "test_recall": 0.5,
                "test_roc_auc": 0.5,
                "confusion_matrix": [[1, 0], [0, 1]],
            },
            artifacts={
                "model_path": "artifacts/models/model_logreg_20260101_000001.joblib",
                "metrics_path": "artifacts/metrics/metrics_logreg_20260101_000001.json",
            },
        ),
    )


def test_train_endpoint_retorna_200_quando_train_temporal_retorna_trainresponse(monkeypatch):
    from src.api.routers import train as train_module

    monkeypatch.setattr(
        train_module,
        "train_temporal",
        lambda payload: TrainResponse(
            status="ok",
            run_id="20260101_000001",
            model_key="Logistic Regression",
            internal_model_key="logreg",
            n_features=2,
            features_used=["idade", "ipp"],
            metrics={
                "test_f1": 0.5,
                "test_precision": 0.5,
                "test_recall": 0.5,
                "test_roc_auc": 0.5,
                "confusion_matrix": [[1, 0], [0, 1]],
            },
            artifacts={
                "model_path": "artifacts/models/model_logreg_20260101_000001.joblib",
                "metrics_path": "artifacts/metrics/metrics_logreg_20260101_000001.json",
            },
        ),
    )

    client = TestClient(app)
    r = client.post("/train", json=_payload_train())

    assert r.status_code == 200
    assert r.json()["run_id"] == "20260101_000001"