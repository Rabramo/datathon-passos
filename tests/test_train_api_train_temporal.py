from pathlib import Path

import pandas as pd
import pytest

from src.models import train_api as mod


def test_train_temporal_fluxo_basico_com_mocks(tmp_path, monkeypatch):
    train_path = tmp_path / "train.parquet"
    test_path = tmp_path / "test.parquet"
    art_dir = tmp_path / "artifacts_models"
    met_dir = tmp_path / "artifacts_metrics"

    art_dir.mkdir(parents=True, exist_ok=True)
    met_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "feature_a": [1, 2, 3],
            "feature_b": [10, 20, 30],
            "y": [0, 1, 0],
        }
    ).to_parquet(train_path, index=False)

    pd.DataFrame(
        {
            "feature_a": [4, 5],
            "feature_b": [40, 50],
            "y": [1, 0],
        }
    ).to_parquet(test_path, index=False)

    class DummyModel:
        def fit(self, X, y):
            return self

        def predict_proba(self, X):
            import numpy as np
            return np.array([[0.8, 0.2] for _ in range(len(X))])

    def fake_dump(model, path):
        Path(path).write_bytes(b"fake-joblib-model")
        return path

    monkeypatch.setattr(mod, "DATA_TRAIN", train_path)
    monkeypatch.setattr(mod, "DATA_TEST", test_path)
    monkeypatch.setattr(mod, "ART_DIR", art_dir)
    monkeypatch.setattr(mod, "MET_DIR", met_dir)
    monkeypatch.setattr(mod, "_make_model", lambda internal_key, seed: DummyModel())
    monkeypatch.setattr(mod.joblib, "dump", fake_dump)

    result = mod.train_temporal(
        internal_model_key="logreg",
        features=["feature_a", "feature_b"],
        seed=42,
        threshold=0.5,
    )

    assert result.run_id
    assert result.model_path.endswith(".joblib")
    assert result.metrics_path.endswith(".json")
    assert Path(result.model_path).exists()
    assert Path(result.metrics_path).exists()
    assert result.metrics["internal_model_key"] == "logreg"
    assert result.metrics["threshold"] == 0.5
    assert result.metrics["n_features"] == 2
    assert "test_f1" in result.metrics
    assert "test_precision" in result.metrics
    assert "test_recall" in result.metrics
    assert "confusion_matrix" in result.metrics


def test_train_temporal_dummy_sem_artefato_modelo(tmp_path, monkeypatch):
    train_path = tmp_path / "train.parquet"
    test_path = tmp_path / "test.parquet"
    art_dir = tmp_path / "artifacts_models"
    met_dir = tmp_path / "artifacts_metrics"

    art_dir.mkdir(parents=True, exist_ok=True)
    met_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "feature_a": [1, 2, 3],
            "feature_b": [10, 20, 30],
            "y": [0, 1, 0],
        }
    ).to_parquet(train_path, index=False)

    pd.DataFrame(
        {
            "feature_a": [4, 5],
            "feature_b": [40, 50],
            "y": [1, 0],
        }
    ).to_parquet(test_path, index=False)

    monkeypatch.setattr(mod, "DATA_TRAIN", train_path)
    monkeypatch.setattr(mod, "DATA_TEST", test_path)
    monkeypatch.setattr(mod, "ART_DIR", art_dir)
    monkeypatch.setattr(mod, "MET_DIR", met_dir)

    result = mod.train_temporal(
        internal_model_key="dummy",
        features=["feature_a", "feature_b"],
        seed=42,
        threshold=0.5,
    )

    assert result.run_id
    assert result.model_path == ""
    assert result.metrics_path.endswith(".json")
    assert Path(result.metrics_path).exists()
    assert result.metrics["threshold"] == 0.5
    assert result.metrics["n_features"] == 2
    assert "test_roc_auc" in result.metrics
    assert "confusion_matrix" in result.metrics


def test_train_temporal_levanta_erro_sem_coluna_y(tmp_path, monkeypatch):
    train_path = tmp_path / "train.parquet"
    test_path = tmp_path / "test.parquet"

    pd.DataFrame(
        {
            "feature_a": [1, 2, 3],
            "feature_b": [10, 20, 30],
        }
    ).to_parquet(train_path, index=False)

    pd.DataFrame(
        {
            "feature_a": [4, 5],
            "feature_b": [40, 50],
            "y": [1, 0],
        }
    ).to_parquet(test_path, index=False)

    monkeypatch.setattr(mod, "DATA_TRAIN", train_path)
    monkeypatch.setattr(mod, "DATA_TEST", test_path)

    with pytest.raises(ValueError, match="Coluna alvo 'y' não encontrada"):
        mod.train_temporal(
            internal_model_key="logreg",
            features=["feature_a", "feature_b"],
            seed=42,
            threshold=0.5,
        )