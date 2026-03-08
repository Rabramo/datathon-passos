from __future__ import annotations

import json

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.pipelines.evaluate import (
    EvaluationResult,
    align_features,
    evaluate_dataset,
    evaluate_file,
    extract_bundle_parts,
    resolve_model_artifact_path,
    save_evaluation_outputs,
    split_xy,
)


def build_training_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "f1": [0.0, 1.0, 0.0, 1.0, 2.0, 2.0],
            "f2": [0.0, 0.0, 1.0, 1.0, 1.0, 2.0],
            "y": [0, 0, 0, 1, 1, 1],
        }
    )


def test_split_xy_separates_target() -> None:
    df = build_training_frame()

    X, y = split_xy(df, target_column="y")

    assert "y" not in X.columns
    assert list(y.tolist()) == [0, 0, 0, 1, 1, 1]


def test_align_features_selects_expected_columns() -> None:
    X = pd.DataFrame(
        {
            "f1": [1, 2],
            "f2": [3, 4],
            "extra": [5, 6],
        }
    )

    aligned = align_features(X, selected_features=["f2", "f1"])

    assert list(aligned.columns) == ["f2", "f1"]


def test_extract_bundle_parts_supports_raw_model() -> None:
    model = LogisticRegression()

    extracted_model, preprocessor, selected_features, threshold = extract_bundle_parts(
        model
    )

    assert extracted_model is model
    assert preprocessor is None
    assert selected_features is None
    assert threshold == 0.5


def test_evaluate_dataset_with_raw_model() -> None:
    df = build_training_frame()
    X = df.drop(columns=["y"])
    y = df["y"]

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(random_state=42)),
        ]
    )
    model.fit(X, y)

    result, predictions = evaluate_dataset(
        dataset=df,
        artifact=model,
        target_column="y",
    )

    assert result.n_rows == len(df)
    assert 0.0 <= result.accuracy <= 1.0
    assert 0.0 <= result.precision <= 1.0
    assert 0.0 <= result.recall <= 1.0
    assert 0.0 <= result.f1 <= 1.0
    assert result.confusion_matrix and len(result.confusion_matrix) == 2
    assert set(predictions.columns) == {"y_true", "y_score", "y_pred"}
    assert len(predictions) == len(df)


def test_evaluate_dataset_with_bundle_and_threshold_override() -> None:
    df = build_training_frame()
    X = df.drop(columns=["y"])
    y = df["y"]

    model = LogisticRegression(random_state=42)
    model.fit(X[["f1", "f2"]], y)

    artifact = {
        "model": model,
        "preprocessor": None,
        "selected_features": ["f1", "f2"],
        "threshold": 0.9,
    }

    result, predictions = evaluate_dataset(
        dataset=df,
        artifact=artifact,
        target_column="y",
        threshold=0.4,
    )

    assert result.threshold == 0.4
    assert len(predictions) == len(df)


def test_evaluate_file_saves_metrics_and_predictions(tmp_path) -> None:
    df = build_training_frame()
    dataset_path = tmp_path / "eval.csv"
    model_path = tmp_path / "model.joblib"
    output_dir = tmp_path / "artifacts"

    df.to_csv(dataset_path, index=False)

    model = LogisticRegression(random_state=42)
    model.fit(df[["f1", "f2"]], df["y"])

    artifact = {
        "model": model,
        "selected_features": ["f1", "f2"],
        "threshold": 0.5,
    }
    joblib.dump(artifact, model_path)

    result, saved = evaluate_file(
        dataset_path=dataset_path,
        model_path=model_path,
        output_dir=output_dir,
        target_column="y",
        prefix="temporal_2023_2024",
    )

    assert result.n_rows == len(df)
    assert saved.metrics_path.exists()
    assert saved.predictions_path.exists()

    metrics = json.loads(saved.metrics_path.read_text(encoding="utf-8"))
    preds = pd.read_csv(saved.predictions_path)

    assert metrics["n_rows"] == len(df)
    assert "accuracy" in metrics
    assert len(preds) == len(df)


def test_save_evaluation_outputs_writes_files(tmp_path) -> None:
    predictions = pd.DataFrame(
        {
            "y_true": [0, 1],
            "y_score": [0.1, 0.9],
            "y_pred": [0, 1],
        }
    )

    result = EvaluationResult(
        n_rows=2,
        threshold=0.5,
        accuracy=1.0,
        precision=1.0,
        recall=1.0,
        f1=1.0,
        roc_auc=1.0,
        confusion_matrix=[[1, 0], [0, 1]],
    )

    saved = save_evaluation_outputs(
        result=result,
        predictions=predictions,
        output_dir=tmp_path,
        prefix="unit",
    )

    assert saved.metrics_path.exists()
    assert saved.predictions_path.exists()


def test_resolve_model_artifact_path_accepts_direct_joblib(tmp_path) -> None:
    model_path = tmp_path / "model.joblib"
    joblib.dump({"dummy": True}, model_path)

    resolved = resolve_model_artifact_path(model_path)

    assert resolved.resolve() == model_path.resolve()


def test_resolve_model_artifact_path_from_json_pointer(tmp_path) -> None:
    model_path = tmp_path / "model.joblib"
    pointer_path = tmp_path / "latest_logreg.json"

    joblib.dump({"dummy": True}, model_path)
    pointer_path.write_text(
        json.dumps({"model_path": "model.joblib"}),
        encoding="utf-8",
    )

    resolved = resolve_model_artifact_path(pointer_path)

    assert resolved.resolve() == model_path.resolve()

def test_resolve_model_artifact_path_from_json_pointer_with_repo_relative_path(tmp_path, monkeypatch) -> None:
    repo_root = tmp_path
    artifacts_dir = repo_root / "artifacts" / "models"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    model_path = artifacts_dir / "model.joblib"
    pointer_path = artifacts_dir / "latest_logreg.json"

    joblib.dump({"dummy": True}, model_path)
    pointer_path.write_text(
        json.dumps({"model_path": "artifacts/models/model.joblib"}),
        encoding="utf-8",
    )

    monkeypatch.chdir(repo_root)

    resolved = resolve_model_artifact_path(pointer_path)

    assert resolved.resolve() == model_path.resolve()

def test_evaluate_file_accepts_json_pointer(tmp_path) -> None:
    df = pd.DataFrame(
        {
            "f1": [0.0, 1.0, 0.0, 1.0, 2.0, 2.0],
            "f2": [0.0, 0.0, 1.0, 1.0, 1.0, 2.0],
            "y": [0, 0, 0, 1, 1, 1],
        }
    )
    dataset_path = tmp_path / "eval.parquet"
    model_path = tmp_path / "model.joblib"
    pointer_path = tmp_path / "latest_logreg.json"
    output_dir = tmp_path / "artifacts"

    df.to_parquet(dataset_path, index=False)

    model = LogisticRegression(random_state=42)
    model.fit(df[["f1", "f2"]], df["y"])

    artifact = {
        "model": model,
        "selected_features": ["f1", "f2"],
        "threshold": 0.5,
    }
    joblib.dump(artifact, model_path)

    pointer_path.write_text(
        json.dumps({"model_path": "model.joblib"}),
        encoding="utf-8",
    )

    result, saved = evaluate_file(
        dataset_path=dataset_path,
        model_path=pointer_path,
        output_dir=output_dir,
        target_column="y",
        prefix="temporal_pointer",
    )

    assert result.n_rows == len(df)
    assert saved.metrics_path.exists()
    assert saved.predictions_path.exists()