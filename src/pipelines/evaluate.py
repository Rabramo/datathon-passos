# src.pipelines.evaluate.py
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

DEFAULT_TARGET_COLUMN = "y"
DEFAULT_THRESHOLD = 0.5


@dataclass(frozen=True)
class EvaluationArtifacts:
    metrics_path: Path
    predictions_path: Path


@dataclass(frozen=True)
class EvaluationResult:
    n_rows: int
    threshold: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None
    confusion_matrix: list[list[int]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    suffixes = [suffix.lower() for suffix in path.suffixes]

    if suffixes and suffixes[-1] == ".parquet":
        return pd.read_parquet(path)

    if suffixes[-1:] == [".csv"] or suffixes[-2:] == [".csv", ".gz"]:
        return pd.read_csv(path)

    raise ValueError(
        f"Unsupported dataset format for {path}. Expected .csv, .csv.gz or .parquet."
    )


def _extract_candidate_paths_from_pointer(payload: dict[str, Any]) -> list[str]:
    candidate_keys = (
        "model_path",
        "artifact_path",
        "path",
        "model",
        "latest_model_path",
    )

    candidates: list[str] = []
    for key in candidate_keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            candidates.append(value.strip())

    return candidates


def resolve_model_artifact_path(path: str | Path) -> Path:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}")

    if path.suffix.lower() != ".json":
        return path

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Model pointer JSON must contain an object: {path}")

    candidates = _extract_candidate_paths_from_pointer(payload)
    if not candidates:
        raise ValueError(
            "Model pointer JSON does not contain any supported path key. "
            "Expected one of: model_path, artifact_path, path, model, latest_model_path"
        )

    for candidate in candidates:
        raw_candidate = Path(candidate)

        possible_paths: list[Path] = []

        if raw_candidate.is_absolute():
            possible_paths.append(raw_candidate)
        else:
            possible_paths.append(raw_candidate)
            possible_paths.append(path.parent / raw_candidate)

        for possible_path in possible_paths:
            resolved = possible_path.resolve()
            if resolved.exists():
                return resolved

    raise FileNotFoundError(
        f"Resolved model artifact from pointer does not exist. "
        f"Pointer: {path}. Candidates: {candidates}"
    )

def load_model_artifact(path: str | Path) -> Any:
    resolved_path = resolve_model_artifact_path(path)
    return joblib.load(resolved_path)


def extract_bundle_parts(
    artifact: Any,
) -> tuple[Any, Any | None, list[str] | None, float]:

    if isinstance(artifact, dict):
        model = artifact.get("model")
        preprocessor = artifact.get("preprocessor")
        selected_features = artifact.get("selected_features")
        threshold = float(artifact.get("threshold", DEFAULT_THRESHOLD))

        if model is None:
            raise ValueError("Model bundle is missing required key: 'model'")

        if selected_features is not None and not isinstance(selected_features, list):
            raise TypeError("'selected_features' must be a list when provided")

        return model, preprocessor, selected_features, threshold

    return artifact, None, None, DEFAULT_THRESHOLD


def split_xy(
    df: pd.DataFrame,
    target_column: str = DEFAULT_TARGET_COLUMN,
) -> tuple[pd.DataFrame, pd.Series]:
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset")

    y = df[target_column]
    X = df.drop(columns=[target_column])

    return X, y


def align_features(
    X: pd.DataFrame,
    selected_features: list[str] | None = None,
) -> pd.DataFrame:
    if selected_features is None:
        return X

    missing = [column for column in selected_features if column not in X.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing required selected_features columns: {missing}"
        )

    return X[selected_features].copy()


def transform_features(
    X: pd.DataFrame,
    preprocessor: Any | None = None,
) -> Any:
    if preprocessor is None:
        return X

    return preprocessor.transform(X)


def predict_scores(model: Any, X: Any) -> pd.Series:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        if getattr(probabilities, "ndim", None) != 2 or probabilities.shape[1] < 2:
            raise ValueError("predict_proba output must have shape (n_rows, 2)")
        return pd.Series(probabilities[:, 1])

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return pd.Series(scores)

    raise TypeError(
        "Model must implement either predict_proba or decision_function for evaluation"
    )


def scores_to_labels(scores: pd.Series, threshold: float) -> pd.Series:
    return (scores >= threshold).astype(int)


def compute_metrics(
    y_true: pd.Series,
    y_score: pd.Series,
    y_pred: pd.Series,
    threshold: float,
) -> EvaluationResult:
    y_true = pd.Series(y_true).astype(int)
    y_pred = pd.Series(y_pred).astype(int)

    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    try:
        roc_auc: float | None = float(roc_auc_score(y_true, y_score))
    except ValueError:
        roc_auc = None

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

    return EvaluationResult(
        n_rows=int(len(y_true)),
        threshold=float(threshold),
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        roc_auc=roc_auc,
        confusion_matrix=cm,
    )


def evaluate_dataset(
    dataset: pd.DataFrame,
    artifact: Any,
    target_column: str = DEFAULT_TARGET_COLUMN,
    threshold: float | None = None,
) -> tuple[EvaluationResult, pd.DataFrame]:
    model, preprocessor, selected_features, artifact_threshold = extract_bundle_parts(
        artifact
    )

    effective_threshold = (
        float(artifact_threshold) if threshold is None else float(threshold)
    )

    X, y = split_xy(dataset, target_column=target_column)
    X = align_features(X, selected_features=selected_features)
    X_transformed = transform_features(X, preprocessor=preprocessor)

    y_score = predict_scores(model, X_transformed)
    y_pred = scores_to_labels(y_score, threshold=effective_threshold)

    result = compute_metrics(
        y_true=y,
        y_score=y_score,
        y_pred=y_pred,
        threshold=effective_threshold,
    )

    predictions = pd.DataFrame(
        {
            "y_true": pd.Series(y).astype(int).reset_index(drop=True),
            "y_score": pd.Series(y_score).reset_index(drop=True),
            "y_pred": pd.Series(y_pred).astype(int).reset_index(drop=True),
        }
    )

    return result, predictions


def save_evaluation_outputs(
    result: EvaluationResult,
    predictions: pd.DataFrame,
    output_dir: str | Path,
    prefix: str = "evaluation",
) -> EvaluationArtifacts:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / f"{prefix}_metrics.json"
    predictions_path = output_dir / f"{prefix}_predictions.csv"

    metrics_path.write_text(
        json.dumps(result.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    predictions.to_csv(predictions_path, index=False)

    return EvaluationArtifacts(
        metrics_path=metrics_path,
        predictions_path=predictions_path,
    )


def evaluate_file(
    dataset_path: str | Path,
    model_path: str | Path,
    output_dir: str | Path = Path("artifacts/evaluation"),
    target_column: str = DEFAULT_TARGET_COLUMN,
    threshold: float | None = None,
    prefix: str = "evaluation",
) -> tuple[EvaluationResult, EvaluationArtifacts]:
    dataset = load_table(dataset_path)
    artifact = load_model_artifact(model_path)

    result, predictions = evaluate_dataset(
        dataset=dataset,
        artifact=artifact,
        target_column=target_column,
        threshold=threshold,
    )

    saved = save_evaluation_outputs(
        result=result,
        predictions=predictions,
        output_dir=output_dir,
        prefix=prefix,
    )

    return result, saved


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained binary classifier on a processed dataset."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to processed evaluation dataset (.csv, .csv.gz or .parquet)",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model artifact (.joblib) or pointer JSON (e.g. latest_logreg.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/evaluation",
        help="Directory to save metrics and predictions",
    )
    parser.add_argument(
        "--target-column",
        type=str,
        default=DEFAULT_TARGET_COLUMN,
        help="Target column name",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional override for classification threshold",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="evaluation",
        help="Filename prefix for saved artifacts",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()

    result, saved = evaluate_file(
        dataset_path=args.dataset,
        model_path=args.model,
        output_dir=args.output_dir,
        target_column=args.target_column,
        threshold=args.threshold,
        prefix=args.prefix,
    )

    payload = {
        "metrics_path": str(saved.metrics_path),
        "predictions_path": str(saved.predictions_path),
        "result": result.to_dict(),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()