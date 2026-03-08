from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


TARGET_COLUMN = "y"

DEFAULT_ID_CANDIDATES = (
    "id",
    "ID",
    "codigo_aluno",
    "codigo",
    "aluno_id",
    "student_id",
    "NOME",
    "nome",
)

DEFAULT_EXACT_EXCLUDE = {
    TARGET_COLUMN,
}

DEFAULT_PATTERN_EXCLUDE = (
    "_t+1",
    "_tp1",
    "t+1",
)


@dataclass(frozen=True)
class FeatureSelectionResult:
    selected_features: list[str]
    numeric_features: list[str]
    categorical_features: list[str]
    dropped_features: list[str]


def _contains_any_pattern(column_name: str, patterns: Iterable[str]) -> bool:
    lowered = column_name.lower()
    return any(pattern.lower() in lowered for pattern in patterns)


def infer_id_columns(df: pd.DataFrame, candidates: Iterable[str] = DEFAULT_ID_CANDIDATES) -> list[str]:
    existing = set(df.columns)
    return [col for col in candidates if col in existing]


def select_feature_columns(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    id_columns: Iterable[str] | None = None,
    exact_exclude: Iterable[str] | None = None,
    pattern_exclude: Iterable[str] | None = None,
) -> FeatureSelectionResult:

    if id_columns is None:
        id_columns = infer_id_columns(df)

    exact_exclude_set = set(exact_exclude or DEFAULT_EXACT_EXCLUDE)
    exact_exclude_set.add(target_column)

    pattern_exclude = tuple(pattern_exclude or DEFAULT_PATTERN_EXCLUDE)
    id_columns_set = set(id_columns)

    selected_features: list[str] = []
    dropped_features: list[str] = []

    for col in df.columns:
        if col in exact_exclude_set:
            dropped_features.append(col)
            continue

        if col in id_columns_set:
            dropped_features.append(col)
            continue

        if _contains_any_pattern(col, pattern_exclude):
            dropped_features.append(col)
            continue

        selected_features.append(col)

    numeric_features = df[selected_features].select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [col for col in selected_features if col not in numeric_features]

    return FeatureSelectionResult(
        selected_features=selected_features,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        dropped_features=sorted(dropped_features),
    )


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


def fit_preprocessor(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    id_columns: Iterable[str] | None = None,
    exact_exclude: Iterable[str] | None = None,
    pattern_exclude: Iterable[str] | None = None,
) -> tuple[ColumnTransformer, FeatureSelectionResult]:
 
    selection = select_feature_columns(
        df=df,
        target_column=target_column,
        id_columns=id_columns,
        exact_exclude=exact_exclude,
        pattern_exclude=pattern_exclude,
    )

    preprocessor = build_preprocessor(
        numeric_features=selection.numeric_features,
        categorical_features=selection.categorical_features,
    )

    preprocessor.fit(df[selection.selected_features])

    return preprocessor, selection


def transform_features(
    df: pd.DataFrame,
    preprocessor: ColumnTransformer,
    selected_features: list[str],
) -> pd.DataFrame:

    missing_features = [col for col in selected_features if col not in df.columns]
    if missing_features:
        raise ValueError(
            f"Input DataFrame is missing required feature columns: {missing_features}"
        )

    transformed = preprocessor.transform(df[selected_features])
    output_columns = preprocessor.get_feature_names_out().tolist()

    return pd.DataFrame(transformed, columns=output_columns, index=df.index)


def fit_transform_features(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    id_columns: Iterable[str] | None = None,
    exact_exclude: Iterable[str] | None = None,
    pattern_exclude: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, ColumnTransformer, FeatureSelectionResult]:
 

    preprocessor, selection = fit_preprocessor(
        df=df,
        target_column=target_column,
        id_columns=id_columns,
        exact_exclude=exact_exclude,
        pattern_exclude=pattern_exclude,
    )

    transformed_df = transform_features(
        df=df,
        preprocessor=preprocessor,
        selected_features=selection.selected_features,
    )

    return transformed_df, preprocessor, selection


def save_preprocessor(preprocessor: ColumnTransformer, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, output_path)


def load_preprocessor(input_path: str | Path) -> ColumnTransformer:
    return joblib.load(Path(input_path))