from __future__ import annotations

import pandas as pd

from src.features.preprocess import (
    fit_preprocessor,
    fit_transform_features,
    infer_id_columns,
    select_feature_columns,
    transform_features,
)


def test_infer_id_columns_detects_known_id_columns() -> None:
    df = pd.DataFrame(
        {
            "id": [1, 2],
            "idade_t": [10, 11],
            "y": [0, 1],
        }
    )

    result = infer_id_columns(df)

    assert result == ["id"]


def test_select_feature_columns_removes_target_id_and_t_plus_1_columns() -> None:
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "idade_t": [10, 11, 12],
            "fase_t": ["A", "B", "A"],
            "nota_t+1": [7.0, 8.0, 9.0],
            "indicador_t+1": ["x", "y", "z"],
            "y": [0, 1, 0],
        }
    )

    selection = select_feature_columns(df)

    assert selection.selected_features == ["idade_t", "fase_t"]
    assert selection.numeric_features == ["idade_t"]
    assert selection.categorical_features == ["fase_t"]
    assert "id" in selection.dropped_features
    assert "y" in selection.dropped_features
    assert "nota_t+1" in selection.dropped_features
    assert "indicador_t+1" in selection.dropped_features


def test_fit_transform_features_returns_dataframe_without_nulls() -> None:
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "idade_t": [10.0, None, 12.0],
            "bolsa_t": ["SIM", None, "NAO"],
            "y": [0, 1, 0],
        }
    )

    transformed_df, preprocessor, selection = fit_transform_features(df)

    assert transformed_df.shape[0] == 3
    assert transformed_df.isnull().sum().sum() == 0
    assert selection.selected_features == ["idade_t", "bolsa_t"]

    transformed_again = transform_features(df, preprocessor, selection.selected_features)
    assert list(transformed_again.columns) == list(transformed_df.columns)


def test_transform_features_raises_for_missing_required_columns() -> None:
    train_df = pd.DataFrame(
        {
            "id": [1, 2],
            "idade_t": [10, 11],
            "bolsa_t": ["SIM", "NAO"],
            "y": [0, 1],
        }
    )

    test_df = pd.DataFrame(
        {
            "id": [3],
            "idade_t": [12],
        }
    )

    preprocessor, selection = fit_preprocessor(train_df)

    try:
        transform_features(test_df, preprocessor, selection.selected_features)
        assert False, "Expected ValueError for missing required columns"
    except ValueError as exc:
        assert "missing required feature columns" in str(exc)