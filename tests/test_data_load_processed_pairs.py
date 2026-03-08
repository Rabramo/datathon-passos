from pathlib import Path

import pandas as pd
import pytest

from src.data.load_processed_pairs import PairPaths, _split_X_y, load_train_test_from_processed


def test_split_X_y_separates_target_and_drops_default_columns():
    df = pd.DataFrame(
        {
            "ra": [1, 2, 3],
            "year_t": [2022, 2022, 2022],
            "year_t1": [2023, 2023, 2023],
            "ano": [2022, 2022, 2022],
            "feature_a": [1, 2, 3],
            "feature_b": [10, 20, 30],
            "y": [0, 1, 0],
        }
    )

    X, y = _split_X_y(df)

    assert list(X.columns) == ["feature_a", "feature_b"]
    assert list(y.tolist()) == [0, 1, 0]
    assert y.dtype.kind in ("i", "u")


def test_split_X_y_respects_custom_columns():
    df = pd.DataFrame(
        {
            "student_id": [11, 12],
            "target": [1, 0],
            "drop_me": ["x", "y"],
            "feature_a": [5, 6],
        }
    )

    X, y = _split_X_y(
        df,
        id_col="student_id",
        target_col="target",
        drop_cols=("drop_me",),
    )

    assert list(X.columns) == ["feature_a"]
    assert list(y.tolist()) == [1, 0]


def test_load_train_test_from_processed_reads_parquets(tmp_path: Path):
    train_path = tmp_path / "pair_train.parquet"
    test_path = tmp_path / "pair_test.parquet"

    pd.DataFrame(
        {
            "ra": [1, 2],
            "year_t": [2022, 2022],
            "year_t1": [2023, 2023],
            "feature_a": [10, 20],
            "feature_b": [100, 200],
            "y": [0, 1],
        }
    ).to_parquet(train_path, index=False)

    pd.DataFrame(
        {
            "ra": [3],
            "year_t": [2023],
            "year_t1": [2024],
            "feature_a": [30],
            "feature_b": [300],
            "y": [1],
        }
    ).to_parquet(test_path, index=False)

    paths = PairPaths(
        train_pair_path=train_path,
        test_pair_path=test_path,
    )

    X_train, y_train, X_test, y_test = load_train_test_from_processed(paths)

    assert list(X_train.columns) == ["feature_a", "feature_b"]
    assert list(X_test.columns) == ["feature_a", "feature_b"]
    assert list(y_train.tolist()) == [0, 1]
    assert list(y_test.tolist()) == [1]


def test_load_train_test_from_processed_raises_when_train_missing(tmp_path: Path):
    test_path = tmp_path / "pair_test.parquet"
    pd.DataFrame({"ra": [1], "y": [1]}).to_parquet(test_path, index=False)

    paths = PairPaths(
        train_pair_path=tmp_path / "missing_train.parquet",
        test_pair_path=test_path,
    )

    with pytest.raises(FileNotFoundError, match="Arquivo não encontrado"):
        load_train_test_from_processed(paths)


def test_load_train_test_from_processed_raises_when_test_missing(tmp_path: Path):
    train_path = tmp_path / "pair_train.parquet"
    pd.DataFrame({"ra": [1], "y": [1]}).to_parquet(train_path, index=False)

    paths = PairPaths(
        train_pair_path=train_path,
        test_pair_path=tmp_path / "missing_test.parquet",
    )

    with pytest.raises(FileNotFoundError, match="Arquivo não encontrado"):
        load_train_test_from_processed(paths)