from __future__ import annotations

import pandas as pd

from src.features.preprocess import (
    fit_preprocessor,
    load_preprocessor,
    save_preprocessor,
    transform_features,
)


def test_save_and_load_preprocessor(tmp_path) -> None:
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "idade_t": [10.0, None, 12.0],
            "bolsa_t": ["SIM", None, "NAO"],
            "y": [0, 1, 0],
        }
    )

    preprocessor, selection = fit_preprocessor(df)

    output_path = tmp_path / "preprocessor.joblib"
    save_preprocessor(preprocessor, output_path)

    loaded = load_preprocessor(output_path)

    transformed = transform_features(df, loaded, selection.selected_features)

    assert output_path.exists()
    assert transformed.shape[0] == len(df)
    assert transformed.isnull().sum().sum() == 0