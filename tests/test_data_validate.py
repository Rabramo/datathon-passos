import pandas as pd
import pytest

from src.data.validate import validate_unique_id


def test_validate_unique_id_ok():
    df = pd.DataFrame({"id": [1, 2, 3]})
    validate_unique_id(df, "id")


def test_validate_unique_id_raises_for_duplicates():
    df = pd.DataFrame({"id": [1, 1, 2]})
    with pytest.raises(ValueError):
        validate_unique_id(df, "id")