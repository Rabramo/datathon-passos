from src.api.schemas import ModelKey


def test_model_key_enum_values() -> None:
    assert ModelKey.dummy.value == "dummy"
    assert ModelKey.logreg.value == "logreg"
    assert ModelKey.tree.value == "tree"
    assert ModelKey.rf.value == "rf"
    assert ModelKey.xgb.value == "xgb"
    assert ModelKey.cat.value == "cat"