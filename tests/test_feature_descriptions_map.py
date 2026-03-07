def test_get_feature_descriptions_map():
    from src.api.feature_descriptions import get_feature_descriptions_map
    m = get_feature_descriptions_map()
    assert isinstance(m, dict)
    assert "fase" in m
    