def test_api_main_exports_app():
    from src.api.main import app
    assert app is not None