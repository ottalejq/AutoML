from app.services.model_service import save_model_artifacts


def test_save_model_artifacts_is_callable():
    assert callable(save_model_artifacts)