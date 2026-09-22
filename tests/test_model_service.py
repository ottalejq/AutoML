from app.services.model_service import save_model_artifact


def test_save_model_artifact_saves_to_model_directory(tmp_path, monkeypatch):
    from unittest.mock import Mock
    from uuid import uuid4

    monkeypatch.setattr("app.services.model_service.MODEL_DIR", tmp_path)
    model = Mock()
    model_id = uuid4()

    path = save_model_artifact(model=model, model_id=model_id)

    expected_path = tmp_path / f"{model_id}.joblib"
    model.save.assert_called_once_with(expected_path)
    assert path == str(expected_path)
