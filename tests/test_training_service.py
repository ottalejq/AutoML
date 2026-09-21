from app.services.training_service import run_training


def test_run_training_is_callable():
    assert callable(run_training)