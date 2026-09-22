import numpy as np

from ml.config import METRICS


def test_evaluator_returns_metrics():
    target = np.array([2.0, 4.0, 6.0])
    predictions = np.array([3.0, 4.0, 5.0])

    metrics = {
        name: metric(target, predictions)
        for name, metric in METRICS.items()
    }

    assert set(metrics) == {"rmse", "mae", "r2"}
    assert np.isclose(metrics["rmse"], np.sqrt(2.0 / 3.0))
    assert np.isclose(metrics["mae"], 2.0 / 3.0)
    assert np.isclose(metrics["r2"], 0.75)
