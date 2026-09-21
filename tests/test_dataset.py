import numpy as np

from ml.dataset import TabularDataset


def test_tabular_dataset_returns_correct_shapes():
    numeric = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
    ])

    categorical = np.array([
        [1],
        [2],
    ])

    target = np.array([10.0, 20.0])

    dataset = TabularDataset(
        numeric_data=numeric,
        categorical_data=categorical,
        target=target,
    )

    sample = dataset[0]

    assert len(dataset) == 2
    assert sample["numeric"].shape == (2,)
    assert sample["categorical"].shape == (1,)
    assert sample["target"].shape == (1,)