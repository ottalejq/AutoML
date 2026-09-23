import numpy as np
import pandas as pd
import pytest

from ml.preprocessing import TabularPreprocessor
from ml.types import PreprocessingStrategy


@pytest.mark.parametrize("strategy", list(PreprocessingStrategy))
def test_preprocessor_handles_numeric_and_categorical(strategy):
    df = pd.DataFrame({
        "age": [20, 30, 40],
        "country": ["DE", "FR", "DE"],
    })

    preprocessor = TabularPreprocessor(strategy)
    preprocessor.fit(df)

    result = preprocessor.transform(df)

    if strategy == PreprocessingStrategy.ONE_HOT_SCALED:
        assert result.shape == (3, 3)
        np.testing.assert_allclose(result[:, 0].mean(), 0.0, atol=1e-7)
        np.testing.assert_array_equal(result[:, 1:], [[1, 0], [0, 1], [1, 0]])
        assert preprocessor.get_model_params() == {}
    elif strategy == PreprocessingStrategy.NATIVE_CATEGORICAL:
        assert result.shape == (3, 2)
        assert isinstance(result["country"].dtype, pd.CategoricalDtype)
        assert result["country"].tolist() == ["DE", "FR", "DE"]
        assert result["age"].tolist() == [20, 30, 40]
        assert preprocessor.get_model_params() == {}
    else:
        numeric, categorical = result
        assert numeric.shape == (3, 1)
        assert categorical.shape == (3, 1)
        assert numeric.dtype == np.float32
        assert categorical.dtype == np.int64
        np.testing.assert_array_equal(categorical[:, 0], [1, 2, 1])
        assert preprocessor.get_model_params() == {
            "num_numeric": 1,
            "cat_cardinalities": [3],
        }
