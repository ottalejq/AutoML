import pandas as pd

from ml.preprocessing import TabularPreprocessor


def test_preprocessor_handles_numeric_and_categorical():
    df = pd.DataFrame({
        "age": [20, 30, 40],
        "country": ["DE", "FR", "DE"],
    })

    preprocessor = TabularPreprocessor()
    preprocessor.fit(df)

    result = preprocessor.transform(df)

    assert result["numeric"].shape == (3, 1)
    assert result["categorical"].shape == (3, 1)