import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class TabularPreprocessor:
    def __init__(self):
        self.numeric_columns = []
        self.categorical_columns = []

        self.scaler = StandardScaler()
        self.medians = None
        self.category_maps = {}

    def fit(self, X: pd.DataFrame):
        self.numeric_columns = (
            X.select_dtypes(include="number")
            .columns
            .tolist()
        )

        self.categorical_columns = (
            X.select_dtypes(exclude="number")
            .columns
            .tolist()
        )

        if self.numeric_columns:
            numeric = X[self.numeric_columns].copy()

            self.medians = numeric.median()

            numeric = numeric.fillna(self.medians)

            self.scaler.fit(numeric)

        for column in self.categorical_columns:
            values = (
                X[column]
                .fillna("__MISSING__")
                .astype(str)
            )

            categories = sorted(values.unique())

            self.category_maps[column] = {
                category: index + 1
                for index, category in enumerate(categories)
            }

        return self

    def transform(self, X: pd.DataFrame):
        if self.numeric_columns:
            numeric = (
                X[self.numeric_columns]
                .copy()
                .fillna(self.medians)
            )

            numeric_data = self.scaler.transform(numeric)

        else:
            numeric_data = np.empty(
                (len(X), 0),
                dtype=np.float32,
            )

        categorical_arrays = []

        for column in self.categorical_columns:
            values = (
                X[column]
                .fillna("__MISSING__")
                .astype(str)
            )

            mapping = self.category_maps[column]

            encoded = (
                values
                .map(mapping)
                .fillna(0)
                .astype(int)
                .to_numpy()
            )

            categorical_arrays.append(encoded)

        if categorical_arrays:
            categorical_data = np.column_stack(
                categorical_arrays
            )
        else:
            categorical_data = np.empty(
                (len(X), 0),
                dtype=np.int64,
            )

        return {
            "numeric": numeric_data.astype(np.float32),
            "categorical": categorical_data.astype(np.int64),
        }