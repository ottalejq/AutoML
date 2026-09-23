import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.types import PreprocessingStrategy


class TabularPreprocessor:
    def __init__(self, strategy: PreprocessingStrategy):
        self.strategy = strategy

    def fit(self, X):
        self.num_cols = X.select_dtypes(include=np.number).columns.tolist()

        self.cat_cols = X.columns.difference(self.num_cols).tolist()

        self.medians = X[self.num_cols].median()

        if self.strategy in {
            PreprocessingStrategy.ONE_HOT_SCALED,
            PreprocessingStrategy.EMBEDDED_CATEGORICAL,
        }:
            self.scaler = StandardScaler().fit(X[self.num_cols].fillna(self.medians))

        if self.strategy == PreprocessingStrategy.ONE_HOT_SCALED:
            self.encoder = OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ).fit(X[self.cat_cols].fillna("__MISSING__").astype(str))

        elif self.strategy == PreprocessingStrategy.NATIVE_CATEGORICAL:
            self.categories = {
                col: (X[col].fillna("__MISSING__").astype(str).unique().tolist())
                for col in self.cat_cols
            }

        elif self.strategy == PreprocessingStrategy.EMBEDDED_CATEGORICAL:
            self.category_maps = {
                col: {
                    value: i + 1
                    for i, value in enumerate(
                        X[col].fillna("__MISSING__").astype(str).unique()
                    )
                }
                for col in self.cat_cols
            }

            self.cat_cardinalities = [
                len(self.category_maps[col]) + 1 for col in self.cat_cols
            ]

            self.num_numeric = len(self.num_cols)

        else:
            raise ValueError(f"Unsupported preprocessing strategy: {self.strategy}")

        return self

    def transform(self, X):
        if self.strategy == PreprocessingStrategy.ONE_HOT_SCALED:
            X_num = self.scaler.transform(X[self.num_cols].fillna(self.medians))

            X_cat = self.encoder.transform(
                X[self.cat_cols].fillna("__MISSING__").astype(str)
            )

            return np.hstack([X_num, X_cat])

        if self.strategy == PreprocessingStrategy.NATIVE_CATEGORICAL:
            X = X[self.num_cols + self.cat_cols].copy()

            X[self.num_cols] = X[self.num_cols].fillna(self.medians)

            for col in self.cat_cols:
                X[col] = pd.Categorical(
                    X[col].fillna("__MISSING__").astype(str),
                    categories=self.categories[col],
                )

            return X

        if self.strategy == PreprocessingStrategy.EMBEDDED_CATEGORICAL:
            X_num = self.scaler.transform(X[self.num_cols].fillna(self.medians)).astype(
                np.float32
            )

            X_cat = (
                np.column_stack(
                    [
                        X[col]
                        .fillna("__MISSING__")
                        .astype(str)
                        .map(self.category_maps[col])
                        .fillna(0)
                        .astype(np.int64)
                        for col in self.cat_cols
                    ]
                )
                if self.cat_cols
                else np.empty(
                    (len(X), 0),
                    dtype=np.int64,
                )
            )

            return X_num, X_cat

        raise ValueError(f"Unsupported preprocessing strategy: {self.strategy}")

    def get_model_params(self):
        if self.strategy == PreprocessingStrategy.EMBEDDED_CATEGORICAL:
            return {
                "num_numeric": self.num_numeric,
                "cat_cardinalities": self.cat_cardinalities,
            }

        return {}

    def fit_transform(self, X):
        return self.fit(X).transform(X)
