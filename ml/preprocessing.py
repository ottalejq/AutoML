import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def prepare_data(df: pd.DataFrame, target_column: str):
    X = df.drop(columns=[target_column])
    y = df[target_column]

    numeric_columns = X.select_dtypes(include="number").columns.tolist()
    categorical_columns = X.select_dtypes(exclude="number").columns.tolist()

    scaler = StandardScaler()

    if numeric_columns:
        numeric_data = X[numeric_columns].copy()
        numeric_data = numeric_data.fillna(numeric_data.median())
        numeric_data = scaler.fit_transform(numeric_data)
    else:
        numeric_data = np.empty((len(X), 0), dtype=np.float32)

    category_maps = {}
    categorical_arrays = []

    for column in categorical_columns:
        values = X[column].fillna("__MISSING__").astype(str)

        categories = sorted(values.unique())

        mapping = {
            category: index + 1
            for index, category in enumerate(categories)
        }

        category_maps[column] = mapping

        encoded = (
            values.map(mapping)
            .fillna(0)
            .astype(int)
            .to_numpy()
        )

        categorical_arrays.append(encoded)

    if categorical_arrays:
        categorical_data = np.column_stack(categorical_arrays)
    else:
        categorical_data = np.empty((len(X), 0), dtype=np.int64)

    return {
        "numeric_data": numeric_data.astype(np.float32),
        "categorical_data": categorical_data.astype(np.int64),
        "target": y.to_numpy(),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "category_maps": category_maps,
        "scaler": scaler,
    }