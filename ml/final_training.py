from ml.preprocessing import TabularPreprocessor
from ml.models import TabularModel
from ml.trainer import train_model


def train_final_model(
    df,
    target_column,
    model_config,
    training_config,
):
    X = df.drop(columns=[target_column])
    y = df[target_column].to_numpy()

    preprocessor = TabularPreprocessor()
    preprocessor.fit(X)

    full_data = preprocessor.transform(X)

    cardinalities = [
        len(preprocessor.category_maps[column]) + 1
        for column in preprocessor.categorical_columns
    ]

    embedding_dims = [
        min(16, max(2, cardinality // 2))
        for cardinality in cardinalities
    ]

    model = TabularModel(
        num_numeric_features=len(
            preprocessor.numeric_columns
        ),
        categorical_cardinalities=cardinalities,
        embedding_dims=embedding_dims,
        **model_config,
    )

    model = train_model(
        model=model,
        numeric_data=full_data["numeric"],
        categorical_data=full_data["categorical"],
        target=y,
        **training_config,
    )

    return model, preprocessor