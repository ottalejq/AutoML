import numpy as np
from sklearn.model_selection import KFold

from ml.preprocessing import TabularPreprocessor
from ml.models import TabularModel
from ml.trainer import train_model
from ml.evaluator import evaluate_model

import random
import itertools


MODEL_SEARCH_SPACE = {
    "hidden_dim": [32, 64, 128],
    "num_layers": [1, 2, 3],
    "dropout": [0.0, 0.1, 0.2],
}


TRAINING_SEARCH_SPACE = {
    "learning_rate": [1e-4, 5e-4, 1e-3],
    "weight_decay": [0.0, 1e-5, 1e-4],
}


def generate_all_configs():
    model_keys = list(MODEL_SEARCH_SPACE.keys())
    model_values = list(MODEL_SEARCH_SPACE.values())

    training_keys = list(TRAINING_SEARCH_SPACE.keys())
    training_values = list(TRAINING_SEARCH_SPACE.values())

    all_configs = []

    for model_combination in itertools.product(*model_values):
        model_config = dict(
            zip(model_keys, model_combination)
        )

        for training_combination in itertools.product(
            *training_values
        ):
            training_config = dict(
                zip(training_keys, training_combination)
            )

            all_configs.append(
                (model_config, training_config)
            )

    return all_configs


def cross_validate(
    df,
    target_column,
    model_config,
    training_config,
    n_splits=5,
):
    X = df.drop(columns=[target_column])
    y = df[target_column].to_numpy()

    kfold = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42,
    )

    fold_scores = []

    for fold_number, (train_idx, val_idx) in enumerate(
        kfold.split(X),
        start=1,
    ):
        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]

        y_train = y[train_idx]
        y_val = y[val_idx]

        preprocessor = TabularPreprocessor()
        preprocessor.fit(X_train)

        train_data = preprocessor.transform(X_train)
        val_data = preprocessor.transform(X_val)

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
            numeric_data=train_data["numeric"],
            categorical_data=train_data["categorical"],
            target=y_train,
            **training_config,
        )

        metrics = evaluate_model(
            model=model,
            numeric_data=val_data["numeric"],
            categorical_data=val_data["categorical"],
            target=y_val,
        )

        fold_scores.append(metrics["rmse"])

        print(
            f"Fold {fold_number}: "
            f"RMSE = {metrics['rmse']:.4f}"
        )

    return {
        "fold_rmse": fold_scores,
        "mean_rmse": float(np.mean(fold_scores)),
        "std_rmse": float(np.std(fold_scores)),
    }



def hyperparameter_search(
    df,
    target_column,
    n_trials=10,
    n_splits=5,
):
    all_configs = generate_all_configs()

    random.shuffle(all_configs)

    selected_configs = all_configs[:n_trials]

    trial_results = []

    for trial_number, (
        model_config,
        training_config,
    ) in enumerate(selected_configs, start=1):

        print(
            f"Trial {trial_number}/{len(selected_configs)}"
        )

        cv_results = cross_validate(
            df=df,
            target_column=target_column,
            model_config=model_config,
            training_config=training_config,
            n_splits=n_splits,
        )

        trial_results.append({
            "model_config": model_config,
            "training_config": training_config,
            "mean_rmse": cv_results["mean_rmse"],
            "std_rmse": cv_results["std_rmse"],
            "fold_rmse": cv_results["fold_rmse"],
        })

    best_trial = min(
        trial_results,
        key=lambda trial: trial["mean_rmse"],
    )

    return {
        "best_trial": best_trial,
        "all_trials": trial_results,
    }