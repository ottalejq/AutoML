from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)

from ml.models import (
    ElasticNetModel,
    GradientBoostingModel,
    NeuralNetworkModel,
)


METRICS = {
    "rmse": {
        "fn": root_mean_squared_error,
        "direction": "minimize",
    },
    "mae": {
        "fn": mean_absolute_error,
        "direction": "minimize",
    },
    "r2": {
        "fn": r2_score,
        "direction": "maximize",
    },
}


SEARCH_CONFIG = {
    "metric": "rmse",
    "n_folds": 5,
    "test_size": 0.2,
    "val_size": 0.2,
    "seed": 42,
    "n_trials": 25,
}

MODEL_SEARCH_SPACES = {
    ElasticNetModel: {
        "alpha": [
            1e-4,
            1e-3,
            1e-2,
            1e-1,
            1.0,
        ],
        "l1_ratio": [
            0.05,
            0.25,
            0.5,
            0.75,
            0.95,
        ],
    },

    GradientBoostingModel: {
        "n_estimators": [
            300,
            600,
            1000,
        ],
        "learning_rate": [
            0.02,
            0.05,
            0.1,
        ],
        "num_leaves": [
            15,
            31,
            63,
        ],
        "min_child_samples": [
            10,
            20,
            50,
        ],
        "subsample": [
            0.8,
            1.0,
        ],
        "colsample_bytree": [
            0.8,
            1.0,
        ],
        "reg_alpha": [
            0.0,
            0.1,
        ],
        "reg_lambda": [
            0.0,
            0.1,
            1.0,
        ],
    },

    NeuralNetworkModel: {
        "embedding_dim": [
            4,
            8,
            16,
            32,
        ],
        "hidden_dim": [
            64,
            128,
            256,
        ],
        "n_layers": [
            1,
            2,
            3,
        ],
    },
}


FIT_SEARCH_SPACES = {
    ElasticNetModel: {},

    GradientBoostingModel: {},

    NeuralNetworkModel: {
        "learning_rate": [
            3e-4,
            1e-3,
            3e-3,
        ],
        "weight_decay": [
            0.0,
            1e-5,
            1e-4,
            1e-3,
        ],
        "batch_size": [
            32,
            64,
            128,
        ],
    },
}


MODEL_FIXED_PARAMS = {
    ElasticNetModel: {
        "max_iter": 10_000,
    },

    GradientBoostingModel: {
        "verbosity": -1,
    },

    NeuralNetworkModel: {},
}


FIT_FIXED_PARAMS = {
    ElasticNetModel: {},

    GradientBoostingModel: {},

    NeuralNetworkModel: {
        "epochs": 500,
        "patience": 20,
    },
}