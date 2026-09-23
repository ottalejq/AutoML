from sklearn.model_selection import train_test_split

from ml.config import (
    SEARCH_CONFIG,
    MODEL_SEARCH_SPACES,
    FIT_SEARCH_SPACES,
    MODEL_FIXED_PARAMS,
    FIT_FIXED_PARAMS,
    METRICS,
)
from ml.models import Model
from ml.search import hyperparameter_search




def train_pipeline(X, y):
    seed = SEARCH_CONFIG["seed"]
    val_size = SEARCH_CONFIG["val_size"]

    metric_config = METRICS[SEARCH_CONFIG["metric"]]
    metric = metric_config["fn"]
    direction = metric_config["direction"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=SEARCH_CONFIG["test_size"],
        random_state=seed,
    )

    best_result = None

# def train_pipeline(X, y):
#     seed = SEARCH_CONFIG["seed"]
#     val_size = SEARCH_CONFIG["val_size"]
#     metric = METRICS[SEARCH_CONFIG["metric"]]

#     X_train, X_test, y_train, y_test = train_test_split(
#         X,
#         y,
#         test_size=SEARCH_CONFIG["test_size"],
#         random_state=seed,
#     )

#     best_result = {
#         "score": float("inf"),
#         "model_class": None,
#         "model_params": None,
#         "fit_params": None,
#     }

    for model_class in MODEL_SEARCH_SPACES:
        result = hyperparameter_search(
            X=X_train,
            y=y_train,
            model_class=model_class,
            model_search_space=MODEL_SEARCH_SPACES[model_class],
            fit_search_space=FIT_SEARCH_SPACES[model_class],
            model_fixed_params=MODEL_FIXED_PARAMS[model_class],
            fit_fixed_params=FIT_FIXED_PARAMS[model_class],
            metric=metric,
            n_folds=SEARCH_CONFIG["n_folds"],
            val_size=val_size,
            seed=seed,
            n_trials=SEARCH_CONFIG["n_trials"],
        )

        # if result["score"] < best_result["score"]:
        #     best_result = {
        #         **result,
        #         "model_class": model_class,
        #     }

        if best_result is None:
            better = True
        elif direction == "minimize":
            better = result["score"] < best_result["score"]
        elif direction == "maximize":
            better = result["score"] > best_result["score"]
        else:
            raise Exception('Optimization direction not defined!')

        if better:
            best_result = {
                **result,
                "model_class": model_class,
            }

    evaluation_model = Model(
        best_result["model_class"],
        **best_result["model_params"],
    )

    evaluation_model.fit(
        X_train,
        y_train,
        val_size=val_size,
        seed=seed,
        **best_result["fit_params"],
    )

    y_pred = evaluation_model.predict(X_test)

    test_score = metric(
        y_test,
        y_pred,
    )

    final_model = Model(
        best_result["model_class"],
        **best_result["model_params"],
    )

    final_model.fit(
        X,
        y,
        val_size=val_size,
        seed=seed,
        **best_result["fit_params"],
    )

    return {
        "model": final_model,
        "test_score": test_score,
        "cross_validation_score": best_result["score"],
        "model_class": best_result["model_class"],
        "model_params": best_result["model_params"],
        "fit_params": best_result["fit_params"],
    }