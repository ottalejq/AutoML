import numpy as np
import random
import itertools

from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error


from ml.models import Model



def generate_all_configs(
    model_search_space,
    fit_search_space,
):
    model_keys = list(model_search_space.keys())
    model_values = list(model_search_space.values())

    fit_keys = list(fit_search_space.keys())
    fit_values = list(fit_search_space.values())

    model_configs = [
        dict(zip(model_keys, values))
        for values in itertools.product(*model_values)
    ]

    fit_configs = [
        dict(zip(fit_keys, values))
        for values in itertools.product(*fit_values)
    ]

    return list(
        itertools.product(
            model_configs,
            fit_configs,
        )
    )



def cross_validate(
    X,
    y,
    model_class,
    model_params,
    fit_params,
    metric,
    n_folds,
    val_size,
    seed,
):
    kfold = KFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=seed,
    )

    fold_scores = []

    for train_idx, test_idx in kfold.split(X):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]

        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]

        model = Model(
            model_class,
            **model_params,
        )

        model.fit(
            X_train,
            y_train,
            val_size=val_size,
            seed=seed,
            **fit_params,
        )

        y_pred = model.predict(X_test)

        score = metric(y_test, y_pred)

        fold_scores.append(score)

    return np.mean(fold_scores)



def hyperparameter_search(
    X,
    y,
    model_class,
    model_search_space,
    fit_search_space,
    model_fixed_params=None,
    fit_fixed_params=None,
    metric=root_mean_squared_error,
    n_folds=5,
    val_size=0.2,
    seed=42,
    n_trials=25,
):
    model_fixed_params = model_fixed_params or {}
    fit_fixed_params = fit_fixed_params or {}

    configs = generate_all_configs(
        model_search_space,
        fit_search_space,
    )

    rng = random.Random(seed)
    rng.shuffle(configs)

    configs = configs[:min(n_trials, len(configs))]

    best_score = float("inf")
    best_model_params = None
    best_fit_params = None

    for model_params, fit_params in configs:
        model_params = {
            **model_fixed_params,
            **model_params,
            "seed": seed,
        }

        fit_params = {
            **fit_fixed_params,
            **fit_params,
        }

        score = cross_validate(
            X=X,
            y=y,
            model_class=model_class,
            model_params=model_params,
            fit_params=fit_params,
            metric=metric,
            n_folds=n_folds,
            val_size=val_size,
            seed=seed,
        )

        if score < best_score:
            best_score = score
            best_model_params = model_params
            best_fit_params = fit_params

    return {
        "score": best_score,
        "model_params": best_model_params,
        "fit_params": best_fit_params,
    }