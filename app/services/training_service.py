import pandas as pd

from ml.search import hyperparameter_search
from ml.final_training import train_final_model

from app.services.model_service import save_model_artifacts

def run_training(dataset, target_column):
    df = pd.read_csv(
        dataset.storage_path,
        sep=r"\s+",
    )

    results = hyperparameter_search(
        df=df,
        target_column=target_column,
        n_trials=10,
        n_splits=5,
    )

    best_trial = results["best_trial"]

    model, preprocessor = train_final_model(
        df=df,
        target_column=target_column,
        model_config=best_trial["model_config"],
        training_config=best_trial["training_config"],
    )

    artifacts = save_model_artifacts(
        model=model,
        preprocessor=preprocessor,
        dataset_id=dataset.id,
        model_config=best_trial["model_config"],
        training_config=best_trial["training_config"],
        target_column=target_column,
    )

    return {
        "dataset_id": dataset.id,
        "target_column": target_column,
        "search_results": results,
        "artifacts": artifacts,
    }