import pandas as pd

from ml.search import hyperparameter_search
from ml.final_training import train_final_model

from app.services.model_service import save_model_artifacts

from app.db.models import Model



def run_training(
    dataset,
    target_column,
    db,
    progress_callback=None,
):

    df = pd.read_csv(
        dataset.storage_path,
        sep=r"\s+",
    )

    results = hyperparameter_search(
        df=df,
        target_column=target_column,
        n_trials=10,
        n_splits=5,
        progress_callback=progress_callback,
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

    model_record = Model(
        dataset_id=dataset.id,
        target_column=target_column,
        task_type="regression",
        model_path=artifacts["model_path"],
        preprocessor_path=artifacts["preprocessor_path"],
        model_config=best_trial["model_config"],
        training_config=best_trial["training_config"],
        mean_rmse=best_trial["mean_rmse"],
        std_rmse=best_trial["std_rmse"],
    )

    db.add(model_record)
    db.commit()
    db.refresh(model_record)



    return {
        "dataset_id": dataset.id,
        "model_id": model_record.id,
        "target_column": target_column,
        "task_type": "regression",
        "best_trial": results["best_trial"],
        "artifacts": artifacts,
    }