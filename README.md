# AutoML

A Python service that automatically compares machine-learning models for predicting numeric values, such as house prices, from CSV data. It handles data preparation, model selection, evaluation, and predictions through a REST API with interactive documentation.

## How it works

1. **Upload data:** provide a CSV file and choose the column to predict.
2. **Start training:** a background worker compares elastic net (scikit-learn), gradient boosting (LightGBM), and a neural network (PyTorch).
3. **Evaluate:** models are compared across multiple data splits using RMSE, a measure of prediction error where lower is better. The selected configuration is also evaluated on separate test data.
4. **Predict:** the saved model and its data-preparation steps are reused to predict values for new rows.

Training runs separately from the API, so users can check progress while it continues. The API supports dataset upload, training status, model details, and predictions.

## Architecture

| Technology | Role |
| --- | --- |
| FastAPI | Provides the API and interactive documentation |
| Celery and Redis | Queue and process background training jobs |
| PostgreSQL and SQLAlchemy | Store dataset, training-job, and model information |
| scikit-learn, LightGBM, and PyTorch | Prepare data, train models, and generate predictions |
| Shared file storage | Keeps uploaded datasets and saved models available to the API and worker |
| Docker | Packages the API and worker in the same image, with different startup commands |

## Google Cloud

The cloud deployment runs in `europe-west3` (Frankfurt), using managed services in place of the local Docker Compose stack:

| Service | Purpose |
| --- | --- |
| Cloud Run Service | Hosts the API, with authenticated access and automatic scaling |
| Cloud Run Worker Pool | Runs the Celery worker continuously |
| Cloud SQL and Memorystore | Provide PostgreSQL and Redis |
| Cloud Storage | Stores datasets and models; mounted in both containers through Cloud Storage FUSE |
| Cloud Build and Artifact Registry | Build and store the Docker image, using CPU-only PyTorch |
| Secret Manager and IAM | Manage credentials and access permissions |
| Cloud Run Job | Runs Alembic database migrations separately from the API |

The API and worker reach Redis through a private VPC connection. A shared storage mount lets both use the existing file-based Python code.

To access the deployed API with an authorized Google Cloud account:

```sh
gcloud run services proxy automl-api --project=ottalejq-automl --region=europe-west3 --port=8080
```

Open [localhost:8080/docs](http://localhost:8080/docs). Cloud infrastructure is configured separately; this repository does not include automated cloud provisioning.

## Run locally

With Docker Compose installed, run from the repository root:

```sh
docker compose up --build -d
docker compose exec api python -m alembic upgrade head
```

The second command creates or updates the database tables. Retry it if PostgreSQL is still starting. Open [localhost:8000/docs](http://localhost:8000/docs) to upload data and try the full workflow. Data persists in `storage/` and the PostgreSQL Docker volume.

## Development and scope

Uses Python 3.13. Training settings are defined in [ml/config.py](ml/config.py); defaults compare up to 25 configurations per model family with five-fold cross-validation and a 20% test split. Each fit also reserves validation data, including the final saved model.

Source code is organized into `app/` (API and database), `ml/` (models and training), `worker/` (background tasks), and `alembic/` (database migrations). CI runs Ruff checks, migration validation, and pytest tests.

The current scope is single-target tabular regression with numeric and categorical features. Datasets must fit in memory and have a numeric target without missing values. Classification and time-series-specific splitting are not supported. There is no separate web frontend or application-level authentication; the cloud deployment protects access through Cloud Run IAM.

[MIT License](LICENSE).
