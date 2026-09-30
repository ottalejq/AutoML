# AutoML

A Python API for training regression models on CSV data and serving predictions. Upload a dataset, choose a numeric target, and start a background job that searches model configurations, evaluates the winner on held-out data, and saves it with its fitted preprocessor.

Built with FastAPI, SQLAlchemy, Celery, Redis, scikit-learn, LightGBM, and PyTorch. PostgreSQL stores dataset, job, and model metadata.

The current scope is single-target tabular regression. There is no web frontend; use the interactive API docs or an HTTP client. Training compares elastic net, gradient boosting, and a neural network automatically. Classification and time-series-specific splitting are not implemented.

## Architecture

```mermaid
flowchart LR
    Client --> API[FastAPI]
    API -->|Queue training| Redis
    Redis --> Worker[Celery worker]
    Worker --> Pipeline[Search and training]
    API --> DB[(PostgreSQL)]
    Worker --> DB
    API -->|Upload CSV / predict| Storage[Shared storage]
    Pipeline -->|Read CSV / save model| Storage
```

Training runs in the worker; prediction runs in the API. Both need the same database and storage directory. CSVs are saved under `storage/datasets/<dataset-id>.csv`; model/preprocessor bundles use `storage/models/<model-id>.joblib`.

The database contains three tables: `datasets` records the uploaded file and target, `training_jobs` records execution status and the resulting model ID, and `models` records the selected configuration, test score, and artifact path. Job polling reads PostgreSQL, not the Celery result backend. Each prediction request loads its model artifact from disk.

## Run with Docker

From the repository root, with Docker Compose installed:

```sh
docker compose up --build -d
```

Once PostgreSQL is ready, apply the database migrations:

```sh
docker compose exec api python -m alembic upgrade head
docker compose logs -f api worker
```

Compose starts the API, worker, PostgreSQL, and Redis using `.env.docker`. **The API does not create tables on startup; migrations are required before uploading data.** The initial migration creates the current UUID-based schema. Existing databases created outside Alembic need their schema and migration history reconciled before applying it.

- Interactive API docs: [localhost:8000/docs](http://localhost:8000/docs)
- Liveness: [localhost:8000/health](http://localhost:8000/health)
- Readiness: [localhost:8000/ready](http://localhost:8000/ready) checks database connectivity and the Redis broker, but not migrations or worker availability.

PostgreSQL data persists in the `postgres_data` volume; datasets and artifacts persist in the host's `storage/` directory. Compose publishes PostgreSQL on port **5433** and the API on **8000**; Redis is internal only. No service readiness checks are configured in Compose, so retry migrations if PostgreSQL is still starting.

Rebuild with `docker compose up --build -d` after changing Python code.

## API workflow

All steps are available through `/docs`. Dataset, job, and model IDs are UUID strings.

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `POST` | `/datasets/` | Upload a CSV and select its target column |
| `POST` | `/training/` | Queue training for an uploaded dataset |
| `GET` | `/training/{job_id}` | Read job status and the resulting model ID |
| `GET` | `/models/{model_id}/info` | Read saved model metadata |
| `POST` | `/models/{model_id}/predict` | Predict from JSON feature rows |
| `GET` | `/health` | Check API liveness |
| `GET` | `/ready` | Check database and Redis broker connectivity |

### 1. Upload a dataset

Call `POST /datasets/` with multipart fields `file` (a CSV with a header row and `.csv` filename) and `target_column`:

```sh
curl -X POST http://localhost:8000/datasets/ -F "file=@housing.csv" -F "target_column=price"
```

Use `curl.exe` on Windows PowerShell. The response contains `id`, `filename`, and `target_column`; keep `id` for training.

Use a numeric target without missing values and enough rows for splitting and cross-validation. Feature types are inferred by pandas. Include at least one numeric feature for scaled preprocessing, and observed values for median imputation. Upload validation checks CSV parsing and target-column presence, not all training requirements.

### 2. Start training

Send to `POST /training/`:

```json
{
  "dataset_id": "<dataset-id>"
}
```

The response contains `job_id`, `dataset_id`, `target_column`, and `status: "queued"`. Search settings are configured in Python, not in the request.

### 3. Check the job and model

Call `GET /training/{job_id}`. Status changes from `queued` to `running`, then `completed` or `failed`. The response includes dataset and target details, timestamps, `model_id` after success, and `error_message` on failure. It does not expose per-trial progress.

Use `GET /models/{model_id}/info` to retrieve the model's class, model/fit parameters, metric, held-out `test_score`, dataset ID, target column, artifact path, and creation time.

If a job stays queued, check worker and Redis logs. For failed jobs, inspect `error_message` and worker logs.

### 4. Make predictions

Send feature rows to `POST /models/{model_id}/predict`:

```json
{
  "rows": [
    {"area": 85.0, "rooms": 3, "city": "Berlin"},
    {"area": 120.0, "rooms": 4, "city": "Hamburg"}
  ]
}
```

Include the training feature columns; the target is not required. The response contains `model_id` and a `predictions` array with one value per row in input order.

Extra columns are ignored. Keep numeric features numeric in JSON; preprocessing uses the feature types learned during training. Missing feature values can be imputed, but a feature column absent from the entire request is rejected.

Unknown dataset/job IDs in training requests and unknown model IDs return HTTP 404. Invalid UUIDs and missing prediction feature columns return HTTP 422. A rejected filename, invalid CSV, or absent target column returns HTTP 400 during upload.

## Training and configuration

The pipeline in [ml/pipeline.py](ml/pipeline.py):

1. Reserves 20% of the data for testing.
2. Shuffles parameter combinations with a fixed seed and evaluates up to 25 configurations per model family using five-fold cross-validation.
3. Selects the lowest mean RMSE and evaluates a separately fitted model on the test set.
4. Fits the selected configuration again using the full input dataset and saves the model with its preprocessor.

Every `Model.fit` call, including the final fit, reserves 20% of its input for validation. Preprocessing is fitted only on that call's training portion. Neural-network validation controls early stopping and best-weight selection. Elastic net ignores the validation portion. Consequently, passing the full dataset to the final fit does not train on every row. The reported test score belongs to the evaluation model, before the final fit.

With the default settings, search performs 375 cross-validation fits (three families × 25 configurations × five folds), followed by one evaluation fit and one final fit. A failed trial propagates out of the search and fails the job; trials are not skipped or retried. Cross-validation scores are returned by the Python pipeline but are not stored in model metadata.

| Model | Implementation | Preprocessing |
| --- | --- | --- |
| Elastic net | scikit-learn `ElasticNet` | Scaled numeric and one-hot categorical features |
| Gradient boosting | LightGBM `LGBMRegressor` | Numeric and native categorical features |
| Neural network | PyTorch MLP with categorical embeddings | Scaled numeric and integer-encoded categorical features |

Numeric missing values use training medians; categorical missing values use `__MISSING__`. Unseen categories become all-zero one-hot encodings, missing native categories, or embedding index `0`, respectively.

Edit [ml/config.py](ml/config.py) to change the search:

| Setting | Default |
| --- | --- |
| Metric | `rmse` |
| Cross-validation folds | `5` |
| Test / per-fit validation fraction | `0.2` / `0.2` |
| Trials per model family | `25` |
| Random seed | `42` |

`MODEL_SEARCH_SPACES` and `FIT_SEARCH_SPACES` define candidates; `MODEL_FIXED_PARAMS` and `FIT_FIXED_PARAMS` define shared parameters. The neural network uses AdamW, MSE loss, up to 500 epochs, and early-stopping patience of 20. Reduce trials or epochs for shorter development runs. Compose does not configure GPU access; CPU is the default there.

The metric registry includes RMSE, MAE, and R-squared. Selection between model families respects the metric direction, but search within each family still minimizes the score, so R-squared selection is not yet correct. Use RMSE or MAE for the current search implementation. The stored metric name follows `SEARCH_CONFIG`.

### Use the ML layer directly

The ML package can run without the API, database, Redis, or Celery. Run this from the repository root after installing dependencies; `housing.csv` must meet the same data requirements as an uploaded CSV:

```python
from pathlib import Path

import pandas as pd

from ml.models import Model
from ml.pipeline import train_pipeline

df = pd.read_csv("housing.csv")
features = df.drop(columns=["price"])
result = train_pipeline(features, df["price"])
print(result["model_class"].__name__, result["test_score"])

artifact = Path("housing.joblib")
result["model"].save(artifact)
model = Model.load(artifact)
predictions = model.predict(features.head())
print(predictions)
```

This runs the configured search across all model families. To fit one configuration directly, use `Model(ElasticNetModel, alpha=0.01).fit(features, df["price"])` after importing `ElasticNetModel` from `ml.models`.

## Local development

Use Python 3.13, matching Docker and CI:

```sh
python -m venv .venv
```

Activate with `.venv\Scripts\Activate.ps1` on PowerShell or `source .venv/bin/activate` on Linux/macOS, then install dependencies:

```sh
python -m pip install -r requirements.txt
```

Settings come from environment variables or a root `.env` file. All three are required (`.env.example` is currently empty):

```dotenv
DATABASE_URL=postgresql+psycopg2://postgres:postgres@localhost:5433/automl
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/1
```

| Variable | Purpose |
| --- | --- |
| `DATABASE_URL` | SQLAlchemy connection URL used by the API, worker, and Alembic |
| `CELERY_BROKER_URL` | Redis connection used to queue training tasks |
| `CELERY_RESULT_BACKEND` | Celery task-result storage connection |

Environment variables take precedence over `.env`. Compose supplies container hostnames through the checked-in `.env.docker`; those hostnames differ from the `localhost` addresses used above.

This database URL uses Compose's published PostgreSQL port. The Redis URLs require a separately available Redis service or an added Compose port mapping. The API and worker must use the same database and run from the repository root so their storage paths agree.

Apply migrations, then start the API:

```sh
python -m alembic upgrade head
python -m uvicorn app.main:app --reload
```

Start the worker in a separate terminal:

```sh
python -m celery -A worker.celery_app:celery_app worker --loglevel=info
```

Use the Docker worker when developing on Windows.

### Checks

```sh
python -m ruff check .
python -m ruff format --check .
python -m pytest -v
```

CI runs lint, formatting, migrations against PostgreSQL, and tests on Python 3.13 for pushes and pull requests.

The test suite covers dataset lookup, metrics, preprocessing, model fitting/prediction, configuration generation, artifact saving, and API health. Training-to-prediction tests exercise all three model families with an isolated in-memory SQLite database and temporary artifacts, reducing the search to one trial and two folds. Tests do not require live PostgreSQL or Redis services and do not exercise HTTP uploads, readiness checks, or a live queued Celery job.

## Current limitations

- Uploads and training data are read into memory. There is no configured upload-size limit or streaming training path; one-hot encoding also produces dense arrays.
- Scaled preprocessing requires at least one numeric feature. Numeric columns with no observed training values retain missing values after median imputation.
- Splits are random, with no grouping or chronological ordering. The target must already be suitable for regression; upload validation does not check target type or missing target values.
- The neural network automatically selects CUDA when available, but its validation tensors remain on CPU. For validation-based training on a GPU-equipped machine, set `"device": "cpu"` in `FIT_FIXED_PARAMS[NeuralNetworkModel]` until device handling is corrected.
- The API commits a queued job before publishing its Celery task. A broker failure can leave a queued record without a dispatched task. There is no application-configured automatic retry or job cancellation.
- There are no authentication, dataset/model listing, or deletion endpoints. The API exposes the routes listed above.

## Project structure

| Path | Purpose |
| --- | --- |
| `app/api/` | Dataset, training, prediction, and model-info routes |
| `app/services/` | Dataset lookup, training orchestration, and artifacts |
| `app/core/`, `app/db/` | Settings, ORM models, and database sessions |
| `ml/base.py`, `ml/types.py` | Model interface and preprocessing strategy enum |
| `ml/models.py`, `ml/preprocessing.py` | Model implementations, fitted model wrapper, and feature transformations |
| `ml/config.py`, `ml/search.py`, `ml/pipeline.py` | Search spaces, metrics, cross-validation, selection, and final fitting |
| `worker/` | Celery application and training task |
| `alembic/` | Database migrations |
| `tests/` | Unit and integration tests |
| `storage/` | Uploaded datasets and saved models (ignored by Git) |
| `compose.yaml`, `Dockerfile` | Container setup |
| `.github/workflows/` | CI configuration |

## License

[MIT](LICENSE).
