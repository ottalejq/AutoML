# AutoML

A Python API for training regression models on CSV data and serving predictions. Upload a dataset, choose a numeric target, and start a background job that searches model configurations, evaluates the winner on held-out data, and saves it with its fitted preprocessor.

Built with FastAPI, SQLAlchemy, Celery, Redis, scikit-learn, LightGBM, and PyTorch. PostgreSQL stores dataset, job, and model metadata.

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

Unknown dataset/job IDs in training requests and unknown model IDs return HTTP 404. Missing prediction feature columns return HTTP 422.

## Training and configuration

The pipeline in [ml/pipeline.py](ml/pipeline.py):

1. Reserves 20% of the data for testing.
2. Shuffles parameter combinations with a fixed seed and evaluates up to 25 configurations per model family using five-fold cross-validation.
3. Selects the lowest mean RMSE and evaluates a separately fitted model on the test set.
4. Fits the selected configuration again using the full input dataset and saves the model with its preprocessor.

Every `Model.fit` call, including the final fit, reserves 20% of its input for validation. Preprocessing is fitted only on that call's training portion. Neural-network validation controls early stopping and best-weight selection. The reported test score belongs to the evaluation model, before the final fit.

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

The metric registry includes RMSE, MAE, and R-squared. Selection between model families respects the metric direction, but search within each family still minimizes the score, so R-squared selection is not yet correct. The stored metric name follows `SEARCH_CONFIG`.

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

The test suite covers preprocessing, model fitting/prediction, configuration generation, artifact-save delegation, and API health. It currently also has stale references to removed `ml.dataset`, an outdated metric-registry test, and a training-to-prediction test that requires a missing `db_session` fixture. These need updating before the full suite can pass; it does not exercise a live queued Celery job.

## Project structure

| Path | Purpose |
| --- | --- |
| `app/api/` | Dataset, training, prediction, and model-info routes |
| `app/services/` | Dataset lookup, training orchestration, and artifacts |
| `app/core/`, `app/db/` | Settings, ORM models, and database sessions |
| `ml/` | Models, preprocessing, search, metrics, and training pipeline |
| `worker/` | Celery application and training task |
| `alembic/` | Database migrations |
| `tests/` | Unit and integration tests |
| `storage/` | Uploaded datasets and saved models (ignored by Git) |
| `compose.yaml`, `Dockerfile` | Container setup |
| `.github/workflows/` | CI configuration |

## License

[MIT](LICENSE).
