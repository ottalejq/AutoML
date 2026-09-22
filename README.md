# AutoML

A Python API for training regression models on CSV data and serving predictions. Upload a dataset, choose a numeric target column, and start a background training job. The training pipeline searches model configurations, evaluates the selected configuration on held-out data, and saves a model together with its fitted preprocessor.

Built with FastAPI, SQLAlchemy, Celery, Redis, scikit-learn, LightGBM, and PyTorch. The Docker Compose setup uses PostgreSQL for dataset, job, and model metadata.

## Architecture

```mermaid
flowchart LR
    Client[Client] --> API[FastAPI]
    API -->|Enqueue training| Redis[Redis]
    Redis --> Worker[Celery worker]
    Worker --> Training[Training service]
    Training --> Pipeline[Search and training pipeline]
    Pipeline --> Models[Models and preprocessing]
    API --> DB[(PostgreSQL)]
    Worker --> DB
    Training --> DB
    API -->|Upload CSV / load model| Storage[Shared storage]
    Training -->|Read CSV / save model| Storage
```

Training runs in the worker. Prediction runs in the API process using the saved model. Both processes must access the same database and storage directory.

## Run with Docker

Install Docker with Docker Compose, then run these commands from the repository root:

```sh
docker compose up --build -d
docker compose logs -f api worker
```

The Compose file starts the API, a Celery worker, PostgreSQL, and Redis. It reads the service connection settings from `.env.docker`.

- API documentation: [localhost:8000/docs](http://localhost:8000/docs)
- Alternative API documentation: [localhost:8000/redoc](http://localhost:8000/redoc)
- Health endpoint: [localhost:8000/health](http://localhost:8000/health)

The API creates missing database tables on startup. PostgreSQL data is stored in the `postgres_data` Docker volume; uploaded datasets and model artifacts are stored under the host's `storage/` directory.

The services have no readiness health checks configured. If the API starts before PostgreSQL is ready, check the logs and restart it with `docker compose restart api`. The `/health` endpoint reports that the API is responding; it does not check the database, Redis, or worker.

Application code is copied into the images. After changing Python code, run `docker compose up --build -d` again.

**Existing databases:** startup uses `create_all`, which does not update existing tables. The Alembic revisions currently describe an older schema with integer IDs and different columns, while the current models use UUIDs. Reconcile the migrations with [app/db/models.py](app/db/models.py) before using them to upgrade an existing database. The startup path above is intended for a fresh database.

## API workflow

You can perform all four steps through `/docs`. IDs returned by the API are UUID strings.

### 1. Upload a dataset

Call `POST /datasets/` with multipart form fields:

| Field | Value |
| --- | --- |
| `file` | A CSV file with a header row and a `.csv` filename |
| `target_column` | The name of the numeric column to predict |

For example, if `housing.csv` contains `area`, `rooms`, `city`, and `price`:

```sh
curl -X POST http://localhost:8000/datasets/ -F "file=@housing.csv" -F "target_column=price"
```

On Windows PowerShell, use `curl.exe` for this command. Keep the returned `id` as your dataset ID. The response also includes `filename` and `target_column`.

Use a numeric target without missing values and enough rows for the train/test split and cross-validation. Feature types are inferred by pandas. Numeric features can contain missing values, but should have observed values available for median imputation. Include at least one numeric feature for the scaled preprocessing strategies. Upload validation checks CSV parsing and the presence of the target column; it does not validate all training requirements.

### 2. Start training

Send this JSON body to `POST /training/`, replacing the placeholder with the dataset ID:

```json
{
  "dataset_id": "<dataset-id>"
}
```

The response contains `job_id`, `dataset_id`, `target_column`, and an initial `status` of `queued`. Model selection and training settings are configured in Python rather than supplied in this request.

### 3. Check the job

Call `GET /training/{job_id}`. Status changes from `queued` to `running`, then to `completed` or `failed`.

The response includes:

- `job_id`, `dataset_id`, and `target_column`
- `status` and `model_id` (available after successful training)
- `created_at`, `started_at`, and `completed_at`
- `error_message` when training fails

Timestamps that have not been set are `null`. The endpoint does not currently expose per-trial progress or evaluation scores. The selected model's test score is stored in the database.

If a job remains queued, check the worker and Redis logs. If it fails, inspect `error_message` and the worker logs.

### 4. Make predictions

Send feature rows to `POST /models/{model_id}/predict` using the model ID from the completed job:

```json
{
  "rows": [
    {"area": 85.0, "rooms": 3, "city": "Berlin"},
    {"area": 120.0, "rooms": 4, "city": "Hamburg"}
  ]
}
```

Include the feature columns used during training; the target column is not required. The API loads the saved preprocessor and model and returns `model_id` plus a `predictions` array with one numeric value per row, in input order.

Unknown dataset or job IDs return HTTP 404 from the training endpoints. Prediction errors for unknown models or missing feature columns currently raise service exceptions without an HTTP error mapping.

## How training works

The pipeline in [ml/pipeline.py](ml/pipeline.py):

1. Reserves 20% of the dataset for testing.
2. Builds the parameter combinations for each model family, shuffles them with a fixed seed, and evaluates up to 25 configurations per family.
3. Scores each configuration using five-fold cross-validation on the remaining data.
4. Selects the configuration with the lowest mean RMSE and evaluates a separately fitted model on the held-out test set.
5. Fits a final model from the full input dataset and saves it with its preprocessor.

Each call to the `Model.fit` wrapper, including the final fit, reserves 20% of its input for validation. Preprocessing is fitted only on that call's training portion. For the neural network, validation controls early stopping and selection of the best weights. The reported test score belongs to the evaluation model, before the final fit.

| Model | Implementation | Preprocessing |
| --- | --- | --- |
| Elastic net | scikit-learn `ElasticNet` | Standardized numeric features and one-hot categorical features |
| Gradient boosting | LightGBM `LGBMRegressor` | Numeric features and pandas categorical columns |
| Neural network | PyTorch MLP with categorical embeddings | Standardized numeric features and integer-encoded categories |

Numeric missing values are filled with training medians. Categorical missing values use `__MISSING__`. Unseen categories become all-zero one-hot encodings, missing native categories, or embedding index `0`, depending on the strategy.

The neural network uses AdamW and MSE loss. The search configuration allows up to 500 epochs with early-stopping patience of 20. It uses CUDA when available, otherwise CPU; the supplied Compose file does not configure GPU access. The fitted network is moved to CPU for prediction and serialization.

## Configuration

Application settings are read from environment variables or a root `.env` file:

| Variable | Purpose |
| --- | --- |
| `DATABASE_URL` | SQLAlchemy database connection URL |
| `CELERY_BROKER_URL` | Redis connection used to queue training tasks |
| `CELERY_RESULT_BACKEND` | Redis connection used for Celery task results |

All three are required. `.env.docker` supplies them for Compose; `.env.example` is currently empty.

Search settings live in [ml/config.py](ml/config.py):

| Setting | Default |
| --- | --- |
| Selection metric | `rmse` |
| Cross-validation folds | `5` |
| Test fraction | `0.2` |
| Validation fraction within each fit | `0.2` |
| Trials per model family | `25` |
| Random seed | `42` |

`MODEL_SEARCH_SPACES` and `FIT_SEARCH_SPACES` define the candidate parameters. `MODEL_FIXED_PARAMS` and `FIT_FIXED_PARAMS` define parameters shared across trials. Reduce the trial count or neural-network epoch limit for shorter development runs.

The metric registry contains RMSE, MAE, and R², but the search always minimizes the score. R² therefore requires a change to the selection logic before it can be used correctly. The training service currently records the metric name as `rmse` regardless of configuration.

## Local development

Use Python 3.13, matching the Docker image and CI. Create a virtual environment and install dependencies:

```sh
python -m venv .venv
```

Activate it with `.venv\Scripts\Activate.ps1` on PowerShell or `source .venv/bin/activate` on Linux/macOS, then run:

```sh
python -m pip install -r requirements.txt
```

For an API and worker running on your host, create a `.env` file pointing at reachable PostgreSQL and Redis services, for example:

```dotenv
DATABASE_URL=postgresql+psycopg2://postgres:postgres@localhost:5432/automl
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/1
```

The supplied Compose file does not publish PostgreSQL or Redis ports to the host. These localhost examples require separately available services or explicit port mappings.

Run the API and worker in separate terminals from the repository root:

```sh
python -m uvicorn app.main:app --reload
```

```sh
python -m celery -A worker.celery_app:celery_app worker --loglevel=info
```

Use the Docker setup to run the worker in Linux when developing on Windows.

### Tests

With dependencies installed and the three environment variables configured:

```sh
python -m pytest -v
```

The current suite tests dataset shapes, model output, neural-network fitting and prediction, preprocessing strategies, metric calculations, configuration generation, artifact-save delegation, and the health endpoint. It does not run a complete queued training job or validate database migrations.

The tests do not require live PostgreSQL or Redis services. For example, CI uses `DATABASE_URL=sqlite:///./test.db` alongside localhost Redis URLs. CI runs on Python 3.13 for pushes and pull requests.

## Project structure

```text
AutoML/
├── app/
│   ├── api/             # Dataset, training, and prediction routes
│   ├── core/            # Environment-based settings
│   ├── db/              # ORM models, sessions, and table creation
│   ├── schemas/         # Reserved package; request schemas live in routes
│   └── services/        # Dataset lookup, training, and artifact handling
├── ml/
│   ├── base.py          # Common model interface
│   ├── config.py        # Search spaces, fixed parameters, and metrics
│   ├── models.py        # Regressors and preprocessing/model wrapper
│   ├── pipeline.py      # Model selection, evaluation, and final fit
│   ├── preprocessing.py # Model-specific feature transformations
│   ├── search.py        # Configuration generation and cross-validation
│   ├── types.py         # Preprocessing strategy enum
│   ├── dataset.py       # Standalone PyTorch dataset helper
│   └── prediction.py    # Legacy prediction implementation
├── worker/              # Celery application and training task
├── tests/               # Unit and smoke tests
├── alembic/             # Historical database migrations
├── storage/
│   ├── datasets/        # Uploaded CSV files, named by dataset UUID
│   └── models/          # Model/preprocessor bundles, named by model UUID
├── .github/workflows/   # CI configuration
├── compose.yaml         # API, worker, PostgreSQL, and Redis services
├── Dockerfile
└── requirements.txt
```

The active prediction path is `app.services.model_service.predict_with_model` → `ml.models.Model.predict`. The legacy `ml/prediction.py` still references the removed `TabularModel` and is not used by the API. Neural-network training uses `TensorDataset` rather than the standalone helper in `ml/dataset.py`.

Artifacts are saved as `storage/models/<model-id>.joblib`; each bundle contains the model wrapper and fitted preprocessor. Dataset and model storage directories are ignored by Git.

## License

[MIT](LICENSE).
