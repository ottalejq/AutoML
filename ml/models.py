import joblib
import numpy as np
import torch
from lightgbm import LGBMRegressor
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ml.base import BaseModel
from ml.preprocessing import TabularPreprocessor
from ml.types import PreprocessingStrategy


class ElasticNetModel(BaseModel):
    preprocessing = PreprocessingStrategy.ONE_HOT_SCALED

    def __init__(self, **model_params):
        seed = model_params.pop("seed", 42)
        model_params.setdefault("random_state", seed)

        self.model = ElasticNet(**model_params)

    def fit(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        **fit_params,
    ):
        self.model.fit(
            X=X_train,
            y=y_train,
            **fit_params,
        )
        return self

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path):
        joblib.dump(self, path)

    @classmethod
    def load(cls, path):
        return joblib.load(path)


class GradientBoostingModel(BaseModel):
    preprocessing = PreprocessingStrategy.NATIVE_CATEGORICAL

    def __init__(self, **model_params):
        seed = model_params.pop("seed", 42)
        model_params.setdefault("random_state", seed)

        self.model = LGBMRegressor(**model_params)

    def fit(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        **fit_params,
    ):
        self.model.fit(
            X=X_train,
            y=y_train,
            eval_X=X_val,
            eval_y=y_val,
            categorical_feature="auto",
            **fit_params,
        )
        return self

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path):
        joblib.dump(self, path)

    @classmethod
    def load(cls, path):
        return joblib.load(path)


class TabularNetwork(nn.Module):
    def __init__(
        self,
        num_numeric,
        cat_cardinalities,
        embedding_dim,
        hidden_dim,
        n_layers,
    ):
        super().__init__()

        self.embeddings = nn.ModuleList([
            nn.Embedding(
                num_embeddings=cardinality,
                embedding_dim=embedding_dim,
            )
            for cardinality in cat_cardinalities
        ])

        input_dim = (
            num_numeric
            + len(cat_cardinalities) * embedding_dim
        )

        layers = []
        in_dim = input_dim

        for _ in range(n_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
            ])
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, 1))

        self.mlp = nn.Sequential(*layers)

    def forward(self, X_num, X_cat):
        parts = [X_num]

        parts.extend(
            embedding(X_cat[:, i])
            for i, embedding in enumerate(self.embeddings)
        )

        return self.mlp(
            torch.cat(parts, dim=1)
        )


class NeuralNetworkModel(BaseModel):
    preprocessing = PreprocessingStrategy.EMBEDDED_CATEGORICAL

    def __init__(self, **model_params):
        self.seed = model_params.get("seed", 42)

        torch.manual_seed(self.seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        self.model = TabularNetwork(
            num_numeric=model_params["num_numeric"],
            cat_cardinalities=model_params.get(
                "cat_cardinalities",
                [],
            ),
            embedding_dim=model_params.get(
                "embedding_dim",
                8,
            ),
            hidden_dim=model_params.get(
                "hidden_dim",
                128,
            ),
            n_layers=model_params.get(
                "n_layers",
                2,
            ),
        )

    def fit(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        **fit_params,
    ):
        epochs = fit_params.get("epochs", 100)
        batch_size = fit_params.get("batch_size", 64)
        learning_rate = fit_params.get(
            "learning_rate",
            1e-3,
        )
        weight_decay = fit_params.get(
            "weight_decay",
            1e-4,
        )
        patience = fit_params.get("patience", 20)
        seed = fit_params.get("seed", self.seed)

        device = torch.device(
            fit_params.get(
                "device",
                "cuda" if torch.cuda.is_available() else "cpu",
            )
        )

        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self.model.to(device)

        X_num_train, X_cat_train = X_train

        X_num_train = torch.as_tensor(
            np.array(X_num_train, dtype=np.float32, copy=True),
        )

        X_cat_train = torch.as_tensor(
            np.array(X_cat_train, dtype=np.int64, copy=True),
        )

        y_train = torch.as_tensor(
            np.array(y_train, dtype=np.float32, copy=True),
        ).reshape(-1, 1)

        generator = torch.Generator()
        generator.manual_seed(seed)

        loader = DataLoader(
            TensorDataset(
                X_num_train,
                X_cat_train,
                y_train,
            ),
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
        )

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        loss_fn = nn.MSELoss()

        use_validation = (
            X_val is not None
            and y_val is not None
        )

        if use_validation:
            X_num_val, X_cat_val = X_val

            X_num_val = torch.as_tensor(
                np.array(X_num_val, dtype=np.float32, copy=True),
            )

            X_cat_val = torch.as_tensor(
                np.array(X_cat_val, dtype=np.int64, copy=True),
            )

            y_val = torch.as_tensor(
                np.array(y_val, dtype=np.float32, copy=True),
            ).reshape(-1, 1)



        best_loss = float("inf")
        best_state = None
        no_improvement = 0

        for _ in range(epochs):
            self.model.train()

            for X_num_batch, X_cat_batch, y_batch in loader:
                X_num_batch = X_num_batch.to(device)
                X_cat_batch = X_cat_batch.to(device)
                y_batch = y_batch.to(device)

                optimizer.zero_grad()

                predictions = self.model(
                    X_num_batch,
                    X_cat_batch,
                )

                loss = loss_fn(
                    predictions,
                    y_batch,
                )

                loss.backward()
                optimizer.step()

            if not use_validation:
                continue

            self.model.eval()

            with torch.no_grad():
                val_predictions = self.model(
                    X_num_val,
                    X_cat_val,
                )

                val_loss = loss_fn(
                    val_predictions,
                    y_val,
                ).item()

            if val_loss < best_loss:
                best_loss = val_loss

                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.model.state_dict().items()
                }

                no_improvement = 0

            else:
                no_improvement += 1

                if no_improvement >= patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.model.to("cpu")

        return self

    def predict(self, X):
        X_num, X_cat = X

        X_num = torch.as_tensor(
            np.asarray(X_num),
            dtype=torch.float32,
        )

        X_cat = torch.as_tensor(
            np.asarray(X_cat),
            dtype=torch.long,
        )

        self.model.eval()

        with torch.no_grad():
            predictions = self.model(
                X_num,
                X_cat,
            )

        return predictions.squeeze(1).numpy()

    def save(self, path):
        joblib.dump(self, path)

    @classmethod
    def load(cls, path):
        return joblib.load(path)


class Model:
    def __init__(
        self,
        model_class,
        **model_params,
    ):
        self.preprocessor = TabularPreprocessor(
            model_class.preprocessing
        )

        self.model_class = model_class
        self.model_params = model_params
        self.model = None

    def fit(
        self,
        X,
        y,
        val_size=0.2,
        seed=42,
        **fit_params,
    ):
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=val_size,
            random_state=seed,
        )

        X_train = self.preprocessor.fit_transform(X_train)
        X_val = self.preprocessor.transform(X_val)

        model_params = {
            **self.model_params,
            **self.preprocessor.get_model_params(),
        }

        self.model = self.model_class(
            **model_params,
        )

        self.model.fit(
            X_train,
            y_train,
            X_val,
            y_val,
            **fit_params,
        )

        return self

    def predict(self, X):
        if self.model is None:
            raise RuntimeError("Model has not been fitted.")

        X = self.preprocessor.transform(X)

        return self.model.predict(X)

    def save(self, path):
        joblib.dump(self, path)

    @classmethod
    def load(cls, path):
        return joblib.load(path)