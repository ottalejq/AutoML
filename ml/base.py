from abc import ABC, abstractmethod
from pathlib import Path
from typing import Self


from abc import ABC, abstractmethod
from pathlib import Path
from typing import Self


from abc import ABC, abstractmethod
from pathlib import Path
from typing import Self


class BaseModel(ABC):
    @abstractmethod
    def __init__(self, **model_params):
        ...

    @abstractmethod
    def fit(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        **fit_params,
    ) -> Self:
        ...

    @abstractmethod
    def predict(self, X):
        ...

    @abstractmethod
    def save(self, path: Path) -> None:
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> Self:
        ...