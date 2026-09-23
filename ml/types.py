from enum import Enum


class PreprocessingStrategy(Enum):
    ONE_HOT_SCALED = "one_hot_scaled"
    NATIVE_CATEGORICAL = "native_categorical"
    EMBEDDED_CATEGORICAL = "embedded_categorical"
