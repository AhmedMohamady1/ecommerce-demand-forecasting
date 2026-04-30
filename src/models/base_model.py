"""
src/models/base_model.py
─────────────────────────────────────────────────────────────────────────────
Abstract base class and shared constants for all Spark MLlib models.

Gold layer columns (written by src/feature_engineering/engineer.py):
    store, item, year, week_of_year, abs_week,
    weekly_sales,           ← TARGET
    week_has_holiday,
    month, quarter, is_year_end,
    lag_1_week, lag_4_week, lag_52_week,
    rolling_4_week_avg, rolling_12_week_avg,
    store_idx, item_idx,    ← EXCLUDED (per project spec)
    store_ohe, item_ohe     ← used as SparseVector features

Excluded from training features (per project spec):
    year, abs_week, store_idx, item_idx

Feature columns used for VectorAssembler (66 dims total):
    Scalar  (10): week_of_year, week_has_holiday, month, quarter, is_year_end,
                  lag_1_week, lag_4_week, lag_52_week,
                  rolling_4_week_avg, rolling_12_week_avg
    store_ohe   :  9-dim SparseVector  (drop-last OHE of 10 stores)
    item_ohe    : 49-dim SparseVector  (drop-last OHE of 50 items)
    ─────────────────────────────────────────────────────────────
    Total       : 68 dims

Target column : weekly_sales
"""

from __future__ import annotations
from abc import ABC, abstractmethod

from pyspark.sql import DataFrame
from pyspark.ml.feature import VectorAssembler


# ── Shared feature/target column definitions ──────────────────────────────────

TARGET_COL = "weekly_sales"

# Scalar feature columns (excluded: year, abs_week, store_idx, item_idx)
SCALAR_FEATURE_COLS = [
    "week_of_year",
    "week_has_holiday",
    "month",
    "quarter",
    "is_year_end",
    "lag_1_week",
    "lag_4_week",
    "lag_52_week",
    "rolling_4_week_avg",
    "rolling_12_week_avg",
]

# OHE SparseVector columns produced by engineer.py
OHE_FEATURE_COLS = [
    "store_ohe",   # size=9  (drop-last of 10 stores)
    "item_ohe",    # size=49 (drop-last of 50 items)
]

# All feature columns fed to VectorAssembler
FEATURE_COLS = SCALAR_FEATURE_COLS + OHE_FEATURE_COLS

# Combined feature vector column name (Spark MLlib convention)
FEATURES_VEC_COL = "features"


# ── Shared VectorAssembler factory ────────────────────────────────────────────

def build_assembler() -> VectorAssembler:
    """
    Return a VectorAssembler that merges SCALAR_FEATURE_COLS + OHE_FEATURE_COLS
    into a single 'features' DenseVector / SparseVector column.

    handleInvalid='keep' prevents crashes on rare NaN/Inf values.
    """
    return VectorAssembler(
        inputCols=FEATURE_COLS,
        outputCol=FEATURES_VEC_COL,
        handleInvalid="keep",
    )


# ── Abstract base class ───────────────────────────────────────────────────────

class BaseSparkModel(ABC):
    """
    Abstract interface for all Spark MLlib regression models in this project.

    Subclasses must implement:
        train(train_df)  → fit the model, store self.model
        predict(test_df) → return a DataFrame with 'prediction' column
        save(path)       → persist the fitted model to MinIO (s3a://)
    """

    def __init__(self) -> None:
        self.model = None          # set by train()
        self.assembler = build_assembler()

    @abstractmethod
    def train(self, train_df: DataFrame) -> None:
        """Fit the model on train_df. Stores trained model in self.model."""
        ...

    @abstractmethod
    def predict(self, test_df: DataFrame) -> DataFrame:
        """
        Transform test_df through the trained model pipeline.

        Returns
        -------
        DataFrame with at least columns: weekly_sales, prediction
        """
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Persist the fitted model to the given s3a:// or local path.

        Parameters
        ----------
        path : str
            Destination path, e.g. 's3a://ecommerce-lake/gold/models/linear_regression/'
        """
        ...

    def _assemble(self, df: DataFrame) -> DataFrame:
        """Convenience helper: run VectorAssembler on a DataFrame."""
        return self.assembler.transform(df)

    def _validate_trained(self) -> None:
        """Raise RuntimeError if train() has not been called yet."""
        if self.model is None:
            raise RuntimeError(
                f"{self.__class__.__name__}.train() must be called before predict() or save()."
            )
