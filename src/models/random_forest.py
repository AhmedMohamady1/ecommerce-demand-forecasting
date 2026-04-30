"""
src/models/random_forest.py
─────────────────────────────────────────────────────────────────────────────
Spark MLlib Random Forest Regressor for weekly demand forecasting.

Role       : Model B — non-linear interactions + built-in feature importance
Library    : pyspark.ml.regression.RandomForestRegressor
Hyperparams: numTrees=100, maxDepth=10, featureSubsetStrategy="auto"
Persistence: s3a://ecommerce-lake/gold/models/random_forest/

Pipeline structure:
    VectorAssembler  →  RandomForestRegressor
"""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.ml import Pipeline
from pyspark.ml.regression import RandomForestRegressor

from .base_model import BaseSparkModel, FEATURES_VEC_COL, TARGET_COL, FEATURE_COLS

# MinIO destination path
MINIO_MODEL_PATH = "s3a://ecommerce-lake/gold/models/random_forest/"


class SparkRandomForestModel(BaseSparkModel):
    """
    Random Forest Regressor wrapper using Spark MLlib.

    Parameters
    ----------
    num_trees              : int  Number of trees in the forest (default 100)
    max_depth              : int  Maximum depth of each tree (default 10)
    feature_subset_strategy: str  Features sampled per split (default "auto")
    seed                   : int  Random seed for reproducibility (default 42)
    """

    def __init__(
        self,
        num_trees: int = 100,
        max_depth: int = 10,
        feature_subset_strategy: str = "auto",
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.num_trees               = num_trees
        self.max_depth               = max_depth
        self.feature_subset_strategy = feature_subset_strategy
        self.seed                    = seed

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, train_df: DataFrame) -> None:
        """
        Assemble feature vector and fit RandomForestRegressor on train_df.

        Parameters
        ----------
        train_df : DataFrame
            Gold-layer training data (year <= 2016).
        """
        print("\n▶ [RandomForest] Training...")
        print(f"   numTrees={self.num_trees} | maxDepth={self.max_depth} "
              f"| featureSubsetStrategy={self.feature_subset_strategy}")

        rf = RandomForestRegressor(
            featuresCol=FEATURES_VEC_COL,
            labelCol=TARGET_COL,
            predictionCol="prediction",
            numTrees=self.num_trees,
            maxDepth=self.max_depth,
            featureSubsetStrategy=self.feature_subset_strategy,
            seed=self.seed,
        )

        pipeline   = Pipeline(stages=[self.assembler, rf])
        self.model = pipeline.fit(train_df)

        # Log feature importances (top 10)
        rf_model     = self.model.stages[-1]
        importances  = rf_model.featureImportances.toArray()
        top_n        = min(10, len(importances))
        # Map flat importances back to feature names where possible
        # OHE vectors expand into multiple slots; show the scalar features first
        scalar_names = [
            "week_of_year", "week_has_holiday", "month", "quarter", "is_year_end",
            "lag_1_week", "lag_4_week", "lag_52_week",
            "rolling_4_week_avg", "rolling_12_week_avg",
        ]
        print(f"\n   Top {top_n} feature importances (scalar features only):")
        for i, name in enumerate(scalar_names[:top_n]):
            print(f"     [{i:2d}] {name:<25s}: {importances[i]:.6f}")

        print("   ✅ RandomForest training complete.")

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, test_df: DataFrame) -> DataFrame:
        """
        Generate predictions on test_df.

        Returns a DataFrame with original columns plus 'prediction'.
        """
        self._validate_trained()
        print("\n▶ [RandomForest] Generating predictions on test set...")
        predictions = self.model.transform(test_df)
        print("   ✅ Predictions generated.")
        return predictions

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str = MINIO_MODEL_PATH) -> None:
        """
        Persist the fitted PipelineModel to MinIO (overwrite mode).

        Parameters
        ----------
        path : str
            s3a:// destination; defaults to the Gold models path.
        """
        self._validate_trained()
        print(f"\n▶ [RandomForest] Saving model to: {path}")
        self.model.write().overwrite().save(path)
        print("   ✅ Model saved.")
