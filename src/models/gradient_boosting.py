"""
src/models/gradient_boosting.py
─────────────────────────────────────────────────────────────────────────────
Spark MLlib Gradient-Boosted Tree (GBT) Regressor for weekly demand forecasting.

Role       : Model C — highest accuracy; captures complex non-linear patterns
Library    : pyspark.ml.regression.GBTRegressor
Hyperparams: maxIter=50, maxDepth=5, stepSize=0.1
Persistence: s3a://ecommerce-lake/gold/models/gradient_boosting/

Pipeline structure:
    VectorAssembler  →  GBTRegressor

Note: GBT does NOT support MulticlassClassificationEvaluator or RegressionEvaluator
      training summary — evaluation is done externally via src/evaluation/metrics.py.
"""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.ml import Pipeline
from pyspark.ml.regression import GBTRegressor

from .base_model import BaseSparkModel, FEATURES_VEC_COL, TARGET_COL

# MinIO destination path
MINIO_MODEL_PATH = "s3a://ecommerce-lake/gold/models/gradient_boosting/"


class SparkGBTModel(BaseSparkModel):
    """
    Gradient-Boosted Tree Regressor wrapper using Spark MLlib.

    Parameters
    ----------
    max_iter  : int   Number of boosting iterations (default 50)
    max_depth : int   Max depth per tree (default 5; GBT prefers shallow trees)
    step_size : float Learning rate / shrinkage (default 0.1)
    seed      : int   Random seed for reproducibility (default 42)
    """

    def __init__(
        self,
        max_iter: int = 50,
        max_depth: int = 5,
        step_size: float = 0.1,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.max_iter  = max_iter
        self.max_depth = max_depth
        self.step_size = step_size
        self.seed      = seed

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, train_df: DataFrame) -> None:
        """
        Assemble feature vector and fit GBTRegressor on train_df.

        Parameters
        ----------
        train_df : DataFrame
            Gold-layer training data (year <= 2016).
        """
        print("\n▶ [GradientBoosting] Training...")
        print(f"   maxIter={self.max_iter} | maxDepth={self.max_depth} "
              f"| stepSize={self.step_size}")

        gbt = GBTRegressor(
            featuresCol=FEATURES_VEC_COL,
            labelCol=TARGET_COL,
            predictionCol="prediction",
            maxIter=self.max_iter,
            maxDepth=self.max_depth,
            stepSize=self.step_size,
            seed=self.seed,
        )

        pipeline   = Pipeline(stages=[self.assembler, gbt])
        self.model = pipeline.fit(train_df)

        # Log feature importances (scalar portion — top 10)
        gbt_model   = self.model.stages[-1]
        importances = gbt_model.featureImportances.toArray()
        scalar_names = [
            "week_of_year", "week_has_holiday", "month", "quarter", "is_year_end",
            "lag_1_week", "lag_4_week", "lag_52_week",
            "rolling_4_week_avg", "rolling_12_week_avg",
        ]
        top_n = min(10, len(scalar_names))
        print(f"\n   Top {top_n} feature importances (scalar features):")
        for i, name in enumerate(scalar_names[:top_n]):
            print(f"     [{i:2d}] {name:<25s}: {importances[i]:.6f}")

        print("   ✅ GBT training complete.")

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, test_df: DataFrame) -> DataFrame:
        """
        Generate predictions on test_df.

        Returns a DataFrame with original columns plus 'prediction'.
        """
        self._validate_trained()
        print("\n▶ [GradientBoosting] Generating predictions on test set...")
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
        print(f"\n▶ [GradientBoosting] Saving model to: {path}")
        self.model.write().overwrite().save(path)
        print("   ✅ Model saved.")
