"""
src/models/linear_regression.py
─────────────────────────────────────────────────────────────────────────────
Spark MLlib Linear Regression model for weekly demand forecasting.

Role       : Model A — interpretable baseline
Library    : pyspark.ml.regression.LinearRegression
Hyperparams: maxIter=100, regParam=0.1, elasticNetParam=0.0  (L2 / Ridge)
Persistence: s3a://ecommerce-lake/gold/models/linear_regression/

Pipeline structure:
    VectorAssembler  →  LinearRegression
"""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression

from .base_model import BaseSparkModel, FEATURES_VEC_COL, TARGET_COL

# MinIO destination path
MINIO_MODEL_PATH = "s3a://ecommerce-lake/gold/models/linear_regression/"


class SparkLinearRegressionModel(BaseSparkModel):
    """
    Linear Regression wrapper using Spark MLlib.

    Parameters
    ----------
    max_iter       : int   Maximum number of iterations (default 100)
    reg_param      : float Regularisation strength — L2 ridge (default 0.1)
    elastic_net    : float ElasticNet mixing param 0=L2, 1=L1 (default 0.0)
    """

    def __init__(
        self,
        max_iter: int = 100,
        reg_param: float = 0.1,
        elastic_net: float = 0.0,
    ) -> None:
        super().__init__()
        self.max_iter    = max_iter
        self.reg_param   = reg_param
        self.elastic_net = elastic_net

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, train_df: DataFrame) -> None:
        """
        Assemble feature vector and fit LinearRegression on train_df.

        Stores the fitted PipelineModel in self.model.

        Parameters
        ----------
        train_df : DataFrame
            Gold-layer training data (year <= 2016).
        """
        print("\n▶ [LinearRegression] Training...")
        print(f"   maxIter={self.max_iter} | regParam={self.reg_param} "
              f"| elasticNetParam={self.elastic_net}")

        lr = LinearRegression(
            featuresCol=FEATURES_VEC_COL,
            labelCol=TARGET_COL,
            predictionCol="prediction",
            maxIter=self.max_iter,
            regParam=self.reg_param,
            elasticNetParam=self.elastic_net,
        )

        pipeline = Pipeline(stages=[self.assembler, lr])
        self.model = pipeline.fit(train_df)

        # Log training summary metrics
        lr_model = self.model.stages[-1]
        summary  = lr_model.summary
        print(f"   Training RMSE : {summary.rootMeanSquaredError:.4f}")
        print(f"   Training R²   : {summary.r2:.4f}")
        print("   ✅ LinearRegression training complete.")

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, test_df: DataFrame) -> DataFrame:
        """
        Generate predictions on test_df.

        Returns a DataFrame with the original columns plus a 'prediction' column.
        """
        self._validate_trained()
        print("\n▶ [LinearRegression] Generating predictions on test set...")
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
        print(f"\n▶ [LinearRegression] Saving model to: {path}")
        self.model.write().overwrite().save(path)
        print("   ✅ Model saved.")
