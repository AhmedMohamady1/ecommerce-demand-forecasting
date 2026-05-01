"""
src/models/train_evaluate.py
─────────────────────────────────────────────────────────────────────────────
ML Engineer orchestration script — Steps 7.2 → 7.5

Workflow:
    1. Load Gold-layer Parquet from MinIO (all 10 store partitions)
    2. Chronological train/test split
         Train → year <= 2016  (~104,000 rows)
         Test  → year == 2017  (~26,000 rows)
    3. Train all five models
         A. Linear Regression  (Spark MLlib)
         B. Random Forest      (Spark MLlib)
         C. Gradient Boosting  (Spark MLlib)
         D. ARIMA(1,1,1)       (statsmodels — per store/item pair, parallel)
         E. Prophet            (facebook/prophet — per store/item pair, parallel)
    4. Generate predictions and compute metrics (RMSE, MAE, MAPE, R²)
    5. Save all models to MinIO Gold zone
    6. Save metrics to results/metrics/model_comparison.csv + MinIO

Usage:
    # From project root:
    python -m src.models.train_evaluate

    # Or directly:
    python src/models/train_evaluate.py

MinIO paths:
    Input  : s3a://ecommerce-lake/gold/features/
    Models : s3a://ecommerce-lake/gold/models/<model_name>/
    Metrics: s3a://ecommerce-lake/gold/metrics/model_comparison.csv
"""

from __future__ import annotations

import os
import sys
import time

# ── Make project root importable when run as __main__ ─────────────────────────
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from pyspark.sql import functions as F

from config.spark_config import get_spark
from src.models.linear_regression  import SparkLinearRegressionModel
from src.models.random_forest      import SparkRandomForestModel
from src.models.gradient_boosting  import SparkGBTModel
from src.models.arima_model        import ArimaModel
from src.models.prophet_model      import ProphetModel
from src.evaluation.metrics        import (
    compute_spark_metrics,
    compute_pandas_metrics,
    save_metrics,
)

# ── Paths ─────────────────────────────────────────────────────────────────────

GOLD_FEATURES_PATH = "s3a://ecommerce-lake/gold/features/"

MODEL_PATHS = {
    "linear_regression": "s3a://ecommerce-lake/gold/models/linear_regression/",
    "random_forest":     "s3a://ecommerce-lake/gold/models/random_forest/",
    "gradient_boosting": "s3a://ecommerce-lake/gold/models/gradient_boosting/",
}

# ── Step 7.2 — Train/Test Split ───────────────────────────────────────────────

def load_and_split(spark):
    """
    Read the Gold-layer Parquet (all 10 store partitions) and split
    chronologically into train (year <= 2016) and test (year == 2017).

    Returns
    -------
    (full_df, train_df, test_df)
    """
    print("\n" + "=" * 65)
    print("  STEP 7.2 — TRAIN / TEST SPLIT")
    print("=" * 65)
    print(f"▶ Reading Gold features from: {GOLD_FEATURES_PATH}")

    df = spark.read.parquet(GOLD_FEATURES_PATH)

    total = df.count()
    print(f"   ↳ Total Gold rows: {total:,}")
    df.printSchema()

    # ── Chronological split ────────────────────────────────────────────────
    train_df = df.filter(F.col("year") <= 2016)
    test_df  = df.filter(F.col("year") == 2017)

    train_count = train_df.count()
    test_count  = test_df.count()

    print(f"\n   Train set: year ≤ 2016 → {train_count:,} rows")
    print(f"   Test  set: year = 2017  → {test_count:,} rows")
    print(f"   Split ratio: {train_count / total * 100:.1f}% / {test_count / total * 100:.1f}%")

    # Cache the splits — they are reused by all three Spark models
    train_df.cache()
    test_df.cache()
    # Trigger materialisation
    train_df.count()
    test_df.count()
    print("\n   ✅ DataFrames cached and splits confirmed.")

    return df, train_df, test_df


# ── Helper: time a model step ─────────────────────────────────────────────────

class _Timer:
    def __init__(self, label: str):
        self.label = label
        self._start = None

    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, *_):
        elapsed = time.time() - self._start
        print(f"\n   ⏱  {self.label} — completed in {elapsed:.1f}s")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 65)
    print("  E-COMMERCE DEMAND FORECASTING — ML PIPELINE")
    print("  Steps 7.2 → 7.5: Split → Train → Persist → Evaluate")
    print("=" * 65)

    # ── 1. Initialise Spark ────────────────────────────────────────────────
    spark = get_spark("ECommerceForecasting")

    # ── 2. Load & split ───────────────────────────────────────────────────
    full_df, train_df, test_df = load_and_split(spark)

    # Accumulate metric dicts across all models
    all_metrics = []

    # =========================================================================
    # STEP 7.3 — MODEL TRAINING
    # =========================================================================
    print("\n" + "=" * 65)
    print("  STEP 7.3 — MODEL TRAINING")
    print("=" * 65)

    # ── Model A: Linear Regression ────────────────────────────────────────
    with _Timer("Linear Regression"):
        lr_model = SparkLinearRegressionModel(
            max_iter=100, reg_param=0.1, elastic_net=0.0
        )
        lr_model.train(train_df)

    # ── Model B: Random Forest ─────────────────────────────────────────────
    with _Timer("Random Forest"):
        rf_model = SparkRandomForestModel(
            num_trees=100, max_depth=10, feature_subset_strategy="auto", seed=42
        )
        rf_model.train(train_df)

    # ── Model C: Gradient Boosting ─────────────────────────────────────────
    with _Timer("Gradient Boosting (GBT)"):
        gbt_model = SparkGBTModel(
            max_iter=50, max_depth=5, step_size=0.1, seed=42
        )
        gbt_model.train(train_df)

    # ── Model D: ARIMA ─────────────────────────────────────────────────────
    with _Timer("ARIMA (all store/item pairs)"):
        arima_model = ArimaModel(order=(1, 1, 1), n_jobs=-1)
        arima_model.prepare_data(train_df, test_df)
        arima_model.train()

    # ── Model E: Prophet ───────────────────────────────────────────────────
    with _Timer("Prophet (all store/item pairs)"):
        prophet_model = ProphetModel(n_jobs=-1)
        prophet_model.prepare_data(train_df, test_df)
        prophet_model.train()

    # =========================================================================
    # STEP 7.4 — MODEL PERSISTENCE
    # =========================================================================
    print("\n" + "=" * 65)
    print("  STEP 7.4 — MODEL PERSISTENCE")
    print("=" * 65)

    lr_model.save(MODEL_PATHS["linear_regression"])
    rf_model.save(MODEL_PATHS["random_forest"])
    gbt_model.save(MODEL_PATHS["gradient_boosting"])
    arima_model.save()      # uses default MINIO_OBJECT_KEY
    prophet_model.save()    # uses default MINIO_OBJECT_KEY

    print("\n✅ All models persisted to MinIO Gold zone.")

    # =========================================================================
    # STEP 7.5 — EVALUATION
    # =========================================================================
    print("\n" + "=" * 65)
    print("  STEP 7.5 — EVALUATION")
    print("=" * 65)

    # ── Spark model predictions ────────────────────────────────────────────
    lr_preds  = lr_model.predict(test_df)
    rf_preds  = rf_model.predict(test_df)
    gbt_preds = gbt_model.predict(test_df)

    all_metrics.append(
        compute_spark_metrics(lr_preds,  model_name="LinearRegression")
    )
    all_metrics.append(
        compute_spark_metrics(rf_preds,  model_name="RandomForest")
    )
    all_metrics.append(
        compute_spark_metrics(gbt_preds, model_name="GradientBoosting")
    )

    # ── ARIMA / Prophet predictions ────────────────────────────────────────
    arima_preds  = arima_model.predict()
    prophet_preds = prophet_model.predict()

    all_metrics.append(
        compute_pandas_metrics(arima_preds,   model_name="ARIMA")
    )
    all_metrics.append(
        compute_pandas_metrics(prophet_preds, model_name="Prophet")
    )

    # ── Save metrics ───────────────────────────────────────────────────────
    save_metrics(all_metrics)

    print("\n" + "=" * 65)
    print("  PIPELINE COMPLETE")
    print("=" * 65)
    print("  Models → s3a://ecommerce-lake/gold/models/")
    print("  Metrics → results/metrics/model_comparison.csv")
    print("           s3a://ecommerce-lake/gold/metrics/model_comparison.csv")

    # ── Unpersist cached DFs ───────────────────────────────────────────────
    train_df.unpersist()
    test_df.unpersist()

    spark.stop()


if __name__ == "__main__":
    main()
