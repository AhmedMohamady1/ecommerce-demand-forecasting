"""
src/feature_engineering/engineer.py
─────────────────────────────────────────────────────────────────────────────
Step E — Read Silver/weekly_sales, add all features, write Gold/features.

Data lake flow:
  Silver/weekly_sales  →  this module  →  Gold/features

Features added (in order):
  ① Temporal   : month, quarter, is_year_end (from year + week_of_year)
  ② Lag        : lag_1_week, lag_4_week, lag_52_week  (Window lag on weekly_sales)
  ③ Rolling    : rolling_4_week_avg, rolling_12_week_avg  (Window rowsBetween)
  ④ OHE        : store_ohe (10 dims), item_ohe (50 dims) as sparse ML vectors
  ⑤ Null drop  : rows missing any lag feature are dropped (first ~52 per series)

Gold Layer schema (final):
  store, item, year, week_of_year, abs_week,
  weekly_sales, week_has_holiday,
  month, quarter, is_year_end,
  lag_1_week, lag_4_week, lag_52_week,
  rolling_4_week_avg, rolling_12_week_avg,
  store_idx, item_idx,
  store_ohe (SparseVector), item_ohe (SparseVector)

Gold path: s3a://ecommerce-lake/gold/features/
Partitioned by: store

Train/test convention (applied downstream by ML Engineer):
  Train → year <= 2016  (~104,000 rows)
  Test  → year == 2017  (~26,000 rows)

Usage (standalone):
    python -m src.feature_engineering.engineer
"""

import os
import sys

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config.spark_config import get_spark

# ── Constants ────────────────────────────────────────────────────────────────
SILVER_WEEKLY_PATH = "s3a://ecommerce-lake/silver/weekly_sales/"
GOLD_PATH          = "s3a://ecommerce-lake/gold/features/"


# ── Step 1: Read Silver ───────────────────────────────────────────────────────

def read_silver_weekly(spark: SparkSession, path: str = SILVER_WEEKLY_PATH) -> DataFrame:
    print(f"\n▶ Reading Silver weekly layer from: {path}")
    df = spark.read.parquet(path)
    count = df.count()
    print(f"  ↳ Loaded {count:,} weekly rows")
    df.printSchema()
    return df


# ── Step 2: Temporal Features ─────────────────────────────────────────────────

def add_temporal_features(df: DataFrame) -> DataFrame:
    """
    Derive month, quarter, and is_year_end from (year, week_of_year).

    Strategy: compute the approximate start-of-week date by adding
    (week_of_year - 1) * 7 days to Jan 1 of that year, then extract
    month and quarter from that date. is_year_end = 1 for weeks 51–52.

    Also adds `abs_week` — a monotonically increasing integer across years
    used as the sort key for all Window functions.
    """
    print("\n▶ Adding temporal features (month, quarter, is_year_end, abs_week)...")

    df = (
        df
        # Absolute week index: enables correct cross-year ordering in Windows
        .withColumn(
            "abs_week",
            (F.col("year") - 2013) * 53 + F.col("week_of_year")
        )
        # Approximate date for the start of the week (good enough for month/quarter)
        .withColumn(
            "_week_start",
            F.date_add(
                F.make_date(F.col("year"), F.lit(1), F.lit(1)),
                (F.col("week_of_year") - 1) * 7
            )
        )
        .withColumn("month",       F.month(F.col("_week_start")))
        .withColumn("quarter",     F.quarter(F.col("_week_start")))
        .withColumn("is_year_end", (F.col("week_of_year") >= 51).cast("int"))
        .drop("_week_start")
    )

    print("  ↳ Added: month, quarter, is_year_end, abs_week")
    return df


# ── Step 3: Lag Features ──────────────────────────────────────────────────────

def add_lag_features(df: DataFrame) -> DataFrame:
    """
    Add lag features using PySpark Window functions.

    Window: partitioned by (store, item), ordered by abs_week (ascending).

    | Column        | Offset | Meaning                         |
    |---|---|---|
    | lag_1_week   | -1     | Previous week's sales           |
    | lag_4_week   | -4     | ~1 month ago (4 weeks)          |
    | lag_52_week  | -52    | Same week, prior year           |

    Rows where any lag is null (first 52 weeks per store/item pair)
    are dropped AFTER all features are added.
    """
    print("\n▶ Adding lag features (lag_1_week, lag_4_week, lag_52_week)...")

    w = Window.partitionBy("store", "item").orderBy("abs_week")

    df = (
        df
        .withColumn("lag_1_week",  F.lag("weekly_sales",  1).over(w))
        .withColumn("lag_4_week",  F.lag("weekly_sales",  4).over(w))
        .withColumn("lag_52_week", F.lag("weekly_sales", 52).over(w))
    )

    print("  ↳ Added: lag_1_week, lag_4_week, lag_52_week")
    return df


# ── Step 4: Rolling Features ──────────────────────────────────────────────────

def add_rolling_features(df: DataFrame) -> DataFrame:
    """
    Add rolling average features using Window.rowsBetween(-N, -1).

    The window looks at the N rows immediately BEFORE the current row,
    not including the current row — this prevents data leakage.

    | Column              | N  | Meaning               |
    |---|---|---|
    | rolling_4_week_avg  |  4 | Short-term trend      |
    | rolling_12_week_avg | 12 | Quarterly trend       |

    Window: partitioned by (store, item), ordered by abs_week.
    """
    print("\n▶ Adding rolling features (rolling_4_week_avg, rolling_12_week_avg)...")

    w = Window.partitionBy("store", "item").orderBy("abs_week")

    df = (
        df
        .withColumn(
            "rolling_4_week_avg",
            F.avg("weekly_sales").over(w.rowsBetween(-4, -1))
        )
        .withColumn(
            "rolling_12_week_avg",
            F.avg("weekly_sales").over(w.rowsBetween(-12, -1))
        )
    )

    print("  ↳ Added: rolling_4_week_avg, rolling_12_week_avg")
    return df


# ── Step 5: Drop Null Rows ────────────────────────────────────────────────────

def drop_lag_nulls(df: DataFrame) -> DataFrame:
    """
    Drop rows where lag_52_week is null.

    This removes the first 52 weeks from each (store, item) series —
    the minimum history needed to compute the yearly lag.
    Rows with lag_1 or lag_4 null but lag_52 non-null are extremely rare
    (only if data has gaps) and are also dropped for model safety.
    """
    print("\n▶ Dropping null lag rows (first 52 weeks per store/item)...")

    before = df.count()
    df_clean = df.dropna(subset=["lag_1_week", "lag_4_week", "lag_52_week",
                                  "rolling_4_week_avg", "rolling_12_week_avg"])
    after = df_clean.count()
    dropped = before - after

    print(f"  ↳ Dropped {dropped:,} rows (warmup period)")
    print(f"  ↳ Remaining rows: {after:,}")

    return df_clean


# ── Step 6: One-Hot Encoding ──────────────────────────────────────────────────

def add_ohe_features(df: DataFrame, spark: SparkSession) -> DataFrame:
    """
    One-hot encode store (10 values) and item (50 values) using
    Spark MLlib StringIndexer → OneHotEncoder Pipeline.

    Outputs:
      store_idx   : float index (0–9)
      item_idx    : float index (0–49)
      store_ohe   : SparseVector of size 9  (drop-last encoding)
      item_ohe    : SparseVector of size 49 (drop-last encoding)

    OHE vectors are used downstream by tree-based models via VectorAssembler.
    For Prophet / ARIMA (per-series models), these columns are not needed
    and can be ignored.
    """
    print("\n▶ Adding OHE features (store_ohe, item_ohe)...")

    # Cast store and item to string so StringIndexer can process them
    df = (
        df
        .withColumn("store_str", F.col("store").cast("string"))
        .withColumn("item_str",  F.col("item").cast("string"))
    )

    indexers = [
        StringIndexer(inputCol="store_str", outputCol="store_idx",
                      handleInvalid="keep"),
        StringIndexer(inputCol="item_str",  outputCol="item_idx",
                      handleInvalid="keep"),
    ]
    encoders = [
        OneHotEncoder(inputCol="store_idx", outputCol="store_ohe",
                      dropLast=True),
        OneHotEncoder(inputCol="item_idx",  outputCol="item_ohe",
                      dropLast=True),
    ]

    pipeline = Pipeline(stages=indexers + encoders)
    model    = pipeline.fit(df)
    df       = model.transform(df).drop("store_str", "item_str")

    print("  ↳ Added: store_idx, item_idx, store_ohe (size=9), item_ohe (size=49)")
    return df


# ── Step 7: Select final column order ────────────────────────────────────────

def select_gold_columns(df: DataFrame) -> DataFrame:
    """Order columns clearly for the Gold layer."""
    return df.select(
        # Keys
        "store", "item", "year", "week_of_year", "abs_week",
        # Target
        "weekly_sales",
        # Base features (from Silver)
        "week_has_holiday",
        # Temporal
        "month", "quarter", "is_year_end",
        # Lag
        "lag_1_week", "lag_4_week", "lag_52_week",
        # Rolling
        "rolling_4_week_avg", "rolling_12_week_avg",
        # OHE
        "store_idx", "item_idx", "store_ohe", "item_ohe",
    )


# ── Step 8: Write Gold ────────────────────────────────────────────────────────

def write_gold(df: DataFrame, path: str = GOLD_PATH) -> None:
    """
    Write the feature-engineered DataFrame to Gold zone, partitioned by store.

    Note: OHE SparseVector columns cannot be written as plain Parquet — they
    are stored using the MLlib Vector UDT serialization (Parquet-compatible).

    Parameters
    ----------
    df   : feature-engineered Gold DataFrame
    path : S3A destination path
    """
    print(f"\n▶ Writing Gold layer to: {path}")
    print(f"   Partitioned by: store")

    (
        df.write
        .mode("overwrite")
        .partitionBy("store")
        .parquet(path)
    )

    confirmed = df.sparkSession.read.parquet(path).count()

    print(f"\n✅ Gold layer written successfully.")
    print(f"   Rows confirmed in MinIO : {confirmed:,}")
    print(f"   Path                    : {path}")
    print(f"   Train rows (year<=2016) : use df.filter(F.col('year') <= 2016)")
    print(f"   Test  rows (year==2017) : use df.filter(F.col('year') == 2017)")


def print_gold_summary(df: DataFrame) -> None:
    """Print Gold layer stats."""
    print("\n▶ Gold layer schema:")
    df.printSchema()

    print("\n▶ Gold layer — weekly_sales statistics:")
    df.describe("weekly_sales", "lag_1_week", "lag_52_week",
                "rolling_4_week_avg", "rolling_12_week_avg").show()

    print("▶ Row count by year:")
    df.groupBy("year").count().orderBy("year").show()

    print("▶ Sample rows (store=1, item=1):")
    (
        df.filter((F.col("store") == 1) & (F.col("item") == 1))
        .orderBy("abs_week")
        .select("year", "week_of_year", "weekly_sales",
                "lag_1_week", "lag_52_week",
                "rolling_4_week_avg", "week_has_holiday")
        .show(10, truncate=False)
    )


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    spark = get_spark("Feature-Engineering-Gold")

    df = read_silver_weekly(spark)
    df = add_temporal_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = drop_lag_nulls(df)
    df = add_ohe_features(df, spark)
    df = select_gold_columns(df)

    print_gold_summary(df)
    write_gold(df)

    spark.stop()
