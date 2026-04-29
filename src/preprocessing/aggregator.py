"""
src/preprocessing/aggregator.py
─────────────────────────────────────────────────────────────────────────────
Step D — Read the Silver daily layer, aggregate daily → weekly, write the
Silver weekly layer.

Data lake flow:
  Bronze  → raw daily data (as-is)
  Silver/daily_sales   → cleaned daily rows + is_holiday  (~913k rows)
  Silver/weekly_sales  → weekly aggregated rows            (~130k rows)  ← THIS FILE
  Gold                 → Silver/weekly + lag/rolling/OHE   (Role 3)

Aggregation logic:
  Group by (store, item, year, week_of_year):
    weekly_sales     = SUM(sales)      ← target variable
    week_has_holiday = MAX(is_holiday) ← 1 if any day in that week was a holiday

Silver weekly schema:
  store            IntegerType
  item             IntegerType
  year             IntegerType
  week_of_year     IntegerType   (1–52)
  weekly_sales     LongType      (sum of daily sales)
  week_has_holiday IntegerType   (0 or 1)

Expected output:
  ~500 store/item pairs × ~261 weeks ≈ 130,000 rows

Silver weekly path: s3a://ecommerce-lake/silver/weekly_sales/
Partitioned by: store

Usage (standalone):
    python -m src.preprocessing.aggregator
"""

import os
import sys

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config.spark_config import get_spark

# ── Constants ────────────────────────────────────────────────────────────────
SILVER_DAILY_PATH  = "s3a://ecommerce-lake/silver/daily_sales/"
SILVER_WEEKLY_PATH = "s3a://ecommerce-lake/silver/weekly_sales/"


# ── Core functions ───────────────────────────────────────────────────────────

def read_silver_daily(spark: SparkSession, path: str = SILVER_DAILY_PATH) -> DataFrame:
    """
    Read the Silver daily layer from MinIO.

    Parameters
    ----------
    spark : active SparkSession
    path  : S3A path to Silver daily layer

    Returns
    -------
    Daily enriched DataFrame with columns: date, store, item, sales, is_holiday
    """
    print(f"\n▶ Reading Silver daily layer from: {path}")
    df = spark.read.parquet(path)
    count = df.count()
    print(f"  ↳ Loaded {count:,} daily rows")
    df.printSchema()
    return df


def aggregate_to_weekly(df: DataFrame) -> DataFrame:
    """
    Aggregate the daily Silver DataFrame to the weekly Gold level.

    Weeks are defined by Spark's weekofyear() function (ISO week numbers,
    1–52). The week_has_holiday flag is collapsed via MAX so that a week is
    marked 1 if any of its 7 days was a US public holiday.

    Parameters
    ----------
    df : Silver daily DataFrame (date, store, item, sales, is_holiday)

    Returns
    -------
    Weekly Gold DataFrame:
        store, item, year, week_of_year, weekly_sales, week_has_holiday
    """
    print("\n▶ Aggregating daily → weekly ...")

    df_weekly = (
        df
        # Extract temporal keys
        .withColumn("year",         F.year(F.col("date")))
        .withColumn("week_of_year", F.weekofyear(F.col("date")))
        # Aggregate per (store, item, year, week)
        .groupBy("store", "item", "year", "week_of_year")
        .agg(
            F.sum("sales").alias("weekly_sales"),
            F.max("is_holiday").alias("week_has_holiday"),
        )
        # Readable column order
        .select(
            "store", "item", "year", "week_of_year",
            "weekly_sales", "week_has_holiday"
        )
        .orderBy("store", "item", "year", "week_of_year")
    )

    weekly_count = df_weekly.count()
    print(f"  ↳ Weekly rows produced : {weekly_count:,}  (expected ~130,000)")

    return df_weekly


def write_silver_weekly(df_weekly: DataFrame, path: str = SILVER_WEEKLY_PATH) -> None:
    """
    Write the weekly aggregated DataFrame to MinIO Silver weekly zone.

    Partitioned by store (10 partitions). The weekly Silver layer is the
    direct input to Role 3 feature engineering, which will add lag features,
    rolling averages, and OHE before writing the final Gold layer.

    Parameters
    ----------
    df_weekly : weekly aggregated DataFrame
    path      : S3A destination path
    """
    print(f"\n▶ Writing Silver weekly layer to: {path}")
    print(f"   Schema     : store, item, year, week_of_year, weekly_sales, week_has_holiday")
    print(f"   Partitioned: store (10 partitions)")

    (
        df_weekly.write
        .mode("overwrite")
        .partitionBy("store")
        .parquet(path)
    )

    spark = df_weekly.sparkSession
    confirmed_count = spark.read.parquet(path).count()

    print(f"\n✅ Silver weekly layer written successfully.")
    print(f"   Rows confirmed in MinIO : {confirmed_count:,}")
    print(f"   Partitioned by          : store (10 partitions)")
    print(f"   Path                    : {path}")
    print(f"   Next step               : Role 3 feature engineering → Gold")


def print_silver_weekly_summary(df_weekly: DataFrame) -> None:
    """Print descriptive statistics for the weekly Silver DataFrame."""
    print("\n▶ Silver weekly — weekly_sales statistics:")
    df_weekly.describe("weekly_sales").show()

    print("▶ week_has_holiday distribution:")
    df_weekly.groupBy("week_has_holiday").count().orderBy("week_has_holiday").show()

    print("▶ Sample rows:")
    df_weekly.show(10, truncate=False)


# ── Entrypoint ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    spark = get_spark("Aggregate-Silver-Weekly")

    df_silver_daily = read_silver_daily(spark)
    df_weekly       = aggregate_to_weekly(df_silver_daily)

    print_silver_weekly_summary(df_weekly)
    write_silver_weekly(df_weekly)

    spark.stop()
