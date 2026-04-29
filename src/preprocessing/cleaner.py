"""
src/preprocessing/cleaner.py
─────────────────────────────────────────────────────────────────────────────
Step C — Clean the raw DataFrame, enrich with the US holiday flag, and write
the Silver layer to MinIO.

Operations (in order):
  1. Cast date → DateType  (safe-guard; schema already enforces this)
  2. Drop duplicate rows
  3. Remove rows where sales < 0
  4. Load US Holiday Dates CSV → deduplicate to one row per date → is_holiday = 1
  5. Left-join holidays onto daily sales; fill nulls → is_holiday = 0
  6. Log all removal / enrichment counts
  7. Write daily enriched DataFrame to Silver zone, partitioned by store

Silver layer schema (one row per day per store/item):
  date           DateType
  store          IntegerType
  item           IntegerType
  sales          IntegerType
  is_holiday     IntegerType  (0 or 1)

Usage (standalone):
    python -m src.preprocessing.cleaner
"""

import os
import sys

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DateType, IntegerType

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config.spark_config import get_spark
from src.ingestion.ingest import read_raw_data, validate, print_validation_summary

# ── Constants ────────────────────────────────────────────────────────────────
HOLIDAYS_CSV_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data",
    "US Holiday Dates (2004-2021).csv"
)

SILVER_PATH = "s3a://ecommerce-lake/silver/daily_sales/"


# ── Helper: load holidays ────────────────────────────────────────────────────

def load_holiday_dates(spark: SparkSession, csv_path: str = HOLIDAYS_CSV_PATH) -> DataFrame:
    """
    Read the US Holiday Dates CSV and return a deduplicated DataFrame with
    one column:
        holiday_date  DateType

    The CSV has one row per (date, holiday_name) combination — a single
    calendar date can appear multiple times (e.g. Christmas Eve AND Christmas
    in the same week). We deduplicate to one row per date so the join does
    not create duplicate rows in the sales DataFrame.

    CSV schema: Date, Holiday, WeekDay, Month, Day, Year  (343 rows)
    """
    csv_path = os.path.abspath(csv_path)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Holidays CSV not found: {csv_path}")

    holidays_raw = (
        spark.read
        .option("header", "true")
        .csv(csv_path)
    )

    # Keep only the Date column, cast to DateType, drop duplicates
    holidays = (
        holidays_raw
        .select(F.to_date(F.col("Date"), "yyyy-MM-dd").alias("holiday_date"))
        .dropDuplicates(["holiday_date"])
        .withColumn("is_holiday", F.lit(1).cast(IntegerType()))
    )

    holiday_count = holidays.count()
    print(f"  ↳ Loaded {holiday_count} unique holiday dates from CSV")

    return holidays


# ── Core function ────────────────────────────────────────────────────────────

def clean_and_enrich(df: DataFrame, spark: SparkSession) -> DataFrame:
    """
    Clean the raw daily sales DataFrame and enrich each row with an
    is_holiday flag sourced from the US Holiday Dates CSV.

    Parameters
    ----------
    df    : raw Spark DataFrame (date, store, item, sales)
    spark : active SparkSession (needed to load holidays CSV)

    Returns
    -------
    Spark DataFrame with columns:
        date, store, item, sales, is_holiday
    """
    original_count = df.count()
    print(f"\n▶ Cleaning  — starting row count: {original_count:,}")

    # ── 1. Ensure date column is DateType ────────────────────────────────────
    if dict(df.dtypes).get("date") != "date":
        df = df.withColumn("date", F.col("date").cast(DateType()))
        print("  ↳ Cast 'date' column to DateType")

    # ── 2. Drop duplicate rows ───────────────────────────────────────────────
    df_deduped = df.dropDuplicates()
    dup_count = original_count - df_deduped.count()
    print(f"  ↳ Dropped {dup_count:,} duplicate rows")
    df = df_deduped

    # ── 3. Remove rows with sales < 0 ────────────────────────────────────────
    df_positive = df.filter(F.col("sales") >= 0)
    neg_count = df.count() - df_positive.count()
    print(f"  ↳ Removed {neg_count:,} rows with negative sales")
    df = df_positive

    # ── 4. Load holiday dates ────────────────────────────────────────────────
    print("▶ Enriching — joining US Holiday dates ...")
    holidays = load_holiday_dates(spark)

    # ── 5. Left-join and fill nulls with 0 ──────────────────────────────────
    df_enriched = (
        df
        .join(holidays, df["date"] == holidays["holiday_date"], how="left")
        .drop("holiday_date")
        .withColumn(
            "is_holiday",
            F.coalesce(F.col("is_holiday"), F.lit(0)).cast(IntegerType())
        )
        # Ensure clean column order
        .select("date", "store", "item", "sales", "is_holiday")
    )

    holiday_day_count = df_enriched.filter(F.col("is_holiday") == 1).count()
    final_count = df_enriched.count()
    print(f"  ↳ {holiday_day_count:,} daily rows flagged as holiday days")
    print(f"▶ Cleaning complete — final row count: {final_count:,}")

    return df_enriched


def write_silver(df_enriched: DataFrame, path: str = SILVER_PATH) -> None:
    """
    Write the cleaned & enriched daily DataFrame to MinIO Silver zone,
    partitioned by store (10 partitions — one per store value).

    Storing the full daily granularity in Silver preserves maximum flexibility
    for downstream steps (EDA, custom aggregations, etc.) before feature
    engineering aggregates to the weekly level in Gold.

    Parameters
    ----------
    df_enriched : daily cleaned DataFrame with is_holiday column
    path        : S3A destination path
    """
    print(f"\n▶ Writing Silver layer to: {path}")
    print(f"   Schema: date, store, item, sales, is_holiday")
    print(f"   Partitioned by: store")

    (
        df_enriched.write
        .mode("overwrite")
        .partitionBy("store")
        .parquet(path)
    )

    spark = df_enriched.sparkSession
    confirmed_count = spark.read.parquet(path).count()

    print(f"\n✅ Silver layer written successfully.")
    print(f"   Rows confirmed in MinIO : {confirmed_count:,}")
    print(f"   Partitions              : 10 (one per store)")
    print(f"   Path                    : {path}")


# ── Entrypoint ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    spark = get_spark("Cleaner-Silver-Layer")

    print("▶ Reading raw data ...")
    df_raw = read_raw_data(spark)
    results = validate(df_raw)
    print_validation_summary(results)

    df_clean = clean_and_enrich(df_raw, spark)

    print("\n▶ Sample rows (clean + enriched with is_holiday):")
    df_clean.orderBy("date", "store", "item").show(15, truncate=False)

    print("\n▶ is_holiday distribution:")
    df_clean.groupBy("is_holiday").count().orderBy("is_holiday").show()

    write_silver(df_clean)

    spark.stop()
