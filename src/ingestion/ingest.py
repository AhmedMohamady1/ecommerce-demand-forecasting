"""
src/ingestion/ingest.py
─────────────────────────────────────────────────────────────────────────────
Step A — Read raw_data.csv into Spark with an explicit schema and validate it.

Returns a validated Spark DataFrame ready for the preprocessing pipeline.

Usage (standalone):
    python -m src.ingestion.ingest
"""

import os
import sys
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, DateType, IntegerType

# Allow running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config.spark_config import get_spark

# ── Constants ────────────────────────────────────────────────────────────────
RAW_CSV_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "raw_data.csv"
)

EXPECTED_ROW_COUNT = 913_000          # approximate
DATE_MIN = "2013-01-01"
DATE_MAX = "2017-12-31"
STORE_MIN, STORE_MAX = 1, 10
ITEM_MIN,  ITEM_MAX  = 1, 50

RAW_SCHEMA = StructType([
    StructField("date",  DateType(),    nullable=False),
    StructField("store", IntegerType(), nullable=False),
    StructField("item",  IntegerType(), nullable=False),
    StructField("sales", IntegerType(), nullable=False),
])


# ── Core function ────────────────────────────────────────────────────────────

def read_raw_data(spark: SparkSession, csv_path: str = RAW_CSV_PATH) -> DataFrame:
    """
    Read raw_data.csv into a Spark DataFrame using the explicit schema defined
    in the project plan (date, store, item, sales).

    Parameters
    ----------
    spark    : active SparkSession
    csv_path : path to raw_data.csv (absolute or relative to project root)

    Returns
    -------
    DataFrame with columns: date (DateType), store, item, sales (IntegerType)
    """
    csv_path = os.path.abspath(csv_path)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Raw CSV not found: {csv_path}")

    df = (
        spark.read
        .option("header", "true")
        .option("dateFormat", "yyyy-MM-dd")
        .schema(RAW_SCHEMA)
        .csv(csv_path)
    )

    return df


def validate(df: DataFrame) -> dict:
    """
    Run all data quality checks described in the plan (Step A, point 3).

    Checks:
      1. Row count is within ±5 % of 913,000
      2. No nulls in any column
      3. sales >= 0
      4. store in [1, 10]
      5. item  in [1, 50]
      6. date  in [2013-01-01, 2017-12-31]

    Parameters
    ----------
    df : raw Spark DataFrame

    Returns
    -------
    dict with keys:
        row_count       (int)
        null_counts     (dict col→count)
        negative_sales  (int)
        bad_store       (int)
        bad_item        (int)
        bad_date        (int)
        passed          (bool)
    """
    row_count = df.count()

    # Null counts per column
    null_counts = (
        df.select([
            F.sum(F.col(c).isNull().cast("int")).alias(c)
            for c in df.columns
        ])
        .collect()[0]
        .asDict()
    )

    agg = df.agg(
        F.sum((F.col("sales") < 0).cast("int")).alias("negative_sales"),
        F.sum(((F.col("store") < STORE_MIN) | (F.col("store") > STORE_MAX)).cast("int")).alias("bad_store"),
        F.sum(((F.col("item")  < ITEM_MIN)  | (F.col("item")  > ITEM_MAX)).cast("int")).alias("bad_item"),
        F.sum(((F.col("date")  < F.lit(DATE_MIN).cast(DateType())) |
               (F.col("date")  > F.lit(DATE_MAX).cast(DateType()))).cast("int")).alias("bad_date"),
    ).collect()[0]

    row_count_ok   = abs(row_count - EXPECTED_ROW_COUNT) / EXPECTED_ROW_COUNT < 0.05
    nulls_ok       = all(v == 0 for v in null_counts.values())
    sales_ok       = agg["negative_sales"] == 0
    store_ok       = agg["bad_store"] == 0
    item_ok        = agg["bad_item"]  == 0
    date_ok        = agg["bad_date"]  == 0

    passed = all([row_count_ok, nulls_ok, sales_ok, store_ok, item_ok, date_ok])

    return {
        "row_count":      row_count,
        "null_counts":    null_counts,
        "negative_sales": agg["negative_sales"],
        "bad_store":      agg["bad_store"],
        "bad_item":       agg["bad_item"],
        "bad_date":       agg["bad_date"],
        "passed":         passed,
    }


def print_validation_summary(results: dict) -> None:
    """Pretty-print the validation report to stdout."""
    status = "✅ PASSED" if results["passed"] else "❌ FAILED"
    print("\n" + "═" * 60)
    print(f"  VALIDATION SUMMARY  {status}")
    print("═" * 60)
    print(f"  Row count          : {results['row_count']:,}  (expected ~{EXPECTED_ROW_COUNT:,})")
    print(f"  Null counts        : {results['null_counts']}")
    print(f"  Negative sales     : {results['negative_sales']}")
    print(f"  Out-of-range store : {results['bad_store']}")
    print(f"  Out-of-range item  : {results['bad_item']}")
    print(f"  Out-of-range date  : {results['bad_date']}")
    print("═" * 60 + "\n")

    if not results["passed"]:
        raise ValueError("Data validation failed — see summary above.")


# ── Entrypoint ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    spark = get_spark("Ingest-Raw-Data")

    print("▶ Reading raw_data.csv ...")
    df = read_raw_data(spark)
    df.printSchema()

    print("▶ Running validation checks ...")
    results = validate(df)
    print_validation_summary(results)

    print("▶ Sample rows:")
    df.show(5, truncate=False)

    spark.stop()
