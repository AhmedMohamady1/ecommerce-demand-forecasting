"""
scripts/export_to_csv.py
─────────────────────────────────────────────────────────────────────────────
Export all three MinIO data lake layers to local CSV files for data analysis
and visualisation (EDA notebooks, pandas-based exploration, etc.).

Reads from MinIO via PySpark + S3A and writes to data/processed/:

  data/processed/
  ├── daily_sales.csv              ← Silver/daily  (913,000 rows)
  │   Columns: date, store, item, sales, is_holiday
  │
  ├── weekly_sales.csv             ← Silver/weekly (131,000 rows)
  │   Columns: store, item, year, week_of_year, weekly_sales, week_has_holiday
  │
  └── weekly_sales_engineered.csv  ← Gold/features (105,000 rows)
      Columns: store, item, year, week_of_year, abs_week,
               weekly_sales, week_has_holiday,
               month, quarter, is_year_end,
               lag_1_week, lag_4_week, lag_52_week,
               rolling_4_week_avg, rolling_12_week_avg,
               store_idx, item_idx
      Note: OHE SparseVector columns (store_ohe, item_ohe) are excluded —
            use store_idx / item_idx for label-encoded integer representations.

Prerequisites:
  - MinIO must be running (bash scripts/start_minio_docker.sh)
  - Full pipeline must have been run at least once
  - .env must be present

Usage:
    python scripts/export_to_csv.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pyspark.sql import functions as F
from config.spark_config import get_spark

# ── Paths ─────────────────────────────────────────────────────────────────────
SILVER_DAILY_PATH  = "s3a://ecommerce-lake/silver/daily_sales/"
SILVER_WEEKLY_PATH = "s3a://ecommerce-lake/silver/weekly_sales/"
GOLD_PATH          = "s3a://ecommerce-lake/gold/features/"

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def export_layer(spark, s3a_path: str, out_path: str,
                 sort_cols: list, drop_cols: list = None,
                 label: str = "") -> None:
    """
    Read a Parquet layer from MinIO, optionally drop columns, sort, and write
    to a single CSV file.

    Parameters
    ----------
    spark      : active SparkSession
    s3a_path   : MinIO S3A path to read Parquet from
    out_path   : local filesystem path to write the CSV to
    sort_cols  : columns to orderBy before writing (for reproducible row order)
    drop_cols  : columns to exclude from the CSV (e.g. SparseVector columns)
    label      : display name for progress messages
    """
    print(f"\n▶ Exporting {label}")
    print(f"   Source : {s3a_path}")
    print(f"   Dest   : {out_path}")

    df = spark.read.parquet(s3a_path)

    if drop_cols:
        df = df.drop(*drop_cols)

    df = df.orderBy(*sort_cols)
    row_count = df.count()

    # Convert to pandas and write CSV (single file, no Spark partitioning)
    df.toPandas().to_csv(out_path, index=False)

    print(f"   ✅ {row_count:,} rows → {os.path.basename(out_path)}")


if __name__ == "__main__":
    # ── Setup ────────────────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    spark = get_spark("Export-to-CSV")

    print("=" * 62)
    print("  MinIO → local CSV export")
    print("=" * 62)

    # ── 1. Silver / daily_sales ───────────────────────────────────────────────
    export_layer(
        spark,
        s3a_path=SILVER_DAILY_PATH,
        out_path=os.path.join(OUTPUT_DIR, "daily_sales.csv"),
        sort_cols=["date", "store", "item"],
        label="Silver / daily_sales → daily_sales.csv",
    )

    # ── 2. Silver / weekly_sales ──────────────────────────────────────────────
    export_layer(
        spark,
        s3a_path=SILVER_WEEKLY_PATH,
        out_path=os.path.join(OUTPUT_DIR, "weekly_sales.csv"),
        sort_cols=["store", "item", "year", "week_of_year"],
        label="Silver / weekly_sales → weekly_sales.csv",
    )

    # ── 3. Gold / features (exclude SparseVector OHE columns) ─────────────────
    export_layer(
        spark,
        s3a_path=GOLD_PATH,
        out_path=os.path.join(OUTPUT_DIR, "weekly_sales_engineered.csv"),
        sort_cols=["store", "item", "year", "week_of_year"],
        # SparseVector columns can't be meaningfully serialised to CSV;
        # store_idx / item_idx carry the same information as plain floats.
        drop_cols=["store_ohe", "item_ohe"],
        label="Gold / features → weekly_sales_engineered.csv",
    )

    print("\n" + "=" * 62)
    print(f"  Export complete → {os.path.abspath(OUTPUT_DIR)}/")
    print("=" * 62)
    print("\n  Files written:")
    for f in ["daily_sales.csv", "weekly_sales.csv", "weekly_sales_engineered.csv"]:
        fpath = os.path.join(OUTPUT_DIR, f)
        size_mb = os.path.getsize(fpath) / 1_048_576
        print(f"    {f:<40} {size_mb:>6.1f} MB")

    spark.stop()
