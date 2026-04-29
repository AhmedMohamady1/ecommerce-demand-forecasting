"""
src/ingestion/upload_to_minio.py
─────────────────────────────────────────────────────────────────────────────
Step B — Write the raw Spark DataFrame to MinIO Bronze zone as Parquet.

Destination: s3a://ecommerce-lake/bronze/raw_sales/

The write is idempotent (mode="overwrite"), so re-running is always safe.

Usage (standalone):
    python -m src.ingestion.upload_to_minio
"""

import os
import sys

from pyspark.sql import DataFrame

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config.spark_config import get_spark
from src.ingestion.ingest import read_raw_data, validate, print_validation_summary

# ── Constants ────────────────────────────────────────────────────────────────
BRONZE_PATH = "s3a://ecommerce-lake/bronze/raw_sales/"


# ── Core function ────────────────────────────────────────────────────────────

def upload_to_bronze(df: DataFrame, path: str = BRONZE_PATH) -> None:
    """
    Write a Spark DataFrame to the MinIO Bronze zone in Parquet format.

    Parameters
    ----------
    df   : validated raw Spark DataFrame
    path : S3A destination path (default: Bronze zone)
    """
    print(f"▶ Writing raw data to Bronze zone: {path}")

    (
        df.write
        .mode("overwrite")
        .parquet(path)
    )

    # Quick read-back count to confirm the write succeeded
    spark = df.sparkSession
    written_count = spark.read.parquet(path).count()

    print(f"✅ Bronze layer written successfully.")
    print(f"   Rows confirmed in MinIO: {written_count:,}")
    print(f"   Path: {path}")


# ── Entrypoint ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    spark = get_spark("Upload-Bronze-Layer")

    print("▶ Reading and validating raw data ...")
    df = read_raw_data(spark)
    results = validate(df)
    print_validation_summary(results)

    upload_to_bronze(df)

    spark.stop()
