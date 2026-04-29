"""
config/spark_config.py
─────────────────────────────────────────────────────────────────────────────
Centralized SparkSession factory for the E-Commerce Demand Forecasting project.

Reads MinIO credentials from environment variables (loaded from .env).
Configures Spark to use the S3A filesystem driver pointed at the local
MinIO instance, so all s3a:// paths resolve to the MinIO data lake.

Usage:
    from config.spark_config import get_spark
    spark = get_spark()
"""

import os
from dotenv import load_dotenv
from pyspark.sql import SparkSession

# Load .env from project root (two levels up from this file)
_env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(dotenv_path=_env_path)


def get_spark(app_name: str = "EcommerceDemandForecasting") -> SparkSession:
    """
    Build and return a SparkSession configured for local mode + MinIO S3A access.

    The session is a singleton — calling this function multiple times returns
    the existing session without creating a new one.

    Parameters
    ----------
    app_name : str
        Display name shown in the Spark UI.

    Returns
    -------
    SparkSession
    """
    minio_endpoint = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
    access_key = os.getenv("MINIO_ACCESS_KEY")
    secret_key = os.getenv("MINIO_SECRET_KEY")

    if not access_key or not secret_key:
        raise EnvironmentError(
            "MINIO_ACCESS_KEY and MINIO_SECRET_KEY must be set in your .env file."
        )

    spark = (
        SparkSession.builder
        .master("local[*]")
        .appName(app_name)
        # ── S3A JARs (required for PySpark 4.x — not bundled by default) ─────
        # hadoop-aws MUST match the Hadoop version bundled in the PySpark wheel:
        #   PySpark 3.5.4  →  Hadoop 3.3.4  →  hadoop-aws 3.3.4
        # Using a newer hadoop-aws (3.3.5+) causes NoClassDefFoundError
        # because it references classes only added to hadoop-common in 3.3.5+.
        .config(
            "spark.jars.packages",
            "org.apache.hadoop:hadoop-aws:3.3.4,"
            "com.amazonaws:aws-java-sdk-bundle:1.12.262"
        )
        # ── S3A / MinIO ──────────────────────────────────────────────────────
        .config("spark.hadoop.fs.s3a.endpoint", minio_endpoint)
        .config("spark.hadoop.fs.s3a.access.key", access_key)
        .config("spark.hadoop.fs.s3a.secret.key", secret_key)
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config(
            "spark.hadoop.fs.s3a.aws.credentials.provider",
            "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider"
        )
        # ── Performance ──────────────────────────────────────────────────────
        .config("spark.driver.memory", "8g")
        .config("spark.sql.shuffle.partitions", "20")
        .config("spark.sql.autoBroadcastJoinThreshold", str(10 * 1024 * 1024))
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        # ── Parquet ──────────────────────────────────────────────────────────
        .config("spark.sql.parquet.compression.codec", "snappy")
        .getOrCreate()
    )

    # Suppress verbose INFO logs — keep output readable
    spark.sparkContext.setLogLevel("WARN")

    return spark
