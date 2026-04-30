"""
src/evaluation/metrics.py
─────────────────────────────────────────────────────────────────────────────
Evaluation helpers — RMSE, MAE, MAPE, R² — for all five forecasting models.

Functions:
    compute_spark_metrics(predictions_df, label_col, pred_col)
        → dict  — for Spark MLlib models (LR, RF, GBT)

    compute_pandas_metrics(y_true, y_pred, model_name)
        → dict  — for statsmodels/Prophet predictions (Pandas arrays)

    save_metrics(results_list, local_csv_path, minio_object_key)
        → None  — writes CSV locally + uploads to MinIO Gold zone

Metric definitions:
    RMSE  = sqrt(mean((y - ŷ)²))
    MAE   = mean(|y - ŷ|)
    MAPE  = mean(|y - ŷ| / max(|y|, ε)) * 100   (ε avoids div-by-zero)
    R²    = 1 - SS_res / SS_tot

Usage:
    from src.evaluation.metrics import compute_spark_metrics, compute_pandas_metrics, save_metrics
"""

from __future__ import annotations

import io
import os
from typing import Dict, List, Optional

import boto3
import numpy as np
import pandas as pd

# ── Constants ────────────────────────────────────────────────────────────────

MINIO_BUCKET          = "ecommerce-lake"
DEFAULT_MINIO_OBJ_KEY = "gold/metrics/model_comparison.csv"
DEFAULT_LOCAL_CSV     = os.path.join("results", "metrics", "model_comparison.csv")

EPSILON = 1e-6   # prevents division-by-zero in MAPE


# ── Spark metrics ─────────────────────────────────────────────────────────────

def compute_spark_metrics(
    predictions_df,
    label_col: str = "weekly_sales",
    pred_col:  str = "prediction",
    model_name: str = "spark_model",
) -> Dict[str, float]:
    """
    Compute RMSE, MAE, MAPE, and R² from a Spark predictions DataFrame.

    The DataFrame is collected to Pandas for metric computation — this is
    safe because the test set is ~26k rows.

    Parameters
    ----------
    predictions_df : pyspark.sql.DataFrame
        Output of model.transform(test_df), must contain label_col and pred_col.
    label_col  : str   Column name for ground-truth values.
    pred_col   : str   Column name for model predictions.
    model_name : str   Label used in the returned dict.

    Returns
    -------
    dict with keys: model, rmse, mae, mape, r2
    """
    print(f"\n▶ [Metrics] Evaluating {model_name} (Spark)...")

    pdf = predictions_df.select(label_col, pred_col).toPandas()
    y_true = pdf[label_col].values.astype(float)
    y_pred = pdf[pred_col].values.astype(float)

    metrics = _compute_metrics(y_true, y_pred, model_name)
    _print_metrics(metrics)
    return metrics


# ── Pandas metrics ────────────────────────────────────────────────────────────

def compute_pandas_metrics(
    predictions_df: pd.DataFrame,
    label_col: str = "weekly_sales",
    pred_col:  str = "prediction",
    model_name: str = "model",
) -> Dict[str, float]:
    """
    Compute RMSE, MAE, MAPE, and R² from a Pandas predictions DataFrame.

    Used for ARIMA and Prophet whose predictions are assembled in Pandas.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        Must contain label_col and pred_col.
    label_col  : str   Ground-truth column name.
    pred_col   : str   Prediction column name.
    model_name : str   Label used in the returned dict.

    Returns
    -------
    dict with keys: model, rmse, mae, mape, r2
    """
    print(f"\n▶ [Metrics] Evaluating {model_name} (Pandas)...")

    # Drop rows where prediction is NaN (fallback rows)
    clean = predictions_df[[label_col, pred_col]].dropna()
    y_true = clean[label_col].values.astype(float)
    y_pred = clean[pred_col].values.astype(float)

    metrics = _compute_metrics(y_true, y_pred, model_name)
    _print_metrics(metrics)
    return metrics


# ── Internal computation ───────────────────────────────────────────────────────

def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
) -> Dict[str, float]:
    """Core metric computation given plain numpy arrays."""
    residuals = y_true - y_pred
    ss_res    = np.sum(residuals ** 2)
    ss_tot    = np.sum((y_true - np.mean(y_true)) ** 2)

    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    mae  = float(np.mean(np.abs(residuals)))
    mape = float(np.mean(np.abs(residuals) / np.maximum(np.abs(y_true), EPSILON)) * 100)
    r2   = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    return {
        "model": model_name,
        "rmse":  round(rmse, 4),
        "mae":   round(mae,  4),
        "mape":  round(mape, 4),
        "r2":    round(r2,   4),
        "n_rows": int(len(y_true)),
    }


def _print_metrics(metrics: Dict) -> None:
    """Print a formatted metric summary."""
    print(f"   Model : {metrics['model']}")
    print(f"   Rows  : {metrics['n_rows']:,}")
    print(f"   RMSE  : {metrics['rmse']:.4f}")
    print(f"   MAE   : {metrics['mae']:.4f}")
    print(f"   MAPE  : {metrics['mape']:.4f}%")
    print(f"   R²    : {metrics['r2']:.4f}")


# ── Save metrics ──────────────────────────────────────────────────────────────

def save_metrics(
    results_list: List[Dict],
    local_csv_path: str = DEFAULT_LOCAL_CSV,
    minio_object_key: str = DEFAULT_MINIO_OBJ_KEY,
) -> None:
    """
    Save a list of metric dicts to:
        1. A local CSV file  (results/metrics/model_comparison.csv)
        2. MinIO Gold zone   (s3a://ecommerce-lake/gold/metrics/model_comparison.csv)

    Parameters
    ----------
    results_list     : list of dicts returned by compute_*_metrics()
    local_csv_path   : local path to write CSV
    minio_object_key : MinIO object key under the ecommerce-lake bucket
    """
    df = pd.DataFrame(results_list)

    # ── Print comparison table ────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  MODEL COMPARISON RESULTS")
    print("=" * 65)
    print(df.to_string(index=False))
    print("=" * 65)

    # ── Write local CSV ───────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(local_csv_path), exist_ok=True)
    df.to_csv(local_csv_path, index=False)
    print(f"\n✅ Metrics saved locally → {local_csv_path}")

    # ── Upload to MinIO ───────────────────────────────────────────────────────
    minio_endpoint = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
    access_key     = os.getenv("MINIO_ACCESS_KEY")
    secret_key     = os.getenv("MINIO_SECRET_KEY")

    try:
        s3_client = boto3.client(
            "s3",
            endpoint_url=minio_endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )

        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_bytes = csv_buffer.getvalue().encode("utf-8")

        s3_client.put_object(
            Bucket=MINIO_BUCKET,
            Key=minio_object_key,
            Body=csv_bytes,
            ContentType="text/csv",
        )
        print(f"✅ Metrics uploaded to MinIO → s3a://{MINIO_BUCKET}/{minio_object_key}")

    except Exception as exc:
        print(f"⚠ MinIO upload failed (metrics still saved locally): {exc}")
