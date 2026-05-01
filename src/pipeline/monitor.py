"""
src/pipeline/monitor.py
─────────────────────────────────────────────────────────────────
Drift detection and performance tracking.
"""

import pandas as pd
import os

MONITORING_LOG = "results/metrics/monitoring_log.csv"
COMPARISON_CSV = "results/metrics/model_comparison.csv"
DRIFT_THRESHOLD = 0.10   # alert if RMSE degrades > 10%
NULL_THRESHOLD  = 0.01   # alert if > 1% null rate in new data

def get_baseline_rmse(model_name: str) -> float:
    try:
        df = pd.read_csv(COMPARISON_CSV)
        row = df[df['model'] == model_name]
        if not row.empty:
            return float(row.iloc[0]['rmse'])
    except Exception:
        pass
    return None


def check_performance_drift(model_name: str, current_rmse: float) -> bool:
    """Alert if current RMSE is > 10% worse than the training baseline."""
    baseline = get_baseline_rmse(model_name)
    if baseline is None:
        return False
    degradation = (current_rmse - baseline) / baseline
    if degradation > DRIFT_THRESHOLD:
        print(f"⚠️  DRIFT ALERT: {model_name} RMSE degraded {degradation:.1%} "
              f"(baseline={baseline}, current={current_rmse:.1f})")
        return True
    return False

def check_missing_data(df: pd.DataFrame) -> bool:
    """Alert if null rate exceeds 1% in any column."""
    null_rate = df.isnull().mean()
    bad_cols = null_rate[null_rate > NULL_THRESHOLD]
    if not bad_cols.empty:
        print(f"⚠️  MISSING DATA: columns with >1% nulls: {bad_cols.to_dict()}")
        return True
    return False


def log_monitoring_result(model: str, rmse: float, mae: float, has_drift: bool):
    """Append one monitoring check result to the log CSV."""
    import datetime
    record = {"timestamp": datetime.datetime.now().isoformat(),
              "model": model, "rmse": rmse, "mae": mae, "drift_detected": has_drift}
    df = pd.DataFrame([record])
    write_header = not os.path.exists(MONITORING_LOG)
    df.to_csv(MONITORING_LOG, mode="a", header=write_header, index=False)
    print(f"📝 Monitoring log updated → {MONITORING_LOG}")

def run_evaluation_merge(predictions_df: pd.DataFrame) -> tuple:
    """
    Connect to MinIO Data Lake to fetch actual ground-truth weekly sales,
    merge them with the provided predictions, and calculate RMSE.
    
    Returns:
        tuple: (current_rmse, merged_df) or (None, None) if no matches found.
    """
    from config.spark_config import get_spark
    from sklearn.metrics import mean_squared_error
    import numpy as np
    
    # 1. Get Actuals via PySpark from Silver Bucket
    spark = get_spark("StreamlitMonitor")
    silver_spark = spark.read.parquet("s3a://ecommerce-lake/silver/weekly_sales/")
    silver_df = silver_spark.select("year", "week_of_year", "store", "item", "weekly_sales").toPandas()
    silver_df = silver_df.rename(columns={"week_of_year": "week"})
    
    # 2. Merge Predictions with Actuals
    merged = pd.merge(predictions_df, silver_df, on=["year", "week", "store", "item"], how="inner")
    
    if merged.empty:
        return None, merged
        
    current_rmse = float(np.sqrt(mean_squared_error(merged["weekly_sales"], merged["sales_prediction"])))
    return current_rmse, merged
