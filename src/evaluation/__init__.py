"""
src/evaluation/__init__.py
─────────────────────────────────────────────────────────────────────────────
Evaluation package for E-Commerce Demand Forecasting.

Provides:
    compute_spark_metrics  — RMSE, MAE, MAPE, R² from a Spark predictions DF
    compute_pandas_metrics — same metrics from numpy arrays (ARIMA / Prophet)
    save_metrics           — writes CSV locally and uploads to MinIO Gold zone
"""
