"""
src/models/arima_model.py
─────────────────────────────────────────────────────────────────────────────
ARIMA model for weekly demand forecasting — fitted per (store, item) pair.

Role       : Model D — classical time-series baseline; captures autocorrelation
Library    : statsmodels.tsa.arima.model.ARIMA
Order      : ARIMA(1, 1, 1)  — one autoregressive, one differencing, one MA term
Strategy   : Group Spark DataFrame by (store, item), collect to Pandas per group,
             fit ARIMA on the training series, forecast the test period.
Parallelism: joblib.Parallel across all 500 store/item pairs.
Persistence: pickle dict  →  s3a://ecommerce-lake/gold/models/arima/arima_models.pkl
             (uploaded via boto3)

Data contract:
    Columns collected from Spark: store, item, year, week_of_year, weekly_sales
    Excluded (per project spec, never collected): abs_week, store_idx, item_idx
    Chronological order is preserved by sorting on (year, week_of_year).
    The train/test split is chronological: train → year<=2016, test → year==2017.

Output:
    predictions_df : pandas DataFrame with columns
        store, item, week_of_year, year, weekly_sales, prediction
"""

from __future__ import annotations

import io
import os
import pickle
import sys
import warnings
from typing import Dict, List, Tuple

import boto3
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from statsmodels.tsa.arima.model import ARIMA

# Suppress harmless statsmodels convergence warnings in parallel runs
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ── Constants ────────────────────────────────────────────────────────────────

ARIMA_ORDER       = (1, 1, 1)
N_JOBS            = -1            # use all CPU cores
MINIO_BUCKET      = "ecommerce-lake"
MINIO_OBJECT_KEY  = "gold/models/arima/arima_models.pkl"

# Columns collected from Spark — abs_week is intentionally excluded (project spec).
# Chronological ordering uses (year, week_of_year) instead.
NEEDED_COLS = ["store", "item", "year", "week_of_year", "weekly_sales"]


# ── Per-pair ARIMA fit/forecast ───────────────────────────────────────────────

def _fit_predict_pair(
    key: Tuple[int, int],
    train_series: pd.Series,
    test_len: int,
) -> Tuple[Tuple[int, int], object, List[float]]:
    """
    Fit ARIMA on train_series, forecast test_len steps ahead.

    Parameters
    ----------
    key          : (store, item) identifier
    train_series : weekly_sales ordered chronologically (train period)
    test_len     : number of test weeks to forecast

    Returns
    -------
    (key, fitted_model_result, forecast_values)
    """
    try:
        model  = ARIMA(train_series, order=ARIMA_ORDER)
        result = model.fit()
        fc     = result.forecast(steps=test_len).tolist()
    except Exception as exc:
        # Fall back to naive forecast (last observed value) if ARIMA fails
        last_val = float(train_series.iloc[-1]) if len(train_series) > 0 else 0.0
        fc       = [last_val] * test_len
        result   = None
        print(f"   ⚠ ARIMA failed for store={key[0]}, item={key[1]}: {exc} "
              f"— using naive fallback")
    return key, result, fc


# ── Main ARIMA class ──────────────────────────────────────────────────────────

class ArimaModel:
    """
    ARIMA forecasting across all (store, item) pairs in parallel.

    Parameters
    ----------
    order   : ARIMA order (p, d, q) — default (1, 1, 1)
    n_jobs  : joblib parallelism — default -1 (all cores)
    """

    def __init__(
        self,
        order: Tuple[int, int, int] = ARIMA_ORDER,
        n_jobs: int = N_JOBS,
    ) -> None:
        self.order          = order
        self.n_jobs         = n_jobs
        self.fitted_models: Dict[Tuple[int, int], object] = {}
        self._train_pdf: pd.DataFrame | None = None
        self._test_pdf:  pd.DataFrame | None = None

    # ── Data preparation ──────────────────────────────────────────────────────

    def prepare_data(
        self,
        train_spark_df,   # pyspark.sql.DataFrame
        test_spark_df,    # pyspark.sql.DataFrame
    ) -> None:
        """
        Collect train and test Spark DataFrames to Pandas for per-pair processing.

        Only the columns required for ARIMA are collected to minimise driver memory.
        """
        print("\n▶ [ARIMA] Collecting data from Spark to Pandas...")

        self._train_pdf = (
            train_spark_df
            .select(NEEDED_COLS)
            .orderBy("store", "item", "year", "week_of_year")
            .toPandas()
        )
        self._test_pdf = (
            test_spark_df
            .select(NEEDED_COLS)
            .orderBy("store", "item", "year", "week_of_year")
            .toPandas()
        )

        n_pairs = self._train_pdf[["store", "item"]].drop_duplicates().shape[0]
        print(f"   ↳ Train rows: {len(self._train_pdf):,}")
        print(f"   ↳ Test  rows: {len(self._test_pdf):,}")
        print(f"   ↳ Unique (store, item) pairs: {n_pairs}")

    # ── Training (parallel ARIMA fits) ────────────────────────────────────────

    def train(self) -> None:
        """
        Fit ARIMA(p,d,q) for every (store, item) pair in parallel.

        Requires prepare_data() to have been called first.
        """
        if self._train_pdf is None:
            raise RuntimeError("Call prepare_data() before train().")

        print(f"\n▶ [ARIMA] Fitting ARIMA{self.order} for all (store, item) pairs ...")
        print(f"   Using {self.n_jobs} parallel jobs (joblib)...")

        # Build list of (key, train_series) tuples
        groups = list(
            self._train_pdf
            .groupby(["store", "item"])["weekly_sales"]
        )

        # Build test lengths per pair
        test_lens = {
            (row["store"], row["item"]): len(grp)
            for (row_key, grp) in [
                ((s, i), g)
                for (s, i), g in self._test_pdf.groupby(["store", "item"])["weekly_sales"]
            ]
            for row in [{"store": row_key[0], "item": row_key[1]}]
        }
        # Simpler rebuild
        test_lens = {}
        for (s, i), grp in self._test_pdf.groupby(["store", "item"])["weekly_sales"]:
            test_lens[(s, i)] = len(grp)

        results = Parallel(n_jobs=self.n_jobs, verbose=5)(
            delayed(_fit_predict_pair)(
                key=(s, i),
                train_series=series.values,
                test_len=test_lens.get((s, i), 52),
            )
            for (s, i), series in groups
        )

        # Store fitted models and forecasts
        self._forecasts: Dict[Tuple[int, int], List[float]] = {}
        for key, fitted, fc in results:
            self.fitted_models[key] = fitted
            self._forecasts[key]    = fc

        print(f"   ✅ ARIMA training complete — {len(self.fitted_models)} models fitted.")

    # ── Prediction assembly ───────────────────────────────────────────────────

    def predict(self) -> pd.DataFrame:
        """
        Assemble forecast values back into a DataFrame aligned with the test set.

        Returns
        -------
        pd.DataFrame with columns:
            store, item, year, week_of_year, weekly_sales, prediction
        """
        if not self._forecasts:
            raise RuntimeError("Call train() before predict().")

        print("\n▶ [ARIMA] Assembling forecast DataFrame...")

        rows = []
        for (s, i), grp in self._test_pdf.groupby(["store", "item"]):
            fc = self._forecasts.get((s, i), [])
            grp = grp.reset_index(drop=True)
            for idx, row in grp.iterrows():
                pred = fc[idx] if idx < len(fc) else float("nan")
                rows.append({
                    "store":        int(s),
                    "item":         int(i),
                    "year":         int(row["year"]),
                    "week_of_year": int(row["week_of_year"]),
                    "weekly_sales": float(row["weekly_sales"]),
                    "prediction":   float(pred),
                })

        predictions_df = pd.DataFrame(rows)
        print(f"   ↳ Forecast rows: {len(predictions_df):,}")
        print("   ✅ ARIMA predictions assembled.")
        return predictions_df

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, object_key: str = MINIO_OBJECT_KEY) -> None:
        """
        Serialize fitted models dict to pickle and upload to MinIO via boto3.

        Parameters
        ----------
        object_key : str
            MinIO object key under the ecommerce-lake bucket.
            Default: 'gold/models/arima/arima_models.pkl'
        """
        if not self.fitted_models:
            raise RuntimeError("No fitted models to save. Call train() first.")

        minio_endpoint = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
        access_key     = os.getenv("MINIO_ACCESS_KEY")
        secret_key     = os.getenv("MINIO_SECRET_KEY")

        print(f"\n▶ [ARIMA] Uploading models to MinIO: s3://{MINIO_BUCKET}/{object_key}")

        s3_client = boto3.client(
            "s3",
            endpoint_url=minio_endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )

        buffer = io.BytesIO()
        pickle.dump(self.fitted_models, buffer)
        buffer.seek(0)

        s3_client.upload_fileobj(buffer, MINIO_BUCKET, object_key)
        print(f"   ✅ ARIMA models uploaded to s3a://{MINIO_BUCKET}/{object_key}")
