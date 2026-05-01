"""
src/models/prophet_model.py
─────────────────────────────────────────────────────────────────────────────
Facebook Prophet model for weekly demand forecasting — fitted per (store, item).

Role       : Model E — handles seasonality + trend automatically
Library    : prophet
Strategy   : Group by (store, item), convert to ds/y Pandas DataFrame, fit Prophet.
Parallelism: joblib.Parallel across all 500 store/item pairs.
Config     : yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False
Persistence: pickle dict  →  s3a://ecommerce-lake/gold/models/prophet/prophet_models.pkl
             (uploaded via boto3)

Data contract:
    Input columns used: store, item, year, week_of_year, abs_week, weekly_sales
    Excluded (per project spec): year, abs_week, store_idx, item_idx

    Prophet input format:
        ds : date-like column (we reconstruct from year + week_of_year)
        y  : target (weekly_sales)

Output:
    predictions_df : pandas DataFrame with columns
        store, item, year, week_of_year, weekly_sales, prediction
"""

from __future__ import annotations

import io
import os
import pickle
import warnings
from typing import Dict, List, Optional, Tuple

import boto3
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# Silence Prophet's verbose logging and Stan warnings
import logging
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=FutureWarning)

from prophet import Prophet

# ── Constants ────────────────────────────────────────────────────────────────

N_JOBS           = -1            # use all CPU cores
MINIO_BUCKET     = "ecommerce-lake"
MINIO_OBJECT_KEY = "gold/models/prophet/prophet_models.pkl"

# Columns collected from Spark — abs_week is intentionally excluded (project spec).
# Chronological ordering uses (year, week_of_year) instead.
NEEDED_COLS = ["store", "item", "year", "week_of_year", "weekly_sales"]


# ── Week → date helper ────────────────────────────────────────────────────────

def _week_to_date(year: int, week: int) -> pd.Timestamp:
    """
    Convert (year, ISO week_of_year) to a Monday date (ds column for Prophet).
    Uses ISO-consistent week start (Monday of that ISO week).
    """
    return pd.Timestamp.fromisocalendar(year, int(week), 1)


# ── Per-pair Prophet fit/forecast ─────────────────────────────────────────────

def _fit_predict_pair(
    key: Tuple[int, int],
    train_pdf: pd.DataFrame,
    test_pdf: pd.DataFrame,
) -> Tuple[Tuple[int, int], Optional[Prophet], List[float]]:
    """
    Fit a Prophet model for one (store, item) pair and forecast the test period.

    Parameters
    ----------
    key       : (store, item)
    train_pdf : Pandas DataFrame for the pair (train period)
    test_pdf  : Pandas DataFrame for the pair (test period)

    Returns
    -------
    (key, fitted_prophet_model, forecast_values_list)
    """
    try:
        # Build Prophet input DataFrames
        train_prophet = pd.DataFrame({
            "ds": [_week_to_date(r["year"], r["week_of_year"]) for _, r in train_pdf.iterrows()],
            "y":  train_pdf["weekly_sales"].values,
        })
        test_prophet = pd.DataFrame({
            "ds": [_week_to_date(r["year"], r["week_of_year"]) for _, r in test_pdf.iterrows()],
        })

        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
        )
        model.fit(train_prophet)

        forecast = model.predict(test_prophet)
        fc = forecast["yhat"].tolist()

    except Exception as exc:
        # Fallback: last observed value (naive)
        last_val = float(train_pdf["weekly_sales"].iloc[-1]) if len(train_pdf) > 0 else 0.0
        fc       = [last_val] * len(test_pdf)
        model    = None
        print(f"   ⚠ Prophet failed for store={key[0]}, item={key[1]}: {exc} "
              f"— using naive fallback")

    return key, model, fc


# ── Main Prophet class ────────────────────────────────────────────────────────

class ProphetModel:
    """
    Facebook Prophet forecasting across all (store, item) pairs in parallel.

    Parameters
    ----------
    n_jobs : int   joblib parallelism — default -1 (all cores)
    """

    def __init__(self, n_jobs: int = N_JOBS) -> None:
        self.n_jobs          = n_jobs
        self.fitted_models:  Dict[Tuple[int, int], Optional[Prophet]] = {}
        self._train_pdf: Optional[pd.DataFrame] = None
        self._test_pdf:  Optional[pd.DataFrame] = None
        self._forecasts: Dict[Tuple[int, int], List[float]] = {}

    # ── Data preparation ──────────────────────────────────────────────────────

    def prepare_data(
        self,
        train_spark_df,   # pyspark.sql.DataFrame
        test_spark_df,    # pyspark.sql.DataFrame
    ) -> None:
        """
        Collect train and test Spark DataFrames to Pandas for per-pair processing.
        """
        print("\n▶ [Prophet] Collecting data from Spark to Pandas...")

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
        print(f"   ↳ Train rows : {len(self._train_pdf):,}")
        print(f"   ↳ Test  rows : {len(self._test_pdf):,}")
        print(f"   ↳ Unique (store, item) pairs: {n_pairs}")

    # ── Training (parallel Prophet fits) ─────────────────────────────────────

    def train(self) -> None:
        """
        Fit Prophet for every (store, item) pair in parallel.

        Requires prepare_data() to have been called first.
        """
        if self._train_pdf is None:
            raise RuntimeError("Call prepare_data() before train().")

        print("\n▶ [Prophet] Fitting Prophet for all (store, item) pairs...")
        print(f"   yearly_seasonality=True | weekly_seasonality=False | "
              f"daily_seasonality=False")
        print(f"   Using {self.n_jobs} parallel jobs (joblib)...")

        # Build per-pair (train, test) DataFrames
        train_groups = {
            (int(s), int(i)): grp.reset_index(drop=True)
            for (s, i), grp in self._train_pdf.groupby(["store", "item"])
        }
        test_groups = {
            (int(s), int(i)): grp.reset_index(drop=True)
            for (s, i), grp in self._test_pdf.groupby(["store", "item"])
        }

        pairs = sorted(train_groups.keys())

        results = Parallel(n_jobs=self.n_jobs, verbose=5)(
            delayed(_fit_predict_pair)(
                key=key,
                train_pdf=train_groups[key],
                test_pdf=test_groups.get(key, pd.DataFrame(columns=NEEDED_COLS)),
            )
            for key in pairs
        )

        for key, model, fc in results:
            self.fitted_models[key] = model
            self._forecasts[key]    = fc

        print(f"   ✅ Prophet training complete — {len(self.fitted_models)} models fitted.")

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

        print("\n▶ [Prophet] Assembling forecast DataFrame...")

        rows = []
        for (s, i), grp in self._test_pdf.groupby(["store", "item"]):
            fc  = self._forecasts.get((s, i), [])
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
        print("   ✅ Prophet predictions assembled.")
        return predictions_df

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, object_key: str = MINIO_OBJECT_KEY) -> None:
        """
        Serialize fitted Prophet models dict to pickle and upload to MinIO.

        Parameters
        ----------
        object_key : str
            MinIO object key under the ecommerce-lake bucket.
            Default: 'gold/models/prophet/prophet_models.pkl'
        """
        if not self.fitted_models:
            raise RuntimeError("No fitted models to save. Call train() first.")

        minio_endpoint = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
        access_key     = os.getenv("MINIO_ACCESS_KEY")
        secret_key     = os.getenv("MINIO_SECRET_KEY")

        print(f"\n▶ [Prophet] Uploading models to MinIO: s3://{MINIO_BUCKET}/{object_key}")

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
        print(f"   ✅ Prophet models uploaded to s3a://{MINIO_BUCKET}/{object_key}")
