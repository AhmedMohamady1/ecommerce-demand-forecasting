# E-Commerce Demand Forecasting — Project Plan
### Distributed Data Analysis | University Project

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset Description](#2-dataset-description)
3. [Repository Structure](#3-repository-structure)
4. [Infrastructure & Environment](#4-infrastructure--environment)
5. [Role 1 — Data Engineer: Ingestion & Storage](#5-role-1--data-engineer-ingestion--storage)
6. [Role 2 — Data Analyst: Exploration & Insights](#6-role-2--data-analyst-exploration--insights)
7. [Role 3 — ML Engineer: Feature Engineering & Modeling](#7-role-3--ml-engineer-feature-engineering--modeling)
8. [Role 4 — Big Data Engineer: Optimization & Performance](#8-role-4--big-data-engineer-optimization--performance)
9. [Role 5 — MLOps Engineer: Deployment & Monitoring](#9-role-5--mlops-engineer-deployment--monitoring)
10. [Deliverables Checklist](#10-deliverables-checklist)
11. [Decisions Log](#11-decisions-log)

---

## 1. Project Overview

| Field | Detail |
|---|---|
| **Goal** | Predict weekly per-store, per-item demand using PySpark |
| **Data** | Historical daily sales (10 stores × 50 items × ~5 years) |
| **Granularity** | Weekly aggregation (target = `weekly_sales`) |
| **Storage** | MinIO object store acting as a distributed Data Lake |
| **Processing** | Apache Spark in local mode (single machine, all cores) |
| **Models** | Linear Regression, Random Forest, Gradient Boosting, ARIMA, Prophet |
| **Language** | Python 3.13 |
| **Submission** | GitHub repository link |

### Objectives

- Predict future product demand at the store/item/week level
- Identify seasonal trends and purchasing patterns
- Produce optimized, partitioned Spark datasets suitable for downstream analytics
- Demonstrate distributed processing best practices (partitioning, replication, Spark tuning)

### Team Assignments

| Person | Roles Covered | Key Responsibilities |
|---|---|---|
| **Person 1** | Data Engineer + Data Analyst + Feature Engineering | Ingestion, cleaning, weekly aggregation, EDA notebook, lag/rolling/OHE features |
| **Person 2** | ML Engineer (modeling & evaluation) | Train Linear Regression, Random Forest, GBT, ARIMA, Prophet; evaluate all models |
| **Person 3** | Big Data Engineer | Spark tuning, partitioning strategy, repartition/coalesce, caching, MinIO layout |
| **Person 4** | MLOps Engineer | End-to-end pipeline script, automation, monitoring module, final documentation |
| **Person 5** | Support / Cross-role | Assists wherever needed; reviews PRs and integration testing |

---

## 2. Dataset Description

### Raw Data (`data/raw_data.csv`)

| Column | Type | Description |
|---|---|---|
| `date` | `DateType` | Daily date, starting 2013-01-01 |
| `store` | `IntegerType` | Store ID (1 – 10) |
| `item` | `IntegerType` | Item ID (1 – 50) |
| `sales` | `IntegerType` | Units sold that day |

- **Size**: ~913,000 rows (17 MB)
- **Period**: 2013-01-01 → 2017-12-31 (5 years)
- **Cardinality**: 10 stores × 50 items = 500 unique store/item pairs
- **No nulls expected**, but nulls/outlier checks will be performed in the pipeline

### Processed (Weekly) Schema

After the Data Engineering pipeline, the schema becomes:

| Column | Type | Notes |
|---|---|---|
| `store_id` | One-Hot Encoded (10 cols) | `store_1` … `store_10` |
| `item_id` | One-Hot Encoded (50 cols) | `item_1` … `item_50` |
| `week_of_year` | `IntegerType` (1–52) | ISO week number |
| `month` | `IntegerType` (1–12) | Calendar month |
| `week_has_holiday` | `IntegerType` (0 or 1) | 1 if any day in that week appears in the US holidays CSV |
| `lag_1_week` | `DoubleType` | Weekly sales, 1 week prior |
| `lag_4_week` | `DoubleType` | Weekly sales, 4 weeks prior |
| `lag_52_week` | `DoubleType` | Weekly sales, same week last year |
| `rolling_4_week_avg` | `DoubleType` | Mean of last 4 weeks |
| `rolling_12_week_avg` | `DoubleType` | Mean of last 12 weeks |
| `weekly_sales` | `DoubleType` | **Target variable** — sum of 7 daily sales |

> **Note:** `day_of_week` and `is_weekend` are dropped. `is_holiday` is NOT dropped — it is instead **aggregated to the week level** as `week_has_holiday` (1 if any day in the ISO week had a US public holiday). This allows the model to learn demand spikes on holiday weeks directly rather than relying solely on `week_of_year`.

**Total feature columns: 66** (10 store OHE + 50 item OHE + 2 temporal + 1 holiday + 3 lag + 2 rolling)

---

## 3. Repository Structure

```
ecommerce-demand-forecasting/
│
├── plan.md                         ← This file
├── README.md                       ← Project overview and setup instructions
├── requirements.txt                ← Python dependencies (PySpark, Prophet, etc.)
├── .env                            ← MinIO credentials and Spark config (gitignored)
├── .gitignore
│
├── scripts/
│   ├── start_minio.sh              ← Launches MinIO natively (macOS/Linux)
│   ├── start_minio.bat             ← Launches MinIO natively (Windows)
│   └── start_minio_docker.sh       ← Launches MinIO via Docker (cross-platform)
│
├── data/
│   ├── raw_data.csv                ← Source dataset (daily, 913k rows)
│   ├── US Holiday Dates (2004-2021).csv  ← US public holidays used for week_has_holiday feature
│   └── sample/                     ← Small subset for local testing (optional)
│
├── config/
│   └── spark_config.py             ← Centralized SparkSession factory + performance tuning (Role 4)
│
├── notebooks/
│   ├── 01_eda.ipynb                ← Exploratory Data Analysis (Role 2)
│   └── 02_model_evaluation.ipynb   ← Side-by-side model comparison and visualizations
│
├── src/
│   ├── __init__.py
│   │
│   ├── ingestion/                  ← Role 1: Data Engineer
│   │   ├── __init__.py
│   │   ├── ingest.py               ← Reads raw CSV into Spark, validates schema
│   │   └── upload_to_minio.py      ← Writes raw layer to MinIO (Bronze zone)
│   │
│   ├── preprocessing/              ← Role 1: Data Engineer (+ partitioning, Role 4)
│   │   ├── __init__.py
│   │   ├── cleaner.py              ← Null handling, type casting, deduplication
│   │   └── aggregator.py           ← Daily → Weekly aggregation (sum sales per store/item/week)
│   │
│   ├── feature_engineering/        ← Role 3: ML Engineer (+ Window optimization, Role 4)
│   │   ├── __init__.py
│   │   ├── temporal_features.py    ← week_of_year, month extraction
│   │   ├── holiday_features.py     ← Joins US holidays CSV; creates week_has_holiday flag
│   │   ├── lag_features.py         ← lag_1, lag_4, lag_52 (using Spark Window functions)
│   │   ├── rolling_features.py     ← rolling_4_avg, rolling_12_avg
│   │   └── encoder.py              ← One-Hot Encoding for store_id and item_id
│   │
│   ├── eda/                        ← Role 2: Data Analyst
│   │   ├── __init__.py
│   │   └── analysis.py             ← Reusable Spark aggregation queries for EDA
│   │
│   ├── models/                     ← Role 3: ML Engineer (+ caching, Role 4)
│   │   ├── __init__.py
│   │   ├── base_model.py           ← Abstract base class / shared train-eval interface
│   │   ├── train_evaluate.py       ← ML orchestration (Steps 7.2–7.5: split → train → evaluate)
│   │   ├── linear_regression.py    ← Spark MLlib Linear Regression
│   │   ├── random_forest.py        ← Spark MLlib Random Forest Regressor
│   │   ├── gradient_boosting.py    ← Spark MLlib GBT Regressor
│   │   ├── arima_model.py          ← ARIMA via statsmodels (per store/item group)
│   │   └── prophet_model.py        ← Facebook Prophet (per store/item group)
│   │
│   ├── evaluation/                 ← Role 3: ML Engineer
│   │   ├── __init__.py
│   │   └── metrics.py              ← RMSE, MAE, MAPE, R² computation helpers
│   │
│   └── pipeline/                   ← Role 5: MLOps Engineer
│       ├── __init__.py
│       ├── full_pipeline.py        ← End-to-end orchestration script
│       └── monitor.py              ← Drift detection and performance tracking helpers
│
├── results/
│   ├── metrics/                    ← Saved model evaluation CSVs
│   └── plots/                      ← Saved visualizations (PNG/HTML)
│
└── docs/
    ├── architecture_diagram.png    ← System architecture visual
    └── final_report.md             ← Full project documentation
```

### MinIO Data Lake Zones (Bucket: `ecommerce-lake`)

```
ecommerce-lake/
├── bronze/                          ← Raw ingested data (unchanged, daily)
│   └── raw_sales/
│       └── *.parquet                (no partitioning)
│
├── silver/                          ← Cleaned & enriched data (two sub-zones)
│   ├── daily_sales/                 ← Daily rows + is_holiday per row
│   │   └── store=1/ … store=10/    (partitioned by store, ~913k rows)
│   └── weekly_sales/                ← Weekly aggregated rows
│       └── store=1/ … store=10/    (partitioned by store, ~130k rows)
│
└── gold/                            ← Feature-engineered, model-ready data
    └── features/
        └── store=1/ … store=10/    (partitioned by store, written by Role 3)
```

| Zone | Sub-zone | Granularity | Key columns | Written by |
|---|---|---|---|---|
| Bronze | `raw_sales/` | Daily (raw) | date, store, item, sales | Role 1 — Step B |
| Silver | `daily_sales/` | Daily (enriched) | date, store, item, sales, **is_holiday** | Role 1 — Step C |
| Silver | `weekly_sales/` | Weekly (aggregated) | store, item, year, week_of_year, **weekly_sales**, **week_has_holiday** | Role 1 — Step D |
| Gold | `features/` | Weekly (feature-engineered) | + lag_1/4/52, rolling_4/12, store/item OHE | Role 3 — Feature Eng. |

---

## 4. Infrastructure & Environment

### Overview

The project supports **two setup paths** — team members can use whichever fits their machine. Both paths expose MinIO on the same `localhost:9000` endpoint, so **all Python/Spark code is identical regardless of which path is used.**

| Component | Native path | Docker path |
|---|---|---|
| **MinIO** | Binary download, run via shell script | `docker run` one-liner |
| **MinIO Client (`mc`)** | Binary download | Installed inside the same container / separate `mc` binary |
| **Apache Spark** | `pyspark` Python package (embedded) | `pyspark` Python package (embedded) |
| **Python deps** | `pip install -r requirements.txt` in `.venv` | Same |

> **Both paths target the same `.env` file** for credentials and the same MinIO endpoint (`http://localhost:9000`). No code changes are needed when switching paths.

---

### Path A — Native Installation (macOS / Linux / Windows)

#### A1. Install MinIO Server Binary

```bash
# macOS — Apple Silicon (ARM64)
curl -O https://dl.min.io/server/minio/release/darwin-arm64/minio
chmod +x minio && sudo mv minio /usr/local/bin/

# macOS — Intel (AMD64)
curl -O https://dl.min.io/server/minio/release/darwin-amd64/minio
chmod +x minio && sudo mv minio /usr/local/bin/

# Linux (x86_64)
curl -O https://dl.min.io/server/minio/release/linux-amd64/minio
chmod +x minio && sudo mv minio /usr/local/bin/
```

```powershell
# Windows (PowerShell) — download to a folder on your PATH, e.g. C:\minio
Invoke-WebRequest -Uri "https://dl.min.io/server/minio/release/windows-amd64/minio.exe" `
    -OutFile "C:\minio\minio.exe"
# Add C:\minio to your system PATH via System Properties → Environment Variables
```

#### A2. Install MinIO Client (`mc`)

```bash
# macOS — Apple Silicon
curl -O https://dl.min.io/client/mc/release/darwin-arm64/mc
chmod +x mc && sudo mv mc /usr/local/bin/

# macOS — Intel
curl -O https://dl.min.io/client/mc/release/darwin-amd64/mc
chmod +x mc && sudo mv mc /usr/local/bin/

# Linux (x86_64)
curl -O https://dl.min.io/client/mc/release/linux-amd64/mc
chmod +x mc && sudo mv mc /usr/local/bin/
```

```powershell
# Windows (PowerShell)
Invoke-WebRequest -Uri "https://dl.min.io/client/mc/release/windows-amd64/mc.exe" `
    -OutFile "C:\minio\mc.exe"
```

#### A3. Start MinIO — macOS / Linux (`scripts/start_minio.sh`)

```bash
#!/usr/bin/env bash
set -e
source .env

export MINIO_ROOT_USER=$MINIO_ACCESS_KEY
export MINIO_ROOT_PASSWORD=$MINIO_SECRET_KEY

mkdir -p ~/minio-data
minio server ~/minio-data --address ':9000' --console-address ':9001' &

echo "Waiting for MinIO to start..."
sleep 3

mc alias set local http://localhost:9000 $MINIO_ACCESS_KEY $MINIO_SECRET_KEY
mc mb --ignore-existing local/ecommerce-lake
echo "MinIO ready at http://localhost:9000  (console: http://localhost:9001)"
```

```bash
bash scripts/start_minio.sh
```

#### A4. Start MinIO — Windows (`scripts/start_minio.bat`)

```bat
@echo off
for /f "tokens=1,2 delims==" %%A in (.env) do set %%A=%%B

set MINIO_ROOT_USER=%MINIO_ACCESS_KEY%
set MINIO_ROOT_PASSWORD=%MINIO_SECRET_KEY%

if not exist "%USERPROFILE%\minio-data" mkdir "%USERPROFILE%\minio-data"

start "MinIO" minio.exe server %USERPROFILE%\minio-data --address :9000 --console-address :9001

timeout /t 4 /nobreak > nul

mc.exe alias set local http://localhost:9000 %MINIO_ACCESS_KEY% %MINIO_SECRET_KEY%
mc.exe mb --ignore-existing local/ecommerce-lake
echo MinIO ready at http://localhost:9000
```

```bat
scripts\start_minio.bat
```

---

### Path B — Docker (cross-platform)

Requires **Docker Desktop** (or Docker Engine on Linux). No binary downloads needed.

#### B1. Start MinIO (`scripts/start_minio_docker.sh`)

```bash
#!/usr/bin/env bash
set -e
source .env

docker run -d \
  --name minio \
  -p 9000:9000 \
  -p 9001:9001 \
  -e MINIO_ROOT_USER=$MINIO_ACCESS_KEY \
  -e MINIO_ROOT_PASSWORD=$MINIO_SECRET_KEY \
  -v ~/minio-data:/data \
  minio/minio server /data --console-address ':9001'

echo "Waiting for MinIO to start..."
sleep 4

docker run --rm --network host \
  -e MC_HOST_local="http://${MINIO_ACCESS_KEY}:${MINIO_SECRET_KEY}@localhost:9000" \
  minio/mc mb --ignore-existing local/ecommerce-lake

echo "MinIO ready at http://localhost:9000  (console: http://localhost:9001)"
```

```bash
bash scripts/start_minio_docker.sh
```

> **On Windows with Docker Desktop**, run the same script inside Git Bash or WSL, or adapt the `docker run` command to a `.bat` / PowerShell script using `%MINIO_ACCESS_KEY%` variable syntax.

#### B2. Stop / Remove MinIO container

```bash
docker stop minio && docker rm minio
```

---

### Environment File (`.env`) — shared by both paths

```
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin123
```

> **Replication note**: In this single-process MinIO setup, objects are stored on the local filesystem. A fault-tolerant production deployment would use MinIO in distributed/erasure-coding mode across multiple nodes (equivalent to HDFS `dfs.replication=3`). This is discussed in `docs/final_report.md`.

> **Why Spark local mode?** The dataset (~130k weekly rows after aggregation) fits comfortably in-process. Local mode still exercises all Spark APIs, partitioning, and Window functions identically to a distributed cluster.

---

### Python Environment Setup (all platforms)

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

```powershell
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### SparkSession Configuration (Local Mode + MinIO / S3A)

```python
spark = SparkSession.builder \
    .master("local[*]") \
    .appName("EcommerceDemandForecasting") \
    .config("spark.hadoop.fs.s3a.endpoint", "http://localhost:9000") \
    .config("spark.hadoop.fs.s3a.access.key", os.getenv("MINIO_ACCESS_KEY")) \
    .config("spark.hadoop.fs.s3a.secret.key", os.getenv("MINIO_SECRET_KEY")) \
    .config("spark.hadoop.fs.s3a.path.style.access", "true") \
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
    .getOrCreate()
```

### Python Dependencies (`requirements.txt`)

```
pyspark>=3.4
pandas
numpy
matplotlib
seaborn
plotly
prophet
statsmodels
scikit-learn
boto3
python-dotenv
pyarrow
joblib
```

---

## 5. Role 1 — Data Engineer: Ingestion & Storage

### 5.1 Goal
Transform `data/raw_data.csv` (daily, 913k rows) into a clean, aggregated weekly Parquet dataset stored in MinIO.

### 5.2 Step-by-Step Tasks

#### Step A — Ingestion (`src/ingestion/ingest.py`)
1. Initialize SparkSession from `config/spark_config.py`
2. Read `data/raw_data.csv` with an explicit schema:
   ```
   date: DateType, store: IntegerType, item: IntegerType, sales: IntegerType
   ```
3. Validate:
   - Row count matches expected ~913,000
   - No nulls in any column
   - `sales` >= 0
   - `store` in [1, 10], `item` in [1, 50]
   - Date range: 2013-01-01 to 2017-12-31
4. Print a validation summary to stdout

#### Step B — Upload Raw Layer (`src/ingestion/upload_to_minio.py`)
1. Write raw Spark DataFrame as Parquet to `s3a://ecommerce-lake/bronze/raw_sales/`
2. Use `mode("overwrite")` to make the step idempotent

#### Step C — Holiday Enrichment (`src/preprocessing/cleaner.py`)
1. Cast `date` column to `DateType` if not already
2. Drop duplicate rows
3. Filter out rows with `sales < 0`
4. Load `data/US Holiday Dates (2004-2021).csv` into a Spark DataFrame; keep only the `Date` column and add a literal `is_holiday = 1` flag
5. Left-join the daily sales DataFrame on `date == holiday_date` → fill nulls in `is_holiday` with `0`
6. Log count of removed/enriched rows

> **Holiday CSV schema**: `Date, Holiday, WeekDay, Month, Day, Year` — 343 rows covering 2004–2021 across 18 holiday types (New Year's Day, Thanksgiving, Christmas, 4th of July, Memorial Day, Labor Day, Valentine's Day, MLK Day, etc.)

#### Step D — Weekly Aggregation (`src/preprocessing/aggregator.py`)
1. Extract `year` and `week_of_year` from `date` using `weekofyear()` and `year()`
2. Group by `(store, item, year, week_of_year)` and compute:
   - `weekly_sales = sum(sales)`
   - `week_has_holiday = max(is_holiday)` → 1 if any day in that week had a holiday, else 0
3. Result: ~500 store/item pairs × ~260 weeks ≈ **130,000 rows**
4. Write Silver layer to `s3a://ecommerce-lake/silver/weekly_sales/`
   - Partitioned by `store`

### 5.3 Output Artifact
`s3a://ecommerce-lake/silver/weekly_sales/` — clean weekly Parquet (with `week_has_holiday`), partitioned by `store`

---

## 6. Role 2 — Data Analyst: Exploration & Insights

### 6.1 Goal
Explore the Silver dataset to understand sales distributions, seasonality, store/item trends, and validate preprocessing decisions before modeling.

### 6.2 Notebook: `notebooks/01_eda.ipynb`

All heavy computation uses PySpark; results are `.toPandas()` before plotting.

#### Section 1 — Basic Statistics
- Schema printout and row count
- `df.describe()` on `weekly_sales`
- Distribution of weekly sales (histogram)
- Null check confirmation

#### Section 2 — Temporal Trends
- **Total weekly sales over time** (line chart): aggregate all stores/items by week
- **Monthly seasonality**: box plot of `weekly_sales` grouped by `month`
- **Week-of-year heatmap**: avg weekly sales by `week_of_year` across all years

#### Section 3 — Store-Level Analysis
- Bar chart: Total sales per store
- Line chart: Average weekly sales per store across time

#### Section 4 — Item-Level Analysis
- Top 10 and bottom 10 items by total sales
- Sales variance per item (volatility analysis)

#### Section 5 — Correlation & Lag Validation
- Scatter plot: `lag_1_week` vs `weekly_sales`
- Pearson correlation matrix of all lag/rolling features with `weekly_sales`

#### Section 6 — Key Insights (written summary)
- Strong yearly seasonality (summer peaks, Q4 holiday patterns)
- High variance between items (some items sell 5× more than others)
- Store-level baselines differ but trend shapes are consistent
- `week_of_year` is a strong predictor, validating the holiday encoding decision

### 6.3 Output Artifacts
- Saved plots in `results/plots/`
- Written insights in `docs/final_report.md`

---

## 7. Role 3 — ML Engineer: Feature Engineering & Modeling

### 7.1 Feature Engineering

Operates on the Silver layer and outputs to the Gold layer.

#### Step A — Temporal Features (`src/feature_engineering/temporal_features.py`)
- `week_of_year = weekofyear(date_col)` — IntegerType, range 1-52
- `month = month(date_col)` — IntegerType, range 1-12

#### Step B — Holiday Feature (`src/feature_engineering/holiday_features.py`)

`week_has_holiday` is produced during **aggregation** (Step D of Role 1), not here. This module provides a validation helper that verifies the column exists in the Silver layer and logs the distribution:

```
week_has_holiday = 0 → majority of weeks
week_has_holiday = 1 → weeks containing at least one US public holiday
```

Expected holiday weeks per year: ~15–20 (depending on which holidays fall in unique ISO weeks).

> **Why `max(is_holiday)` over the week?** A week is flagged as a holiday week if *any* of its 7 days is a holiday. This correctly captures multi-day holiday events (e.g., Christmas Eve + Christmas Day both in week 52) without double-counting.

#### Step C — Lag Features (`src/feature_engineering/lag_features.py`)

Use `pyspark.sql.Window` partitioned by `(store, item)`, ordered by `(year, week_of_year)`:

| Feature | Window Offset | Description |
|---|---|---|
| `lag_1_week` | -1 week | Previous week's sales |
| `lag_4_week` | -4 weeks | ~1 month ago |
| `lag_52_week` | -52 weeks | Same week, prior year |

> Rows with nulls in any lag feature (the first 52 weeks per store/item) are **dropped** before modeling.

#### Step D — Rolling Features (`src/feature_engineering/rolling_features.py`)

Use `Window.rowsBetween(-N, -1)` with `avg()`:

| Feature | Window | Description |
|---|---|---|
| `rolling_4_week_avg` | Last 4 rows | Short-term trend |
| `rolling_12_week_avg` | Last 12 rows | Quarterly trend |

#### Step E — One-Hot Encoding (`src/feature_engineering/encoder.py`)

Use Spark MLlib `StringIndexer` → `OneHotEncoder` pipeline:
- `store` (10 values) → 10 binary columns: `store_1` … `store_10`
- `item` (50 values) → 50 binary columns: `item_1` … `item_50`

Gold Layer written to `s3a://ecommerce-lake/gold/features/`, partitioned by `(store, week_of_year)`.

### 7.2 Train/Test Split

| Split | Period | Approximate Rows |
|---|---|---|
| Train | 2013-W01 → 2016-W52 | ~104,000 |
| Test | 2017-W01 → 2017-W52 | ~26,000 |

Split is **chronological** (not random) to prevent data leakage.

### 7.3 Models

#### Model A — Linear Regression (`src/models/linear_regression.py`)
- **Library**: `pyspark.ml.regression.LinearRegression`
- **Purpose**: Interpretable baseline; coefficients reveal feature importance
- **Hyperparameters**: `maxIter=100`, `regParam=0.1`, `elasticNetParam=0.0`

#### Model B — Random Forest (`src/models/random_forest.py`)
- **Library**: `pyspark.ml.regression.RandomForestRegressor`
- **Purpose**: Non-linear interactions + built-in feature importance ranking
- **Hyperparameters**: `numTrees=100`, `maxDepth=10`, `featureSubsetStrategy="auto"`

#### Model C — Gradient Boosting (`src/models/gradient_boosting.py`)
- **Library**: `pyspark.ml.regression.GBTRegressor`
- **Purpose**: Typically highest accuracy; captures complex patterns
- **Hyperparameters**: `maxIter=50`, `maxDepth=5`, `stepSize=0.1`

#### Model D — ARIMA (`src/models/arima_model.py`)
- **Library**: `statsmodels.tsa.arima.model.ARIMA`
- **Purpose**: Classical time-series baseline; captures autocorrelation
- **Strategy**: Group by `(store, item)`, convert to pandas, fit ARIMA(1,1,1) per group — **run across all 500 store/item pairs**
- **Parallelism**: Use `joblib.Parallel` or Python `multiprocessing` to fit pairs concurrently across CPU cores
- **Output**: Collect per-pair forecasts back into a single Spark DataFrame for unified evaluation

#### Model E — Prophet (`src/models/prophet_model.py`)
- **Library**: `prophet`
- **Purpose**: Handles seasonality + trend automatically; best for long-horizon forecasts
- **Strategy**: Same group-wise approach as ARIMA; columns renamed to `ds` / `y` — **run across all 500 store/item pairs**
- **Config**: `yearly_seasonality=True`, `weekly_seasonality=False`, `daily_seasonality=False`
- **Parallelism**: Same `joblib.Parallel` approach as ARIMA to keep runtime manageable

### 7.4 Model Persistence

All trained models are saved to the **MinIO Gold zone** after training:

| Model | Persistence Method | MinIO Path |
|---|---|---|
| Linear Regression | Spark MLlib `.save()` | `s3a://ecommerce-lake/gold/models/linear_regression/` |
| Random Forest | Spark MLlib `.save()` | `s3a://ecommerce-lake/gold/models/random_forest/` |
| Gradient Boosting | Spark MLlib `.save()` | `s3a://ecommerce-lake/gold/models/gradient_boosting/` |
| ARIMA | Pickle (per-pair dict) | `s3a://ecommerce-lake/gold/models/arima/arima_models.pkl` |
| Prophet | Pickle (per-pair dict) | `s3a://ecommerce-lake/gold/models/prophet/prophet_models.pkl` |

### 7.5 Evaluation (`src/evaluation/metrics.py`)

| Metric | Description |
|---|---|
| **RMSE** | Root Mean Squared Error — penalizes large errors |
| **MAE** | Mean Absolute Error — average magnitude |
| **MAPE** | Mean Absolute Percentage Error — scale-independent |
| **R²** | Coefficient of Determination — goodness of fit |

Results saved to `results/metrics/model_comparison.csv` and also uploaded to `s3a://ecommerce-lake/gold/metrics/`.

---

## 8. Role 4 — Big Data Engineer: Optimization & Performance

> **Implementation note:** Rather than creating a separate `src/optimization/` package, all optimization decisions are applied **in-place** within the modules that use them. This avoids unnecessary wrapper code and keeps tuning configs co-located with the logic they affect.

### 8.1 Partitioning Strategy

Applied at write time via `.partitionBy()` in each pipeline stage:

| Stage | Partition Key(s) | Implemented In | Rationale |
|---|---|---|---|
| Bronze (raw) | None | `src/ingestion/upload_to_minio.py` | Single large file; no predicate pushdown needed |
| Silver (weekly) | `store` | `src/preprocessing/aggregator.py`, `cleaner.py` | 10 balanced partitions; enables per-store parallel reads |
| Gold (features) | `store` | `src/feature_engineering/engineer.py` | 10 partitions for balanced model training reads |

### 8.2 Repartition vs Coalesce

```python
# Before groupBy — shuffle for even distribution
df = df.repartition(10, "store", "item")

# After aggregation — reduce partitions without full shuffle
df = df.coalesce(10)
```

> Use `repartition()` when increasing partition count or requiring a full shuffle; use `coalesce()` only to reduce partition count cheaply.

### 8.3 Spark Performance Tuning (`config/spark_config.py`)

All performance configs are centralized in the SparkSession factory:

| Config | Value | Reason |
|---|---|---|
| `spark.master` | `local[*]` | Uses all available CPU cores on the machine |
| `spark.driver.memory` | `8g` | Driver acts as both master and executor in local mode |
| `spark.sql.shuffle.partitions` | `20` | Avoids the wasteful default of 200 for this dataset size |
| `spark.sql.autoBroadcastJoinThreshold` | `10mb` | Auto-broadcasts small dimension tables |
| `spark.serializer` | `KryoSerializer` | Faster than Java default serializer |
| `spark.sql.parquet.compression.codec` | `snappy` | Fast compression; reduces I/O for MinIO reads/writes |

### 8.4 Window Function Optimization (`src/feature_engineering/engineer.py`)

- Windows use `partitionBy("store", "item").orderBy("abs_week")` — Spark only sorts within each partition, avoiding a global shuffle
- Cache train/test DataFrames before model training to avoid recomputation (`src/models/train_evaluate.py`):
  ```python
  train_df.cache()
  test_df.cache()
  train_df.count()  # trigger materialization
  # … train all 5 models …
  train_df.unpersist()
  test_df.unpersist()
  ```

### 8.5 Replication (MinIO)

- MinIO configured with erasure coding for fault tolerance (equivalent to HDFS RF=3)
- In a production HDFS setup: `dfs.replication=3`
- For local deployment: single-node mode with volume mounts for data persistence (`scripts/start_minio.sh`)
- Object versioning supported for rollback capability

---

## 9. Role 5 — MLOps Engineer: Deployment & Monitoring

### 9.1 End-to-End Pipeline (`src/pipeline/full_pipeline.py`)

```
[1] Ingest raw CSV
    → [2] Upload to Bronze (MinIO)
    → [3] Clean & Aggregate
    → [4] Write Silver (MinIO)
    → [5] Feature Engineering
    → [6] Write Gold (MinIO)
    → [7] Train all models
    → [8] Evaluate & save metrics
    → [9] Log summary
```

- Each step is wrapped in `try/except` with structured logging
- Accepts `--stage` argument to resume from any step (e.g., `--stage feature_engineering`)

### 9.2 Automation

- **Manual**: `python src/pipeline/full_pipeline.py --stage all`
- **Cron/Shell**: Scheduled weekly to retrain on new data
- *(Stretch goal)* **Apache Airflow DAG**: `dags/demand_forecast_dag.py`

### 9.3 Model Monitoring (`src/pipeline/monitor.py`)

| Check | Method |
|---|---|
| **Performance drift** | Compare current RMSE vs baseline; alert if > 10% degradation |
| **Data drift** | KS test on `weekly_sales` distribution of new vs training data |
| **Missing data** | Alert if new input has > 1% null rate |

Results appended to `results/metrics/monitoring_log.csv` with timestamps.

### 9.4 Deliverables from MLOps Role
- `src/pipeline/full_pipeline.py`
- `results/metrics/model_comparison.csv`
- `results/metrics/monitoring_log.csv`
- Documentation section in `docs/final_report.md`

---

## 10. Deliverables Checklist

| # | Deliverable | Owner Role | Location |
|---|---|---|---|
| 1 | Cleaned & preprocessed dataset (Silver) | Data Engineer | `s3a://ecommerce-lake/silver/` |
| 2 | Feature-engineered dataset (Gold) | ML Engineer | `s3a://ecommerce-lake/gold/` |
| 3 | EDA notebook with visualizations | Data Analyst | `notebooks/01_eda.ipynb` |
| 4 | Trained forecasting models (5 total) | ML Engineer | `src/models/` |
| 5 | Model comparison metrics | ML Engineer | `results/metrics/model_comparison.csv` |
| 6 | Model evaluation notebook | ML Engineer | `notebooks/02_model_evaluation.ipynb` |
| 7 | Optimized Spark pipeline | Big Data Engineer | `config/spark_config.py` + in-place across `src/` |
| 8 | End-to-end pipeline script | MLOps Engineer | `src/pipeline/full_pipeline.py` |
| 9 | Monitoring module | MLOps Engineer | `src/pipeline/monitor.py` |
| 10 | Full project documentation | All | `docs/final_report.md` |
| 11 | MinIO startup script | Big Data Engineer | `scripts/start_minio.sh` |

---

