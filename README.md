# E-Commerce Demand Forecasting

![Python](https://img.shields.io/badge/Python-3.13-blue.svg?style=flat) ![Apache Spark](https://img.shields.io/badge/Apache%20Spark-E25A1C.svg?style=flat&logo=apachespark&logoColor=white) ![MinIO](https://img.shields.io/badge/MinIO-C7202C.svg?style=flat&logo=minio&logoColor=white) ![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E.svg?style=flat&logo=scikit-learn&logoColor=white) ![Streamlit](https://img.shields.io/badge/Streamlit-FE4B4B.svg?style=flat&logo=streamlit&logoColor=white)

> An end-to-end distributed machine learning pipeline for predicting weekly product demand at the store and item level using PySpark and a MinIO Data Lake.

---

## Overview

This project implements a complete distributed data processing and forecasting pipeline. It ingests 5 years of historical daily sales data, enriches it with external holiday information, aggregates it to a weekly level, engineers features, and trains multiple predictive models. Finally, an interactive Streamlit dashboard visualizes model predictions and tracking metrics.

**Key Numbers:**
- **913,000** raw daily sales records
- **500** unique store/item pairs (10 stores × 50 items)
- **130,000** aggregated weekly rows for modeling
- **66** engineered features including lags, rolling averages, and one-hot encoding
- **5** models evaluated (Linear Regression, Random Forest, GBT, ARIMA, Prophet)

---

## Project Structure

```text
ecommerce-demand-forecasting/
│
├── data/                           # Datasets
│   ├── raw_data.csv                # Source dataset (daily sales)
│   ├── US Holiday Dates (2004-2021).csv # External holiday features
│   └── 2017_test_data.csv          # Held-out testing dataset
│
├── scripts/                        # Infrastructure Startup & Utilities
│   ├── start_minio_docker.sh       # Launches MinIO via Docker
│   ├── start_minio.sh              # Launches MinIO natively (macOS/Linux)
│   ├── start_minio.bat             # Launches MinIO natively (Windows)
│   └── export_to_csv.py            # Utility script
│
├── config/                         # Configuration
│   └── spark_config.py             # Centralized SparkSession factory + tuning
│
├── notebooks/                      # Exploration & Evaluation
│   ├── 01_eda.ipynb                # Exploratory Data Analysis
│   └── 02_model_evaluation.ipynb   # Model comparison and visualization
│
├── src/                            # Source Code
│   ├── ingestion/                  # Raw CSV to MinIO Bronze layer
│   ├── preprocessing/              # Null handling, weekly aggregation (Silver)
│   ├── feature_engineering/        # Temporal, lag, rolling, OHE features (Gold)
│   ├── models/                     # LR, Random Forest, GBT, ARIMA, Prophet
│   ├── evaluation/                 # Metrics computation (RMSE, MAE, MAPE, R²)
│   ├── pipeline/                   # End-to-end orchestration and monitoring
│   └── app/                        # Streamlit interactive dashboard
│
├── results/                        # Outputs
│   ├── metrics/                    # Saved model evaluation CSVs
│   └── plots/                      # Saved visualizations
│
├── docs/                           # Documentation
│   └── final_report.pdf            # Full project documentation
│
├── requirements.txt                # Python dependencies
├── .env                            # MinIO credentials and Spark config
└── README.md
```

---

## Pipeline Overview

| Layer / Phase | Module | Description |
|---|---|---|
| **Bronze (Raw)** | Ingestion | Ingested ~913k daily sales records into raw unpartitioned Parquet files. |
| **Silver (Cleaned)** | Preprocessing | Handled data quality, joined US holidays, and aggregated daily to weekly data. |
| **Gold (Features)** | Feature Eng. | Computed temporal flags, lag features (1, 4, 52 weeks), rolling averages (4, 12 weeks), and one-hot encoding for stores/items. |
| **Modeling** | Model Training | Split chronologically (Train: 2013-2016, Test: 2017). Trained interpretable ML models (RF, GBT) and time-series specific models (ARIMA, Prophet). |
| **Deployment** | MLOps & Dashboard | Executed via a unified pipeline script; tracked data/performance drift; deployed inference to a Streamlit frontend. |

---

## Tech Stack

| Category | Technologies |
|---|---|
| **Core** | Python 3.13, Pandas, Numpy |
| **Big Data Processing** | Apache Spark (PySpark) |
| **Data Lake Storage** | MinIO, Hadoop AWS (S3A) |
| **Machine Learning** | Scikit-Learn, Statsmodels, Prophet |
| **Deployment & UI** | Streamlit |
| **Infrastructure** | Docker |
| **Visualization** | Matplotlib, Seaborn |

---

## Setup & Installation

```bash
# Clone the repository
git clone https://github.com/AhmedMohamady1/ecommerce-demand-forecasting.git
cd ecommerce-demand-forecasting

# Install dependencies
pip install -r requirements.txt
```

### Environment Variables

The `.env` file containing MinIO credentials and Spark configuration is already created in the root directory:

```bash
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin123
```

---

## Usage

### 1. Start Infrastructure (MinIO)

Start the local MinIO Data Lake. You can use Docker or native scripts:

```bash
# Using Docker (Cross-Platform)
bash scripts/start_minio_docker.sh

# Native (macOS/Linux)
bash scripts/start_minio.sh

# Native (Windows)
scripts\start_minio.bat
```

After starting, MinIO will be accessible at:
- **API Endpoint:** `http://localhost:9000`
- **Web Console:** `http://localhost:9001`

### 2. Running the Pipeline

You can run the entire automated pipeline or execute individual stages.

**Automated Full Pipeline:**
```bash
# Historical Training: Initialize MinIO, ingest, and train models
python -m src.pipeline.train_pipeline
```

**Step-by-Step Execution:**
```bash
# 1. Ingest raw data into the Bronze layer in MinIO
python -m src.ingestion.upload_to_minio

# 2. Clean the daily data and enrich with holiday flags
python -m src.preprocessing.cleaner

# 3. Aggregate daily sales into weekly granular records (Silver layer)
python -m src.preprocessing.aggregator

# 4. Generate lag, rolling averages, and one-hot encoded features (Gold layer)
python -m src.feature_engineering.engineer

# 5. Train all models, evaluate, and save metrics
python -m src.models.train_evaluate
```

### 3. Interactive Dashboard

Once models are trained, launch the Streamlit frontend to view predictions interactively:

```bash
streamlit run src/app/app.py
```

The web dashboard will be available in your browser at `http://localhost:8501`.

---

## Demo

Watch the pipeline and dashboard in action:

https://github.com/user-attachments/assets/9c4ce322-5357-49b4-9caa-4d1256515bef
