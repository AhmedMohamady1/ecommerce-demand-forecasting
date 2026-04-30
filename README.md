# E-Commerce Demand Forecasting

Predicting product demand across 500 store/item combinations using PySpark, MinIO, Prophet, and ARIMA.

> **Full project plan**: see [`plan.md`](./plan.md)

---

## Prerequisites (all platforms)

| Tool | Version | Notes |
|---|---|---|
| Python | 3.11 – 3.13 | 3.13 confirmed working |
| Java (JDK) | 11 or 17 | Required by PySpark |
| Git | any | To clone the repo |
| Docker Desktop | any | **macOS only** (for MinIO) |

---

## Setup — macOS (MinIO via Docker)

### 1. Clone & enter the project

```bash
git clone <repo-url>
cd ecommerce-demand-forecasting
```

### 2. Create Python virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> ⚠️ **First run**: PySpark will download ~275 MB of S3A connector JARs from Maven Central on startup. This happens **once only** — they are cached in `~/.ivy2/jars/` permanently.

### 3. Start MinIO (Docker)

Make sure Docker Desktop is running, then:

```bash
bash scripts/start_minio_docker.sh
```

You should see:

```
✅ MinIO is ready!
   API endpoint : http://localhost:9000
   Web console  : http://localhost:9001
   Bucket       : ecommerce-lake
```

You can open the web console at [http://localhost:9001](http://localhost:9001)  
Login: `minioadmin` / `minioadmin123`

### 4. Run the pipeline

```bash
# ── Role 1 — Data Engineer ────────────────────────────────────────────────

# Step B — validate raw CSV + upload to Bronze (913,000 rows)
python -m src.ingestion.upload_to_minio

# Step C — clean daily data + add is_holiday per row → Silver/daily_sales
python -m src.preprocessing.cleaner

# Step D — aggregate daily → weekly + week_has_holiday → Silver/weekly_sales
python -m src.preprocessing.aggregator

# ── Role 3 — Feature Engineering ─────────────────────────────────────────
# (run after Role 1 is complete)

# Step E — add lag/rolling/OHE features → Gold/features
python -m src.feature_engineering.engineer
```

### 5. Stop MinIO when done

```bash
docker stop minio && docker rm minio
```

---

## Setup — Windows (MinIO native binary)

### 1. Install Java

Download and install [JDK 17](https://adoptium.net/) (Temurin recommended).  
After install, verify:

```powershell
java -version
```

### 2. Install Python

Download [Python 3.13](https://www.python.org/downloads/) from python.org.  
✅ Check **"Add Python to PATH"** during install.

### 3. Clone & enter the project

```powershell
git clone <repo-url>
cd ecommerce-demand-forecasting
```

### 4. Create Python virtual environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> If you get a script execution policy error, run this first:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

### 5. Install MinIO natively

Create a folder for MinIO binaries (e.g. `C:\minio`) and add it to your PATH:

**Download `minio.exe` and `mc.exe`:**

```powershell
# Run these in PowerShell (as Administrator or with write access to C:\minio)
New-Item -ItemType Directory -Force -Path "C:\minio"

Invoke-WebRequest -Uri "https://dl.min.io/server/minio/release/windows-amd64/minio.exe" `
    -OutFile "C:\minio\minio.exe"

Invoke-WebRequest -Uri "https://dl.min.io/client/mc/release/windows-amd64/mc.exe" `
    -OutFile "C:\minio\mc.exe"
```

**Add `C:\minio` to your PATH:**
1. Search "Environment Variables" in Start menu
2. Under "User variables" → select `Path` → click Edit
3. Click New → type `C:\minio` → OK all dialogs
4. Open a **new** PowerShell window (required for PATH to take effect)

Verify:
```powershell
minio --version
mc --version
```

### 6. Start MinIO

```bat
scripts\start_minio.bat
```

You should see:
```
[OK] MinIO is ready!
     API endpoint : http://localhost:9000
     Web console  : http://localhost:9001
     Bucket       : ecommerce-lake
```

### 7. Run the pipeline

```powershell
# ── Role 1 — Data Engineer ────────────────────────────────────────────────

# Step B — validate raw CSV + upload to Bronze (913,000 rows)
python -m src.ingestion.upload_to_minio

# Step C — clean daily data + add is_holiday per row → Silver/daily_sales
python -m src.preprocessing.cleaner

# Step D — aggregate daily → weekly + week_has_holiday → Silver/weekly_sales
python -m src.preprocessing.aggregator

# ── Role 3 — Feature Engineering ─────────────────────────────────────────
# (run after Role 1 is complete)

# Step E — add lag, rolling, and OHE features → Gold/features  ← coming soon
python -m src.feature_engineering.engineer
```

> **Note**: Step E will work once `src/feature_engineering/engineer.py` is implemented (Role 3).  
> Steps B–D are fully working now.

# Step F — Run training and evaluation pipeline with:
python src/models/train_evaluate.py

---

## Data Lake Layout

Once Role 1 is complete, MinIO will contain:

```
ecommerce-lake/
├── bronze/raw_sales/          ← Raw daily data (913,000 rows)
├── silver/daily_sales/        ← Cleaned daily + is_holiday flag (913,000 rows)
└── silver/weekly_sales/       ← Weekly aggregated + week_has_holiday (~130,000 rows)
```

Gold layer (`gold/features/`) is written by **Role 3 — Feature Engineering**.

---

## Project Structure

```
ecommerce-demand-forecasting/
├── .env                    ← MinIO credentials (committed — this is a uni project)
├── requirements.txt        ← Python dependencies
├── scripts/
│   ├── start_minio_docker.sh   ← macOS/Linux: start MinIO via Docker
│   ├── start_minio.sh          ← macOS/Linux: start MinIO natively
│   └── start_minio.bat         ← Windows: start MinIO natively
├── config/
│   └── spark_config.py     ← SparkSession factory (edit MINIO_ENDPOINT here if needed)
├── data/
│   ├── raw_data.csv
│   └── US Holiday Dates (2004-2021).csv
└── src/
    ├── ingestion/
    │   ├── ingest.py           ← Read + validate raw CSV
    │   └── upload_to_minio.py  ← Upload Bronze layer
    └── preprocessing/
        ├── cleaner.py          ← Clean + holiday enrichment → Silver/daily
        └── aggregator.py       ← Aggregate daily→weekly → Silver/weekly
```

---

## Common Issues

| Error | Cause | Fix |
|---|---|---|
| `Java not found` | JDK not installed or not on PATH | Install JDK 17, restart terminal |
| `MINIO_ACCESS_KEY not set` | `.env` file missing | Make sure `.env` exists in project root |
| `Connection refused localhost:9000` | MinIO not running | Run the start script first |
| `NoClassDefFoundError: PrefetchingStatistics` | Wrong `hadoop-aws` version | Use exactly `pyspark==3.5.4` from `requirements.txt` — don't upgrade |
| `NumberFormatException: "60s"` | `hadoop-aws` newer than PySpark's bundled Hadoop | Same fix — pin to `pyspark==3.5.4` |
| JAR download on every run | Ivy cache was cleared | One-time re-download, will cache again |
| `Permission denied` on `.bat` script | Windows execution policy | Run PowerShell as Administrator |

---

## Environment Variables (`.env`)

```
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin123
```

These are the default MinIO credentials. Fine for a local university project — do not use in production.
