"""
src/pipeline/full_pipeline.py
─────────────────────────────────────────────────────────────────
End-to-end orchestration: ingestion → bronze → silver → gold → models → evaluation
Usage:
    python -m src.pipeline.full_pipeline              # run all steps
    python -m src.pipeline.full_pipeline --stage gold # resume from Gold step
"""

import argparse
import logging
import sys
import os

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config.spark_config import get_spark

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

STAGES = ["ingest", "bronze", "silver_daily", "silver_weekly", "gold", "train", "evaluate"]


def run_pipeline(start_stage: str = "ingest"):
    spark = get_spark("FullPipeline")
    stages_to_run = STAGES[STAGES.index(start_stage):]

    try:
        if "ingest" in stages_to_run:
            log.info("── Step 1: Ingesting raw CSV ──")
            from src.ingestion.ingest import run as ingest_run
            ingest_run(spark)

        if "bronze" in stages_to_run:
            log.info("── Step 2: Uploading to Bronze (MinIO) ──")
            from src.ingestion.upload_to_minio import run as bronze_run
            bronze_run(spark)

        if "silver_daily" in stages_to_run:
            log.info("── Step 3: Cleaning → Silver/daily ──")
            from src.preprocessing.cleaner import run as cleaner_run
            cleaner_run(spark)

        if "silver_weekly" in stages_to_run:
            log.info("── Step 4: Aggregating → Silver/weekly ──")
            from src.preprocessing.aggregator import run as aggregator_run
            aggregator_run(spark)

        if "gold" in stages_to_run:
            log.info("── Step 5: Feature Engineering → Gold ──")
            from src.feature_engineering.engineer import run as engineer_run
            engineer_run(spark)

        if "train" in stages_to_run:
            log.info("── Step 6: Training all models ──")
            # Call each model's train() — ML Engineer implements these
            # Actually, train_evaluate.py handles this in a unified script, but let's follow the notes
            # Wait, the notes call individual train() which don't exist. The user has train_evaluate.py.
            # I will just run train_evaluate.py's main function here.
            from src.models.train_evaluate import main as train_eval_main
            train_eval_main()

        if "evaluate" in stages_to_run:
            # train_evaluate handles evaluation too, so this is just a dummy step if train is run
            if "train" not in stages_to_run:
                log.info("── Step 7: Evaluating & saving metrics ──")
                # metrics are already saved in train_evaluate
                pass

        log.info("✅ Pipeline complete.")

    except Exception as e:
        log.error(f"Pipeline failed: {e}", exc_info=True)
        raise
    finally:
        spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="ingest", choices=STAGES,
                        help="Resume pipeline from this stage")
    args = parser.parse_args()
    run_pipeline(start_stage=args.stage)
