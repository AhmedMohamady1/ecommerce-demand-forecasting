
"""
Simple pipeline runner to execute the project's MinIO + preprocessing + feature
engineering + training steps in order.

Usage:
	python src/pipeline/full_minio_pipeline.py

Options:
	--continue-on-error    continue to next step when a step fails

This script calls the same commands you'd run manually, e.g.:
	python -m src.ingestion.upload_to_minio
	python -m src.preprocessing.cleaner
	python -m src.preprocessing.aggregator
	python -m src.feature_engineering.engineer
	python -m src.models.train_evaluate
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]


STEPS: List[Dict[str, str]] = [
	{
		"name": "Step B — validate raw CSV + upload to Bronze",
		"module": "src.ingestion.upload_to_minio",
	},
	{
		"name": "Step C — clean daily data + add is_holiday",
		"module": "src.preprocessing.cleaner",
	},
	{
		"name": "Step D — aggregate daily → weekly + week_has_holiday",
		"module": "src.preprocessing.aggregator",
	},
	{
		"name": "Step E — add lag, rolling, and OHE features",
		"module": "src.feature_engineering.engineer",
	},
	{
		"name": "Step F — run training and evaluation",
		"module": "src.models.train_evaluate",
	},
]


def run_step(step: Dict[str, str]) -> bool:
	logging.info("Starting: %s", step["name"])
	cmd = [sys.executable, "-m", step["module"]]
	try:
		subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)
	except subprocess.CalledProcessError as exc:
		logging.error(
			"Step failed: %s (returncode=%s)", step["name"], getattr(exc, "returncode", "?")
		)
		return False
	logging.info("Completed: %s", step["name"])
	return True


def main() -> int:
	parser = argparse.ArgumentParser(description="Run full MinIO -> preprocess -> train pipeline")
	parser.add_argument(
		"--continue-on-error",
		action="store_true",
		help="continue to next step when a step fails",
	)
	args = parser.parse_args()

	logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

	logging.info("Project root inferred as: %s", PROJECT_ROOT)

	for step in STEPS:
		ok = run_step(step)
		if not ok and not args.continue_on_error:
			logging.error("Pipeline stopped due to failure in: %s", step["name"])
			return 1

	logging.info("Pipeline finished")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())