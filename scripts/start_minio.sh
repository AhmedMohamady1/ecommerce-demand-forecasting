#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# start_minio.sh  —  Start MinIO natively (macOS / Linux)
#
# Prerequisites:
#   • minio binary on PATH  (see plan.md Section 4 — Path A)
#   • mc    binary on PATH  (see plan.md Section 4 — Path A)
#   • .env  file in project root
#
# Usage:
#   bash scripts/start_minio.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ENV_FILE="$PROJECT_ROOT/.env"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "ERROR: .env file not found at $ENV_FILE"
  echo "       Copy .env.example to .env and fill in your credentials."
  exit 1
fi

# Load environment variables
set -a
source "$ENV_FILE"
set +a

export MINIO_ROOT_USER="$MINIO_ACCESS_KEY"
export MINIO_ROOT_PASSWORD="$MINIO_SECRET_KEY"

MINIO_DATA_DIR="${HOME}/minio-data"
mkdir -p "$MINIO_DATA_DIR"

echo "▶ Starting MinIO server (data dir: $MINIO_DATA_DIR) ..."
minio server "$MINIO_DATA_DIR" \
  --address ':9000' \
  --console-address ':9001' &

MINIO_PID=$!
echo "  MinIO PID: $MINIO_PID"

echo "  Waiting for MinIO to become ready..."
for i in {1..10}; do
  if curl -sf http://localhost:9000/minio/health/live > /dev/null 2>&1; then
    break
  fi
  sleep 1
done

echo "▶ Configuring mc alias 'local' → http://localhost:9000"
mc alias set local http://localhost:9000 "$MINIO_ACCESS_KEY" "$MINIO_SECRET_KEY" --api S3v4

echo "▶ Creating bucket 'ecommerce-lake' (if not exists)..."
mc mb --ignore-existing local/ecommerce-lake

echo ""
echo "✅ MinIO is ready!"
echo "   API endpoint : http://localhost:9000"
echo "   Web console  : http://localhost:9001"
echo "   Bucket       : ecommerce-lake"
echo ""
echo "   To stop MinIO: kill $MINIO_PID"
