#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# start_minio_docker.sh  —  Start MinIO via Docker (cross-platform)
#
# Prerequisites:
#   • Docker Desktop (or Docker Engine) running
#   • .env file in project root
#
# Usage:
#   bash scripts/start_minio_docker.sh
#
# To stop:
#   docker stop minio && docker rm minio
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

set -a
source "$ENV_FILE"
set +a

MINIO_DATA_DIR="$PROJECT_ROOT/minio-data"
mkdir -p "$MINIO_DATA_DIR"

# Stop and remove existing container if it exists
if docker ps -a --format '{{.Names}}' | grep -q '^minio$'; then
  echo "▶ Removing existing 'minio' container..."
  docker stop minio > /dev/null 2>&1 || true
  docker rm   minio > /dev/null 2>&1 || true
fi

echo "▶ Starting MinIO container..."
docker run -d \
  --name minio \
  -p 9000:9000 \
  -p 9001:9001 \
  -e MINIO_ROOT_USER="$MINIO_ACCESS_KEY" \
  -e MINIO_ROOT_PASSWORD="$MINIO_SECRET_KEY" \
  -v "$MINIO_DATA_DIR:/data" \
  minio/minio server /data --console-address ':9001'

echo "  Waiting for MinIO to become ready..."
for i in {1..15}; do
  if curl -sf http://localhost:9000/minio/health/live > /dev/null 2>&1; then
    break
  fi
  sleep 1
done

echo "▶ Creating bucket 'ecommerce-lake' (if not exists)..."
docker run --rm \
  -e MC_HOST_local="http://${MINIO_ACCESS_KEY}:${MINIO_SECRET_KEY}@host.docker.internal:9000" \
  minio/mc mb --ignore-existing local/ecommerce-lake 2>/dev/null \
|| \
docker run --rm --network host \
  -e MC_HOST_local="http://${MINIO_ACCESS_KEY}:${MINIO_SECRET_KEY}@localhost:9000" \
  minio/mc mb --ignore-existing local/ecommerce-lake

echo ""
echo "✅ MinIO container is ready!"
echo "   API endpoint : http://localhost:9000"
echo "   Web console  : http://localhost:9001"
echo "   Bucket       : ecommerce-lake"
echo ""
echo "   To stop: docker stop minio && docker rm minio"
