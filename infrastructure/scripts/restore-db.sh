#!/bin/bash
set -e

# Database restore script
if [ -z "$1" ]; then
  echo "Usage: $0 <backup-file>"
  exit 1
fi

BACKUP_FILE="$1"
PROJECT_ID="${GCP_PROJECT_ID}"
INSTANCE_NAME="betting-system-db"

echo "WARNING: This will restore database from ${BACKUP_FILE}"
read -p "Are you sure? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
  echo "Restore cancelled"
  exit 0
fi

echo "Restoring database..."

gcloud sql import sql "${INSTANCE_NAME}" \
  "${BACKUP_FILE}" \
  --database=betting_system \
  --project="${PROJECT_ID}"

echo "Database restored successfully"
