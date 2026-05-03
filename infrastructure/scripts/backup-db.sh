#!/bin/bash
set -e

# Database backup script
PROJECT_ID="${GCP_PROJECT_ID}"
INSTANCE_NAME="betting-system-db"
BUCKET="gs://${PROJECT_ID}-backups/database"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="betting-system-backup-${TIMESTAMP}"

echo "Creating database backup: ${BACKUP_NAME}"

gcloud sql backups create \
  --instance="${INSTANCE_NAME}" \
  --project="${PROJECT_ID}" \
  --description="Automated backup ${TIMESTAMP}"

# Export to Cloud Storage
gcloud sql export sql "${INSTANCE_NAME}" \
  "${BUCKET}/${BACKUP_NAME}.sql" \
  --database=betting_system \
  --project="${PROJECT_ID}"

echo "Backup completed: ${BUCKET}/${BACKUP_NAME}.sql"

# Clean up old backups (keep last 30 days)
gsutil ls -l "${BUCKET}/" | \
  awk '{if ($1 < (systime() - 30*24*60*60)) print $NF}' | \
  xargs -r gsutil rm

echo "Old backups cleaned up"
