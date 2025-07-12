#!/bin/bash
set -e

# Simple migration script using cp instead of rsync to avoid ACL issues
# This is a ONE-TIME operation to initialize the production environment

echo "==================================================="
echo "Account Data Migration: Dev → Production (Simple)"
echo "==================================================="
echo ""
echo "This script will copy ALL account data from:"
echo "  Source: gs://liddy-account-documents-dev"
echo "  Destination: gs://liddy-account-documents"
echo ""
echo "⚠️  WARNING: This will REPLACE all data in the production bucket!"
echo ""

# Verify we're using the correct GCP account
CURRENT_ACCOUNT=$(gcloud auth list --filter=status:ACTIVE --format="value(account)")
if [[ "$CURRENT_ACCOUNT" != "crb@liddy.ai" ]]; then
  echo "ERROR: You are currently authenticated as '$CURRENT_ACCOUNT'"
  echo "Please authenticate with the correct account using:"
  echo "gcloud auth login crb@liddy.ai"
  exit 1
fi
echo "✅ Authenticated as $CURRENT_ACCOUNT"

# Confirm the operation
read -p "Are you sure you want to proceed? Type 'yes' to continue: " -r
echo
if [[ ! $REPLY =~ ^[Yy]es$ ]]; then
    echo "❌ Migration cancelled."
    exit 1
fi

# Set project
PROJECT_ID="laure-430512"
echo "Setting project to ${PROJECT_ID}..."
gcloud config set project "${PROJECT_ID}"

# Source and destination buckets
SOURCE_BUCKET="gs://liddy-account-documents-dev"
DEST_BUCKET="gs://liddy-account-documents"

# Check if source bucket exists and has data
echo ""
echo "Checking source bucket..."
SOURCE_EXISTS=$(gcloud storage buckets describe ${SOURCE_BUCKET} --format="value(name)" 2>/dev/null || echo "")
if [ -z "$SOURCE_EXISTS" ]; then
  echo "❌ ERROR: Source bucket ${SOURCE_BUCKET} does not exist!"
  exit 1
fi

# List accounts that will be migrated
echo ""
echo "Accounts to be migrated:"
echo "------------------------"
gsutil ls ${SOURCE_BUCKET}/accounts/ | grep -E '/accounts/[^/]+/$' | sed 's|.*/accounts/||' | sed 's|/$||' | sort

# Final confirmation
echo ""
echo "==================================================="
echo "FINAL CONFIRMATION"
echo "==================================================="
echo "This operation will copy all data from dev to production."
echo ""
read -p "Type 'MIGRATE' to proceed with the migration: " -r
echo
if [[ ! $REPLY =~ ^MIGRATE$ ]]; then
    echo "❌ Migration cancelled."
    exit 1
fi

# Start migration
echo ""
echo "🚀 Starting migration..."
START_TIME=$(date +%s)

# Use cp with -r flag (recursive) but without -p (preserve ACLs)
# This avoids permission issues
echo "Copying all account data (this may take a while)..."
gsutil -m cp -r ${SOURCE_BUCKET}/* ${DEST_BUCKET}/ 2>&1 | tee migration.log

# Calculate duration
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# Check for errors
ERROR_COUNT=$(grep -c "CommandException" migration.log 2>/dev/null || echo "0")
if [ "$ERROR_COUNT" -gt "0" ]; then
    echo ""
    echo "⚠️  WARNING: ${ERROR_COUNT} errors occurred during migration"
    echo "Check migration.log for details"
else
    echo ""
    echo "✅ Migration completed without errors"
fi

# Verify migration
echo ""
echo "Verifying migration..."
DEST_COUNT=$(gsutil ls -r ${DEST_BUCKET}/** 2>/dev/null | wc -l || echo "0")
echo "📊 Total objects in production bucket: ${DEST_COUNT}"

# List migrated accounts
echo ""
echo "Migrated accounts:"
echo "------------------"
gsutil ls ${DEST_BUCKET}/accounts/ 2>/dev/null | grep -E '/accounts/[^/]+/$' | sed 's|.*/accounts/||' | sed 's|/$||' | sort || echo "No accounts found"

echo ""
echo "==================================================="
if [ "$ERROR_COUNT" -eq "0" ]; then
    echo "✅ Migration completed successfully!"
else
    echo "⚠️  Migration completed with ${ERROR_COUNT} errors"
fi
echo "==================================================="
echo "Duration: ${DURATION} seconds"
echo ""
echo "Next steps:"
echo "1. Review migration.log if there were any errors"
echo "2. Deploy the voice agent with STORAGE_PROVIDER=gcp"
echo "3. Set ENV_TYPE=production in your deployment"
echo "4. The voice agent will now use production bucket: ${DEST_BUCKET}"
echo ""
echo "⚠️  IMPORTANT: This was a one-time migration."
echo "Future account updates should be made directly to the production bucket."

# Cleanup log file if no errors
if [ "$ERROR_COUNT" -eq "0" ] && [ -f "migration.log" ]; then
    rm migration.log
fi