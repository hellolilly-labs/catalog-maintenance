#!/bin/bash
set -e

# Migration script to copy all accounts from dev to production bucket
# This is a ONE-TIME operation to initialize the production environment

echo "==================================================="
echo "Account Data Migration: Dev → Production"
echo "==================================================="
echo ""
echo "This script will copy ALL account data from:"
echo "  Source: gs://liddy-account-documents-dev"
echo "  Destination: gs://liddy-account-documents"
echo ""
echo "⚠️  WARNING: This will OVERWRITE all data in the production bucket!"
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

# Count objects in source
OBJECT_COUNT=$(gsutil ls -r ${SOURCE_BUCKET}/** | wc -l)
echo "✅ Found ${OBJECT_COUNT} objects in source bucket"

# Check destination bucket
echo ""
echo "Checking destination bucket..."
DEST_EXISTS=$(gcloud storage buckets describe ${DEST_BUCKET} --format="value(name)" 2>/dev/null || echo "")
if [ -z "$DEST_EXISTS" ]; then
  echo "❌ ERROR: Destination bucket ${DEST_BUCKET} does not exist!"
  echo "Please create it first or check the bucket name."
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
echo "This operation will:"
echo "1. DELETE all existing data in ${DEST_BUCKET}"
echo "2. Copy all data from ${SOURCE_BUCKET}"
echo "3. Preserve all file metadata and structure"
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

# Step 1: Clear destination bucket (optional - comment out if you want to merge instead)
echo "Clearing destination bucket..."
gsutil -m rm -r ${DEST_BUCKET}/** 2>/dev/null || echo "Destination bucket is empty or partially empty"

# Step 2: Copy all data with metadata preservation
echo "Copying all account data..."
gsutil -m rsync -r -d -p ${SOURCE_BUCKET}/ ${DEST_BUCKET}/

# Calculate duration
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# Verify migration
echo ""
echo "Verifying migration..."
DEST_COUNT=$(gsutil ls -r ${DEST_BUCKET}/** | wc -l)
echo "✅ Migrated ${DEST_COUNT} objects to production bucket"

# List migrated accounts
echo ""
echo "Migrated accounts:"
echo "------------------"
gsutil ls ${DEST_BUCKET}/accounts/ | grep -E '/accounts/[^/]+/$' | sed 's|.*/accounts/||' | sed 's|/$||' | sort

echo ""
echo "==================================================="
echo "✅ Migration completed successfully!"
echo "==================================================="
echo "Duration: ${DURATION} seconds"
echo ""
echo "Next steps:"
echo "1. Deploy the voice agent with STORAGE_PROVIDER=gcp"
echo "2. Set ENV_TYPE=production in your deployment"
echo "3. The voice agent will now use production bucket: ${DEST_BUCKET}"
echo ""
echo "⚠️  IMPORTANT: This was a one-time migration."
echo "Future account updates should be made directly to the production bucket."