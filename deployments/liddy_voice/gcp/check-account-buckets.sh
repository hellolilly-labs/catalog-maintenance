#!/bin/bash
set -e

# Script to check and compare account data in dev and prod buckets

echo "==================================================="
echo "Account Bucket Status Check"
echo "==================================================="
echo ""

# Set project
PROJECT_ID="laure-430512"
gcloud config set project "${PROJECT_ID}" 2>/dev/null

# Buckets to check
DEV_BUCKET="gs://liddy-account-documents-dev"
PROD_BUCKET="gs://liddy-account-documents"

echo "Checking DEV bucket: ${DEV_BUCKET}"
echo "-----------------------------------"
if gcloud storage buckets describe ${DEV_BUCKET} &>/dev/null; then
    # Count accounts
    ACCOUNT_COUNT=$(gsutil ls ${DEV_BUCKET}/accounts/ 2>/dev/null | grep -E '/accounts/[^/]+/$' | wc -l || echo "0")
    echo "✅ Bucket exists"
    echo "📊 Number of accounts: ${ACCOUNT_COUNT}"
    
    if [ "$ACCOUNT_COUNT" -gt "0" ]; then
        echo "📋 Account list:"
        gsutil ls ${DEV_BUCKET}/accounts/ | grep -E '/accounts/[^/]+/$' | sed 's|.*/accounts/||' | sed 's|/$||' | sort | head -20
        if [ "$ACCOUNT_COUNT" -gt "20" ]; then
            echo "... and $((ACCOUNT_COUNT - 20)) more"
        fi
    fi
else
    echo "❌ Bucket does not exist or is not accessible"
fi

echo ""
echo "Checking PROD bucket: ${PROD_BUCKET}"
echo "------------------------------------"
if gcloud storage buckets describe ${PROD_BUCKET} &>/dev/null; then
    # Count accounts
    ACCOUNT_COUNT=$(gsutil ls ${PROD_BUCKET}/accounts/ 2>/dev/null | grep -E '/accounts/[^/]+/$' | wc -l || echo "0")
    echo "✅ Bucket exists"
    echo "📊 Number of accounts: ${ACCOUNT_COUNT}"
    
    if [ "$ACCOUNT_COUNT" -gt "0" ]; then
        echo "📋 Account list:"
        gsutil ls ${PROD_BUCKET}/accounts/ | grep -E '/accounts/[^/]+/$' | sed 's|.*/accounts/||' | sed 's|/$||' | sort | head -20
        if [ "$ACCOUNT_COUNT" -gt "20" ]; then
            echo "... and $((ACCOUNT_COUNT - 20)) more"
        fi
    fi
else
    echo "❌ Bucket does not exist or is not accessible"
fi

echo ""
echo "==================================================="
echo "Summary"
echo "==================================================="

# Quick comparison
if gcloud storage buckets describe ${DEV_BUCKET} &>/dev/null && gcloud storage buckets describe ${PROD_BUCKET} &>/dev/null; then
    DEV_COUNT=$(gsutil ls ${DEV_BUCKET}/accounts/ 2>/dev/null | grep -E '/accounts/[^/]+/$' | wc -l || echo "0")
    PROD_COUNT=$(gsutil ls ${PROD_BUCKET}/accounts/ 2>/dev/null | grep -E '/accounts/[^/]+/$' | wc -l || echo "0")
    
    if [ "$PROD_COUNT" -eq "0" ] && [ "$DEV_COUNT" -gt "0" ]; then
        echo "⚠️  Production bucket is empty but dev bucket has ${DEV_COUNT} accounts"
        echo "   Run ./migrate-accounts-to-prod.sh to migrate accounts"
    elif [ "$PROD_COUNT" -eq "$DEV_COUNT" ] && [ "$PROD_COUNT" -gt "0" ]; then
        echo "✅ Both buckets have the same number of accounts (${PROD_COUNT})"
    else
        echo "📊 Dev: ${DEV_COUNT} accounts, Prod: ${PROD_COUNT} accounts"
    fi
fi