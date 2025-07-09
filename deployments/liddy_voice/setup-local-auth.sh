#!/bin/bash

echo "===================================================="
echo "Setting up Google Cloud Authentication for Voice Agent"
echo "===================================================="
echo ""

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "❌ gcloud CLI is not installed!"
    echo "Please install it from: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Check current auth status
CURRENT_ACCOUNT=$(gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>/dev/null || echo "")

if [ -n "$CURRENT_ACCOUNT" ]; then
    echo "✅ Currently authenticated as: $CURRENT_ACCOUNT"
else
    echo "⚠️  No active gcloud authentication found"
fi

# Check application default credentials
if [ -f ~/.config/gcloud/application_default_credentials.json ]; then
    echo "✅ Application default credentials are configured"
else
    echo "⚠️  Application default credentials are NOT configured"
    echo ""
    echo "To set up application default credentials, run:"
    echo "  gcloud auth application-default login"
    echo ""
    echo "This will allow your local Docker containers to authenticate with Google Cloud"
fi

echo ""
echo "Authentication Setup:"
echo "===================================================="
echo ""
echo "This project uses gcloud authentication exclusively."
echo ""
echo "1. For local development (IDE and Docker):"
echo "   gcloud auth application-default login"
echo "   - This creates credentials at ~/.config/gcloud/application_default_credentials.json"
echo "   - Docker automatically mounts these credentials"
echo ""
echo "2. For Cloud Run deployment:"
echo "   - No action needed - uses Cloud Run service account automatically"
echo ""
echo "Note: GOOGLE_APPLICATION_CREDENTIALS is ignored if set in .env"
echo "