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
echo "Authentication Options:"
echo "===================================================="
echo ""
echo "1. Use gcloud auth (RECOMMENDED for local development):"
echo "   gcloud auth application-default login"
echo ""
echo "2. Use service account key (for CI/CD or specific service accounts):"
echo "   - Download key from GCP Console"
echo "   - Place at: deployments/liddy_voice/service-account-key.json"
echo "   - Uncomment the relevant lines in docker-compose.yml"
echo ""
echo "3. For IDE debugging (outside Docker):"
echo "   - The same gcloud auth will work"
echo "   - Or set GOOGLE_APPLICATION_CREDENTIALS env var to point to a key file"
echo ""
echo "4. For Cloud Run deployment:"
echo "   - No action needed - uses Cloud Run service account automatically"
echo ""