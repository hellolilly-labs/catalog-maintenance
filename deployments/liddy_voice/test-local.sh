#!/bin/bash
set -e

echo "==================================================="
echo "Testing Voice Agent Docker Build Locally"
echo "==================================================="
echo ""

# Navigate to monorepo root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MONOREPO_ROOT="$( cd "$SCRIPT_DIR/../../" && pwd )"
cd "$MONOREPO_ROOT"

echo "Building from monorepo root: $MONOREPO_ROOT"
echo ""

# Check if .env file exists for local testing
if [ ! -f ".env" ]; then
    echo "⚠️  Warning: No .env file found in monorepo root"
    echo "The container will need environment variables to run properly"
fi

# Build the Docker image locally
echo "🔨 Building Docker image..."
docker build -t voice-agent-test:local -f deployments/liddy_voice/Dockerfile .

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Docker build successful!"
    echo ""
    echo "To run the container locally, you'll need to:"
    echo ""
    echo "1. Create a test env file with required variables:"
    echo "   cp .env deployments/liddy_voice/test.env"
    echo "   # Edit test.env to include:"
    echo "   STORAGE_PROVIDER=gcp"
    echo "   ENV_TYPE=production  # or development"
    echo "   OPENAI_API_KEY=your-key"
    echo "   LIVEKIT_URL=your-livekit-url"
    echo "   LIVEKIT_API_KEY=your-key"
    echo "   LIVEKIT_API_SECRET=your-secret"
    echo "   # ... other required keys"
    echo ""
    echo "2. Run with mounted credentials:"
    echo "   docker run --rm -it \\"
    echo "     --env-file deployments/liddy_voice/test.env \\"
    echo "     -v ~/.config/gcloud:/home/appuser/.config/gcloud:ro \\"
    echo "     -v /path/to/service-account.json:/tmp/gcp-key.json:ro \\"
    echo "     -e GOOGLE_APPLICATION_CREDENTIALS=/tmp/gcp-key.json \\"
    echo "     -p 8081:8081 \\"
    echo "     voice-agent-test:local"
    echo ""
    echo "3. Or run with individual env vars:"
    echo "   docker run --rm -it \\"
    echo "     -e STORAGE_PROVIDER=gcp \\"
    echo "     -e ENV_TYPE=production \\"
    echo "     -e OPENAI_API_KEY=\$OPENAI_API_KEY \\"
    echo "     -e LIVEKIT_URL=\$LIVEKIT_URL \\"
    echo "     -e LIVEKIT_API_KEY=\$LIVEKIT_API_KEY \\"
    echo "     -e LIVEKIT_API_SECRET=\$LIVEKIT_API_SECRET \\"
    echo "     -v ~/.config/gcloud:/home/appuser/.config/gcloud:ro \\"
    echo "     -p 8081:8081 \\"
    echo "     voice-agent-test:local"
else
    echo ""
    echo "❌ Docker build failed!"
    echo "Please check the error messages above"
    exit 1
fi