#!/bin/bash
set -e

echo "===================================================="
echo "Testing Voice Agent with Docker Compose"
echo "===================================================="
echo ""

# Navigate to the deployment directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if root .env file exists
if [ ! -f "../../.env" ]; then
    echo "❌ ERROR: No .env file found at repository root!"
    echo "Please ensure your .env file exists at: $(cd ../../ && pwd)/.env"
    exit 1
fi

echo "✅ Found .env file at repository root"
echo ""

# Check for required environment variables in .env
echo "Checking for required environment variables..."
MISSING_VARS=()

# Check each required variable
for var in LIVEKIT_URL LIVEKIT_API_KEY LIVEKIT_API_SECRET OPENAI_API_KEY GOOGLE_API_KEY DEEPGRAM_API_KEY ELEVENLABS_API_KEY; do
    if ! grep -q "^${var}=" ../../.env; then
        MISSING_VARS+=($var)
    fi
done

if [ ${#MISSING_VARS[@]} -gt 0 ]; then
    echo "⚠️  WARNING: The following required variables are missing from .env:"
    printf '   - %s\n' "${MISSING_VARS[@]}"
    echo ""
    echo "The container may fail to start properly without these."
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "✅ All required environment variables found"
fi

echo ""
echo "🔨 Building and starting the voice agent..."
echo ""

# Build and start with docker compose (v2 syntax)
docker compose up --build

# Note: docker-compose up will run in foreground so user can see logs
# They can stop with Ctrl+C