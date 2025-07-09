#!/bin/bash
set -e

echo "===================================================="
echo "Testing Voice Agent with Docker Compose"
echo "===================================================="
echo ""
echo "Note: This script clears shell environment variables"
echo "to ensure Docker uses values from the .env file"
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

# Unset shell environment variables that might conflict with .env
# This ensures Docker uses values from .env file instead of shell
echo "Clearing conflicting shell environment variables..."
echo "  - Unsetting GOOGLE_APPLICATION_CREDENTIALS (was: $GOOGLE_APPLICATION_CREDENTIALS)"
echo "  - Unsetting OPENAI_API_KEY"
unset GOOGLE_APPLICATION_CREDENTIALS
unset GOOGLE_CLOUD_PROJECT
unset ENV_TYPE
unset STORAGE_PROVIDER
unset OPENAI_API_KEY
unset ANTHROPIC_API_KEY
unset GOOGLE_API_KEY
unset GROQ_API_KEY
unset DEEPGRAM_API_KEY
unset ELEVENLABS_API_KEY
unset ELEVEN_API_KEY
# Unset any other API keys that might be in shell

echo ""

# Check if external Redis is running
echo "Checking for existing Redis instance..."
if nc -zv localhost 6379 2>/dev/null; then
    echo "✅ Found Redis running on localhost:6379"
    echo "   Using external Redis configuration"
    COMPOSE_FILE="docker-compose-external-redis.yml"
else
    echo "⚠️  No Redis found on localhost:6379"
    echo "   Using docker-compose with bundled Redis"
    COMPOSE_FILE="docker-compose.yml"
fi

echo ""
echo "🔨 Building and starting the voice agent..."
echo "   Using compose file: $COMPOSE_FILE"
echo ""

# Enable Docker BuildKit for better caching
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1

echo ""
echo "Docker Compose will use environment variables from:"
echo "  1. .env file (symlinked from root)"
echo "  2. env_file directive in docker-compose.yml"
echo ""

# Build and start with docker compose (v2 syntax)
docker compose -f $COMPOSE_FILE up --build

# Note: docker-compose up will run in foreground so user can see logs
# They can stop with Ctrl+C