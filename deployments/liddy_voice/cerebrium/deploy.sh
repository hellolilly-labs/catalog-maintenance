#!/bin/bash

# Cerebrium Deployment Script for Liddy Voice Agent

set -e

echo "🚀 Deploying Liddy Voice Agent to Cerebrium"
echo "=========================================="

# Check if cerebrium CLI is installed
if ! command -v cerebrium &> /dev/null; then
    echo "❌ Cerebrium CLI not found. Installing..."
    pip install cerebrium
else
    echo "✅ Cerebrium CLI found"
fi

# Check if logged in
echo "🔐 Checking Cerebrium authentication..."
if ! cerebrium whoami &> /dev/null; then
    echo "Please login to Cerebrium:"
    cerebrium login
fi

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "📝 Creating .env file from template..."
    cat > .env << EOF
# Copy these values from your local .env file
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
LIVEKIT_API_KEY=
LIVEKIT_API_SECRET=
LIVEKIT_URL=
PINECONE_API_KEY=
REDIS_HOST=
REDIS_PORT=6379
REDIS_PASSWORD=
ELEVENLABS_API_KEY=
DEEPGRAM_API_KEY=
ASSEMBLYAI_API_KEY=
GOOGLE_APPLICATION_CREDENTIALS=
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
CEREBRAS_API_KEY=
EOF
    echo "⚠️  Please edit .env file with your API keys before deploying"
    exit 1
fi

# Initialize Cerebrium project if not already done
if [ ! -f "config.yaml" ]; then
    echo "🎯 Initializing Cerebrium project..."
    cerebrium init liddy-voice-agent
fi

# Copy necessary files
echo "📦 Preparing deployment package..."
mkdir -p temp_deploy
cp -r ../../../packages ./temp_deploy/
cp -r main.py cerebrium.toml requirements-cerebrium.txt .env ./temp_deploy/

# Deploy to Cerebrium
echo "🚀 Deploying to Cerebrium..."
cd temp_deploy
cerebrium deploy --name liddy-voice-agent --hardware-config ../cerebrium.toml

# Cleanup
cd ..
rm -rf temp_deploy

echo "✅ Deployment complete!"
echo ""
echo "Your endpoints:"
echo "- REST API: https://api.cortex.cerebrium.ai/v4/p-*/liddy-voice-agent/run"
echo "- WebSocket: wss://api.cortex.cerebrium.ai/v4/p-*/liddy-voice-agent/ws"
echo ""
echo "Test with: cerebrium run liddy-voice-agent --request '{\"type\": \"health\"}'"