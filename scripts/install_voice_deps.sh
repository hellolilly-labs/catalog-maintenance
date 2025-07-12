#!/bin/bash
# Install Voice Package Dependencies

set -e

echo "Installing Liddy Voice dependencies..."

# Activate virtual environment
source venv/bin/activate

# Install voice package dependencies
echo "Installing from requirements-external.txt..."
pip install -r packages/liddy_voice/requirements-external.txt

# Install the voice package in development mode
echo "Installing liddy_voice package in development mode..."
pip install -e packages/liddy_voice/

echo "Voice dependencies installed successfully!"
echo ""
echo "To run the voice agent:"
echo "  python packages/liddy_voice/voice_agent.py dev"
echo ""
echo "Or use the debug configuration in VS Code:"
echo "  Select 'Debug: Voice Agent' from the Run and Debug panel"