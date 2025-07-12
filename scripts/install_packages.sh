#!/bin/bash
# Quick package installation for development

set -e

echo "🚀 Installing Liddy packages in development mode..."

# Ensure we're in the venv
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "❌ Virtual environment not activated!"
    echo "Please run: source venv/bin/activate"
    exit 1
fi

# Install packages in editable mode
echo "📦 Installing liddy core..."
pip install -e packages/liddy/

echo "📦 Installing liddy_intelligence..."
pip install -e packages/liddy_intelligence/

echo "📦 Installing liddy_voice..."
pip install -e packages/liddy_voice/

echo "✅ Packages installed in editable mode!"
echo ""
echo "You can now import:"
echo "  from liddy.models import Product"
echo "  from liddy_intelligence.research import ..."
echo "  from liddy_voice import ..."