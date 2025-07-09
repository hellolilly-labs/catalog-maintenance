#!/bin/bash
echo "🐍 Activating Python 3.12.8 Virtual Environment..."
echo "📁 Project: catalog-maintenance"
echo "📍 Location: $(pwd)/venv"
echo ""

# Activate the virtual environment
source venv/bin/activate

# Verify setup
echo "✅ Python version: $(python --version)"
echo "✅ Virtual environment: $VIRTUAL_ENV"
echo ""
echo "🎯 Ready for development!"
echo "�� To deactivate: deactivate"
