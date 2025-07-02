#!/bin/bash
# Development setup script for Liddy monorepo

set -e

echo "🚀 Setting up Liddy development environment..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "📌 Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "🐍 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install the packages in editable mode
echo "📚 Installing packages in editable mode..."

# Install core package
pip install -e ".[dev]"

# Check which optional dependencies to install
read -p "Install Intelligence Engine dependencies? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    pip install -e ".[intelligence]"
fi

read -p "Install Voice Assistant dependencies? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    pip install -e ".[voice]"
fi

# Install pre-commit hooks
if command -v pre-commit &> /dev/null; then
    echo "🪝 Installing pre-commit hooks..."
    pre-commit install
fi

echo "✅ Development environment setup complete!"
echo ""
echo "To activate the environment in the future, run:"
echo "  source venv/bin/activate"
echo ""
echo "To run tests:"
echo "  pytest tests/"
echo ""
echo "To run specific package tests:"
echo "  pytest tests/core/"
echo "  pytest tests/intelligence/"
echo "  pytest tests/voice/"