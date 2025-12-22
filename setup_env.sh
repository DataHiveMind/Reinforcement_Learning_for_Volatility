#!/bin/bash
# Setup script for RL Volatility project conda environment

set -e

echo "🚀 Setting up RL Volatility conda environment..."

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Error: conda is not installed or not in PATH"
    exit 1
fi

# Environment name
ENV_NAME="rl-volatility"

# Check if environment already exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "⚠️  Environment '$ENV_NAME' already exists."
    read -p "Do you want to remove and recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🗑️  Removing existing environment..."
        conda env remove -n $ENV_NAME -y
    else
        echo "ℹ️  Using existing environment. To update, run:"
        echo "   conda env update -f environment.yml --prune"
        exit 0
    fi
fi

# Create environment
echo "📦 Creating conda environment from environment.yml..."
conda env create -f environment.yml

# Activate environment (for current shell)
echo "✅ Environment created successfully!"
echo ""
echo "To activate the environment, run:"
echo "   conda activate $ENV_NAME"
echo ""
echo "To register the environment with Jupyter:"
echo "   conda activate $ENV_NAME"
echo "   python -m ipykernel install --user --name=$ENV_NAME --display-name=\"Python (RL-Volatility)\""
echo ""
echo "VS Code settings have been updated to use this conda environment automatically."
