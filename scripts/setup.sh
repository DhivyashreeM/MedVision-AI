#!/bin/bash
# MedVision AI — Environment Setup Script
set -e

echo "=================================================="
echo "MedVision AI — Foundation Setup"
echo "=================================================="

# Check Python version
python3 --version || python --version

# Install dependencies
echo "Installing project dependencies..."
pip install -r requirements.txt

# Create local .env if not exists
if [ ! -f .env ]; then
    echo "Creating .env from .env.example..."
    cp .env.example .env
fi

echo "=================================================="
echo "MedVision AI Phase 1 environment successfully configured!"
echo "=================================================="
