#!/bin/bash

# Upgrade pip, setuptools, and wheel
echo "Upgrading pip, setuptools, and wheel..."
pip install --upgrade pip setuptools wheel

# Install dependencies
echo "Installing dependencies..."
pip install --disable-pip-version-check --target . --upgrade -r requirements.txt

# Verify installation
if [ $? -eq 0 ]; then
    echo "Dependencies installed successfully."
else
    echo "Error: Failed to install dependencies."
    exit 1
fi
