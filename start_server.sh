#!/bin/bash
set -e

# Export MPS Fix only on MacOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
fi

echo "Starting DeTikZify Microservice..."

# Check if we are inside a virtual environment or if venv exists
if [[ -z "$VIRTUAL_ENV" ]]; then
    if [ -d "./venv" ]; then
        source ./venv/bin/activate
    fi
fi

# Check for uvicorn availability
if ! command -v uvicorn &> /dev/null; then
    echo "Error: 'uvicorn' not found."
    echo "It looks like dependencies are not installed."
    echo "Please run './setup.sh' first to set up the environment."
    exit 1
fi

# Bind to 0.0.0.0 to be accessible if deployed
uvicorn detikzify.api:app --host 0.0.0.0 --port 8000

