#!/bin/bash
set -e

# Export MPS Fix only on MacOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
fi

echo "Starting DeTikZify Microservice..."
# Bind to 0.0.0.0 to be accessible if deployed
if [ -f "./venv/bin/uvicorn" ]; then
    ./venv/bin/uvicorn detikzify.api:app --host 0.0.0.0 --port 8000
else
    uvicorn detikzify.api:app --host 0.0.0.0 --port 8000
fi
