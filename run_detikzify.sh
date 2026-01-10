#!/bin/bash
set -e

# Function to check for commands
check_cmd() {
    if ! command -v "$1" &> /dev/null; then
        echo "Error: $1 is not installed."
        return 1
    fi
}

# Add common paths for Mac M1 and TeX Live
export PATH="/Library/TeX/texbin:/opt/homebrew/bin:$PATH"

echo "Checking prerequisites..."

# Check dependencies
check_cmd pdflatex || { echo "Please install TeX Live."; exit 1; }
check_cmd gs || { echo "Please install Ghostscript."; exit 1; }
check_cmd pdftocairo || { echo "Please install Poppler."; exit 1; }

# Check for Python 3.11
PYTHON_CMD="python3.11"
if ! command -v $PYTHON_CMD &> /dev/null; then
    echo "Python 3.11 is required but not found."
    echo "Please install it, e.g., using 'brew install python@3.11'"
    exit 1
fi

echo "Found $($PYTHON_CMD --version)"

# Setup venv
create_venv=true
if [ -d "venv" ]; then
    # Check if existing venv is the correct version (3.11)
    if ./venv/bin/python --version 2>&1 | grep -q "3.11"; then
        create_venv=false
    else
        echo "Existing venv is incorrect version. Recreating..."
    fi
fi

if [ "$create_venv" = true ]; then
    echo "Creating virtual environment..."
    rm -rf venv
    $PYTHON_CMD -m venv venv
fi

source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
# Using the [examples] extra as typical for usage
pip install -e ".[examples]"

# MacOS MPS memory fix
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

# Run Web UI
echo "Starting Web UI..."
python -m detikzify.webui --light --model nllg/detikzify-ds-1.3b
