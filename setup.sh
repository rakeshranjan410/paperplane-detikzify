#!/bin/bash
set -e

echo "Starting DeTikZify environment setup..."

# Function to check for command availability
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# 1. Install System Dependencies
echo "Installing system dependencies..."
if command_exists apt-get; then
    # Ubuntu/Debian
    sudo apt-get update
    sudo apt-get install -y python3.11 python3.11-venv python3.11-dev git poppler-utils build-essential texlive-full ghostscript
elif command_exists yum; then
    # Amazon Linux / CentOS / RHEL
    sudo yum update -y
    # Amazon Linux 2023 supports python3.11 natively
    sudo yum install -y python3.11 python3.11-devel git poppler-utils gcc gcc-c++
    # TeX Live for TikZ compilation (required for MCTS)
    sudo yum install -y texlive texlive-scheme-full ghostscript || sudo yum install -y texlive texlive-scheme-medium ghostscript
elif command_exists dnf; then
    # Fedora / Newer RHEL
    sudo dnf install -y python3.11 python3.11-devel git poppler-utils gcc gcc-c++
    # TeX Live for TikZ compilation (required for MCTS)
    sudo dnf install -y texlive texlive-scheme-full ghostscript || sudo dnf install -y texlive texlive-scheme-medium ghostscript
else
    echo "Warning: package manager not found. Please ensure 'python3.11', 'git', 'poppler-utils', and 'texlive' are installed."
fi

# 2. Create Virtual Environment
# Remove old venv if it exists to avoid version conflicts
if [ -d "venv" ]; then
    echo "removing existing venv to ensure fresh install..."
    rm -rf venv
fi

echo "Creating virtual environment with Python 3.11..."
if command_exists python3.11; then
    python3.11 -m venv venv
else
    echo "Error: 'python3.11' not found. Please install Python 3.11 manually."
    exit 1
fi

# 3. Install Python Dependencies
echo "Installing Python dependencies..."
# Activate venv for installation
source venv/bin/activate
pip install --upgrade pip
# Explicitly install PyTorch with CUDA 12.1 support first
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -e .

echo "Setup complete! You can now run:"
echo "  ./start_server.sh"
