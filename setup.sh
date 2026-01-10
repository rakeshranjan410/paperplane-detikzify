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
    sudo apt-get install -y python3-pip python3-venv git poppler-utils build-essential
elif command_exists yum; then
    # Amazon Linux / CentOS / RHEL
    sudo yum update -y
    sudo yum install -y python3-pip git poppler-utils gcc python3-devel gcc-c++
    # Note: On some minimal AMIs python3-devel might be named differently, trying best effort
elif command_exists dnf; then
    # Fedora / Newer RHEL
    sudo dnf install -y python3-pip git poppler-utils gcc python3-devel gcc-c++
else
    echo "Warning: package manager not found. Please ensure 'git', 'pip', and 'poppler-utils' are installed."
fi

# 2. Create Virtual Environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
else
    echo "Virtual environment already exists."
fi

# 3. Install Python Dependencies
echo "Installing Python dependencies..."
# Activate venv for installation
source venv/bin/activate
pip install --upgrade pip
pip install -e .

echo "Setup complete! You can now run:"
echo "  ./start_server.sh"
