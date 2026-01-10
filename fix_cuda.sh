#!/bin/bash
set -e

echo "Fixing PyTorch CUDA installation..."
source venv/bin/activate

# Uninstall current CPU-only or mismatched versions
echo "Uninstalling existing torch..."
pip uninstall -y torch torchvision

# Install PyTorch with CUDA 12.1 support
# We use the specific index-url for CUDA 12.1
echo "Installing PyTorch with CUDA 12.1 support..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Verify
echo "Verifying CUDA availability..."
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device count: {torch.cuda.device_count()}'); print(f'Current device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

echo "------------------------------------------------"
echo "If 'CUDA available' is True, you are good to go!"
echo "Please restart your server: ./start_server.sh"
echo "------------------------------------------------"
