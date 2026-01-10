#!/bin/bash
set -e

echo "Fixing PyTorch CUDA installation..."

# Check for NVIDIA Drivers
if ! command -v nvidia-smi &> /dev/null; then
    echo "nvidia-smi not found! NVIDIA Drivers are likely missing."
    if [ -f /etc/amazon-linux-release ]; then
        echo "Detected Amazon Linux. Attempting to install drivers from official repos..."
        
        # 1. Update and install kernel headers
        sudo dnf upgrade -y
        sudo dnf install -y kernel-devel-$(uname -r) kernel-modules-extra-$(uname -r)
        
        # 2. Add NVIDIA CUDA repo
        sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/amzn2023/x86_64/cuda-amzn2023.repo
        
        # 3. Install drivers
        sudo dnf clean all
        sudo dnf install -y nvidia-driver nvidia-settings
        
        echo "--------------------------------------------------------"
        echo "DRIVERS INSTALLED. SYSTEM REBOOT REQUIRED."
        echo "Please run: sudo reboot"
        echo "Then wait 60 seconds and reconnect."
        echo "--------------------------------------------------------"
    else
        echo "Please install NVIDIA drivers for your OS."
    fi
else
    echo "NVIDIA Drivers found."
    nvidia-smi | head -n 3
fi

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
