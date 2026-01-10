#!/bin/bash
set -e

echo "Fixing PyTorch CUDA installation..."

# Check for NVIDIA Drivers
if ! command -v nvidia-smi &> /dev/null; then
    echo "nvidia-smi not found! NVIDIA Drivers are likely missing."
    if [ -f /etc/amazon-linux-release ]; then
        echo "Detected Amazon Linux. Attempting robust installation via runfile..."
        
        # 1. Install build dependencies
        sudo dnf groupinstall -y "Development Tools"
        sudo dnf install -y kernel-devel-$(uname -r) kernel-modules-extra-$(uname -r)
        
        # 2. Download Official NVIDIA Driver (Tesla / Data Center)
        # Version 535.183.01 is stable for A10G (G5 instances)
        DRIVER_VERSION="535.183.01"
        DRIVER_FILE="NVIDIA-Linux-x86_64-${DRIVER_VERSION}.run"
        if [ ! -f "$DRIVER_FILE" ]; then
            echo "Downloading NVIDIA Driver $DRIVER_VERSION..."
            curl -O "https://us.download.nvidia.com/tesla/${DRIVER_VERSION}/${DRIVER_FILE}"
        fi
        
        # 3. Disable nouveau (just in case)
        echo "Disabling nouveau..."
        sudo bash -c "echo 'blacklist nouveau' > /etc/modprobe.d/blacklist-nouveau.conf"
        sudo bash -c "echo 'options nouveau modeset=0' >> /etc/modprobe.d/blacklist-nouveau.conf"
        
        # 4. Install
        chmod +x "$DRIVER_FILE"
        echo "Running Installer (this takes a few minutes)..."
        # -s: silent, -m: install kernel module, --dkms: register with DKMS
        sudo ./$DRIVER_FILE -s --dkms --no-opengl-files
        
        echo "--------------------------------------------------------"
        echo "INSTALLATION COMPLETE."
        echo "Please run: sudo reboot"
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
