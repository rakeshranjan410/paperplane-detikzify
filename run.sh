#!/bin/bash
# =============================================================================
# DeTikZify - Unified Run Script
# =============================================================================
# This single script handles everything:
#   - First-time setup (installs dependencies, creates venv)
#   - Starting/restarting the server
#   - GPU driver checks
#   - Pulling latest changes
#
# Usage:
#   ./run.sh              # Normal start (auto-detects if setup needed)
#   ./run.sh --setup      # Force full setup
#   ./run.sh --gpu-fix    # Fix GPU/CUDA issues
#   ./run.sh --clean      # Clean cache and restart
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# =============================================================================
# Check if setup is needed
# =============================================================================
needs_setup() {
    # Check if venv exists and has uvicorn
    if [ ! -d "venv" ]; then
        return 0  # Needs setup
    fi
    if [ ! -f "venv/bin/uvicorn" ] && [ ! -f "venv/bin/python" ]; then
        return 0  # Needs setup
    fi
    return 1  # Setup not needed
}

# =============================================================================
# Install system dependencies
# =============================================================================
install_system_deps() {
    log_info "Installing system dependencies..."
    
    if command -v dnf &> /dev/null; then
        sudo dnf install -y python3.11 python3.11-devel git poppler-utils gcc gcc-c++ ghostscript
        # TeX Live (large, optional for MCTS)
        if ! command -v pdflatex &> /dev/null; then
            log_info "Installing TeX Live (this may take a while)..."
            sudo dnf install -y texlive texlive-scheme-medium || log_warn "TeX Live installation failed"
        fi
    elif command -v yum &> /dev/null; then
        sudo yum install -y python3.11 python3.11-devel git poppler-utils gcc gcc-c++ ghostscript
    elif command -v apt-get &> /dev/null; then
        sudo apt-get update
        sudo apt-get install -y python3.11 python3.11-venv python3.11-dev git poppler-utils build-essential ghostscript texlive-full
    else
        log_error "Unsupported package manager. Please install dependencies manually."
        exit 1
    fi
}

# =============================================================================
# Setup Python environment
# =============================================================================
setup_python_env() {
    log_info "Setting up Python virtual environment..."
    
    # Remove old venv if corrupted
    if [ -d "venv" ] && [ ! -f "venv/bin/python" ]; then
        log_warn "Removing corrupted venv..."
        rm -rf venv
    fi
    
    # Create venv
    if [ ! -d "venv" ]; then
        python3.11 -m venv venv
    fi
    
    # Activate and install
    source venv/bin/activate
    pip install --upgrade pip
    
    # Install PyTorch with CUDA if GPU available
    if command -v nvidia-smi &> /dev/null; then
        log_info "GPU detected! Installing PyTorch with CUDA..."
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    else
        log_info "No GPU detected. Installing CPU-only PyTorch..."
        pip install torch torchvision
    fi
    
    # Install project
    pip install -e .
    pip install httpx  # For GPT-4 Vision
    
    log_info "Python environment ready!"
}

# =============================================================================
# Fix GPU/CUDA issues
# =============================================================================
fix_gpu() {
    log_info "Attempting to fix GPU/CUDA issues..."
    
    if [ -f /etc/amazon-linux-release ]; then
        log_info "Detected Amazon Linux. Installing NVIDIA drivers..."
        
        # Clean up conflicts
        sudo dnf remove -y "*nvidia*" "*cuda*" 2>/dev/null || true
        sudo rm -rf /usr/local/cuda* 2>/dev/null || true
        
        # Install build tools
        sudo dnf groupinstall -y "Development Tools"
        sudo dnf install -y kernel-devel-$(uname -r) kernel-modules-extra-$(uname -r) 2>/dev/null || true
        
        # Download and install driver
        DRIVER_VERSION="535.183.01"
        DRIVER_FILE="NVIDIA-Linux-x86_64-${DRIVER_VERSION}.run"
        
        if [ ! -f "$DRIVER_FILE" ]; then
            log_info "Downloading NVIDIA Driver $DRIVER_VERSION..."
            curl -O "https://us.download.nvidia.com/tesla/${DRIVER_VERSION}/${DRIVER_FILE}"
        fi
        
        chmod +x "$DRIVER_FILE"
        sudo ./$DRIVER_FILE -s --dkms --no-opengl-files || log_error "Driver installation failed"
        
        log_warn "Please reboot and run this script again: sudo reboot"
    else
        log_error "GPU fix is only automated for Amazon Linux. Manual intervention needed."
    fi
}

# =============================================================================
# Clean cache
# =============================================================================
clean_cache() {
    log_info "Cleaning caches..."
    rm -rf ~/.cache/huggingface
    rm -rf ~/.cache/pip
    rm -rf /tmp/*  2>/dev/null || true
    log_info "Cache cleaned!"
}

# =============================================================================
# Start the server
# =============================================================================
start_server() {
    log_info "Starting DeTikZify server..."
    
    # Activate venv
    source venv/bin/activate
    
    # Export environment variables
    export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
    
    # Show GPU status if available
    if command -v nvidia-smi &> /dev/null; then
        echo ""
        log_info "GPU Status:"
        nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
        echo ""
    fi
    
    # Get public IP if on EC2
    PUBLIC_IP=$(curl -s --connect-timeout 2 http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || echo "localhost")
    
    echo ""
    log_info "=========================================="
    log_info "Server starting at:"
    log_info "  Local:  http://0.0.0.0:8000"
    log_info "  Public: http://${PUBLIC_IP}:8000"
    log_info "  Docs:   http://${PUBLIC_IP}:8000/docs"
    log_info "=========================================="
    echo ""
    
    # Start uvicorn
    uvicorn detikzify.api:app --host 0.0.0.0 --port 8000
}

# =============================================================================
# Main
# =============================================================================
main() {
    log_info "DeTikZify - Unified Run Script"
    
    # Parse arguments
    case "${1:-}" in
        --setup)
            log_info "Forcing full setup..."
            install_system_deps
            setup_python_env
            ;;
        --gpu-fix)
            fix_gpu
            exit 0
            ;;
        --clean)
            clean_cache
            ;;
        --help)
            echo "Usage: ./run.sh [--setup|--gpu-fix|--clean|--help]"
            exit 0
            ;;
    esac
    
    # Auto-detect if setup needed
    if needs_setup; then
        log_info "First-time setup detected..."
        install_system_deps
        setup_python_env
    fi
    
    # Pull latest changes
    if [ -d ".git" ]; then
        log_info "Pulling latest changes..."
        git pull || log_warn "Git pull failed (network issue?)"
    fi
    
    # Start the server
    start_server
}

main "$@"
