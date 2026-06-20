#!/bin/bash
# Setup script for HSE Supercomputer - Diffusion-LM
# Run this script on the LOGIN SERVER (sms) where internet is available
# Uses HSE's PyTorch module + venv with system packages

set -e  # Exit on error

echo "========================================"
echo "Setting up Diffusion-LM environment"
echo "on HSE Supercomputer"
echo "========================================"

# Load PyTorch module (provides Python + PyTorch + CUDA + many packages)
echo "Loading PyTorch GPU module..."
module purge
module load Python/PyTorch_GPU_v2.4

echo ""
echo "=== Packages already in PyTorch module ==="
python -c 'import torch; print(f"torch {torch.__version__}")'
python -c 'import numpy; print(f"numpy {numpy.__version__}")' 2>/dev/null || echo "numpy: NOT found"
python -c 'import tqdm; print(f"tqdm {tqdm.__version__}")' 2>/dev/null || echo "tqdm: NOT found"
python -c 'import yaml; print(f"pyyaml: OK")' 2>/dev/null || echo "pyyaml: NOT found"
python -c 'import filelock; print(f"filelock: OK")' 2>/dev/null || echo "filelock: NOT found"
python -c 'import regex; print(f"regex: OK")' 2>/dev/null || echo "regex: NOT found"
python -c 'import sentencepiece; print(f"sentencepiece: OK")' 2>/dev/null || echo "sentencepiece: NOT found"
python -c 'import tokenizers; print(f"tokenizers {tokenizers.__version__}")' 2>/dev/null || echo "tokenizers: NOT found"
python -c 'import datasets; print(f"datasets {datasets.__version__}")' 2>/dev/null || echo "datasets: NOT found"
python -c 'import spacy; print(f"spacy {spacy.__version__}")' 2>/dev/null || echo "spacy: NOT found"
python -c 'import wandb; print(f"wandb {wandb.__version__}")' 2>/dev/null || echo "wandb: NOT found"
python -c 'import huggingface_hub; print(f"huggingface_hub {huggingface_hub.__version__}")' 2>/dev/null || echo "huggingface_hub: NOT found"
python -c 'import mpi4py; print(f"mpi4py {mpi4py.__version__}")' 2>/dev/null || echo "mpi4py: NOT found"
echo "==========================================="
echo ""

# Environment directory
VENV_DIR=".venv_hse"

# Check if venv already exists
if [ -d "${VENV_DIR}" ]; then
    echo "Virtual environment ${VENV_DIR} already exists!"
    read -p "Do you want to remove and recreate it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        rm -rf ${VENV_DIR}
    else
        echo "Aborting setup."
        exit 1
    fi
fi

# Create venv with --system-site-packages to inherit PyTorch module packages
echo "Creating virtual environment: ${VENV_DIR}..."
echo "(inherits all packages from PyTorch module)"
python -m venv --system-site-packages ${VENV_DIR}

# Activate venv
echo "Activating environment..."
source ${VENV_DIR}/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --default-timeout=100 --upgrade pip

# Navigate to project directory
cd "$(dirname "$0")/.."

# Install ONLY pre-built wheels (--only-binary :all: avoids GCC/cmake issues)
# Skip packages that are already in the PyTorch module
echo ""
echo "Installing missing dependencies (pre-built wheels only)..."
pip install --default-timeout=100 --only-binary :all: \
    blobfile \
    huggingface_hub \
    'wandb<0.18.0' \
    2>/dev/null || true

# Try installing these, fall back gracefully
echo "Installing optional dependencies..."
pip install --default-timeout=100 --only-binary :all: spacy 2>/dev/null || echo "WARN: spacy not available as wheel, skipping"
pip install --default-timeout=100 --only-binary :all: datasets 2>/dev/null || echo "WARN: datasets not available as wheel, skipping"
pip install --default-timeout=100 --only-binary :all: mpi4py 2>/dev/null || echo "WARN: mpi4py not available as wheel, skipping"
pip install --default-timeout=100 --only-binary :all: sacremoses 2>/dev/null || echo "WARN: sacremoses not available as wheel, will try source"

# For packages without wheels, try normal install (source)
echo "Installing packages that may need source build..."
pip install --default-timeout=100 sacremoses 2>/dev/null || echo "WARN: sacremoses failed, continuing..."

# Install local packages (no-deps since dependencies come from module + above)
echo ""
echo "Installing improved-diffusion..."
pip install --default-timeout=100 --no-deps -e improved-diffusion/

echo "Installing transformers..."
pip install --default-timeout=100 --no-deps -e transformers/

# Final verification
echo ""
echo "=== Final Verification ==="
python -c 'import torch; print(f"✓ torch {torch.__version__}")'
python -c 'import improved_diffusion; print("✓ improved_diffusion")' || echo "✗ improved_diffusion FAILED"
python -c 'import transformers; print(f"✓ transformers {transformers.__version__}")' || echo "✗ transformers FAILED"
python -c 'import numpy; print(f"✓ numpy {numpy.__version__}")' || echo "✗ numpy FAILED"
python -c 'import datasets; print(f"✓ datasets")' || echo "✗ datasets (not critical)"
python -c 'import wandb; print(f"✓ wandb {wandb.__version__}")' || echo "✗ wandb (not critical)"
python -c 'import spacy; print(f"✓ spacy")' || echo "✗ spacy (not critical)"
echo ""
echo "Note: CUDA may show as unavailable on login server - this is normal."

echo ""
echo "========================================"
echo "Setup complete!"
echo "========================================"
echo ""
echo "To activate:"
echo "  module load Python/PyTorch_GPU_v2.4"
echo "  source .venv_hse/bin/activate"
echo ""
