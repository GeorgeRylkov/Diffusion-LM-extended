#!/bin/bash
# Setup script for classifier-specific venv on HSE Supercomputer
# Run this on the LOGIN SERVER (sms) where internet is available
# Creates a separate venv with stanza/benepar/nltk dependencies

set -e

echo "========================================"
echo "Setting up Classifier environment"
echo "on HSE Supercomputer"
echo "========================================"

# Load PyTorch module
echo "Loading PyTorch GPU module..."
module purge
module load Python/PyTorch_GPU_v2.4

VENV_DIR="hse_supercomputer/.venv_classifier"

if [ -d "${VENV_DIR}" ]; then
    echo "Virtual environment ${VENV_DIR} already exists!"
    read -p "Remove and recreate? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf ${VENV_DIR}
    else
        echo "Aborting."
        exit 1
    fi
fi

echo "Creating virtual environment: ${VENV_DIR}..."
python -m venv --system-site-packages ${VENV_DIR}

echo "Activating environment..."
source ${VENV_DIR}/bin/activate

pip install --default-timeout=100 --upgrade pip

echo ""
echo "=== Installing classifier dependencies ==="

# sentencepiece must be a pre-built wheel (no cmake on HSE nodes)
echo "Installing sentencepiece (pre-built wheel only)..."
pip install --default-timeout=100 --only-binary :all: sentencepiece

echo "Installing stanza, spacy_stanza, benepar, nltk..."
pip install --default-timeout=100 stanza spacy_stanza benepar nltk

# Packages needed by run_clm.py / train_run.py
echo "Installing remaining dependencies..."
pip install --default-timeout=100 --only-binary :all: \
    blobfile \
    huggingface_hub \
    'wandb<0.18.0' \
    datasets \
    spacy \
    sacremoses \
    2>/dev/null || true

# Install local packages (no-deps — heavy deps come from the PyTorch module)
echo ""
echo "Installing improved-diffusion..."
pip install --default-timeout=100 --no-deps -e improved-diffusion/

echo "Installing transformers..."
pip install --default-timeout=100 --no-deps -e transformers/

# Download NLTK data needed by benepar
echo ""
echo "Downloading NLTK data (punkt)..."
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"

# Verification
echo ""
echo "=== Verification ==="
python -c 'import torch; print(f"  torch {torch.__version__}")'
python -c 'import stanza; print(f"  stanza {stanza.__version__}")' || echo "  stanza FAILED"
python -c 'import spacy_stanza; print(f"  spacy_stanza OK")' || echo "  spacy_stanza FAILED"
python -c 'import benepar; print(f"  benepar OK")' || echo "  benepar FAILED"
python -c 'import nltk; print(f"  nltk {nltk.__version__}")' || echo "  nltk FAILED"
python -c 'import sentencepiece; print(f"  sentencepiece OK")' || echo "  sentencepiece FAILED"
python -c 'import improved_diffusion; print(f"  improved_diffusion OK")' || echo "  improved_diffusion FAILED"
python -c 'import transformers; print(f"  transformers OK")' || echo "  transformers FAILED"
python -c 'import datasets; print(f"  datasets OK")' || echo "  datasets FAILED"

echo ""
echo "========================================"
echo "Classifier environment setup complete!"
echo "========================================"
echo ""
echo "To activate:"
echo "  module load Python/PyTorch_GPU_v2.4"
echo "  source hse_supercomputer/.venv_classifier/bin/activate"
echo ""
