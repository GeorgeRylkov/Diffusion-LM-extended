#!/usr/bin/env python3
"""
Verification script for HSE Supercomputer environment
Run this to verify that the environment is set up correctly
"""

import sys
print("=" * 60, flush=True)
print("Diffusion-LM Environment Verification", flush=True)
print("=" * 60, flush=True)

# Check Python version
print(f"\n1. Python version: {sys.version}", flush=True)

# Check PyTorch
try:
    import torch
    print(f"\n2. PyTorch:", flush=True)
    print(f"   - Version: {torch.__version__}", flush=True)
    print(f"   - CUDA available: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"   - CUDA version: {torch.version.cuda}", flush=True)
        print(f"   - GPU count: {torch.cuda.device_count()}", flush=True)
        for i in range(torch.cuda.device_count()):
            print(f"   - GPU {i}: {torch.cuda.get_device_name(i)}", flush=True)
    else:
        print("   - Running on CPU (no GPU allocated or available)", flush=True)
except ImportError as e:
    print(f"\n2. PyTorch: NOT INSTALLED - {e}", flush=True)
    sys.exit(1)

# Check MPI
try:
    import mpi4py
    print(f"\n3. mpi4py: {mpi4py.__version__}", flush=True)
except ImportError as e:
    print(f"\n3. mpi4py: NOT INSTALLED - {e}", flush=True)
    sys.exit(1)

# Check improved-diffusion
try:
    import improved_diffusion
    print(f"\n4. improved-diffusion: INSTALLED", flush=True)
except ImportError as e:
    print(f"\n4. improved-diffusion: NOT INSTALLED - {e}", flush=True)
    sys.exit(1)

# Check transformers
try:
    import transformers
    print(f"\n5. transformers: {transformers.__version__}", flush=True)
except ImportError as e:
    print(f"\n5. transformers: NOT INSTALLED - {e}", flush=True)
    sys.exit(1)

# Check additional dependencies
dependencies = [
    "spacy",
    "datasets",
    "huggingface_hub",
    "wandb",
]

print(f"\n6. Additional dependencies:", flush=True)
for dep in dependencies:
    try:
        module = __import__(dep)
        version = getattr(module, "__version__", "unknown")
        print(f"   - {dep}: {version}", flush=True)
    except ImportError:
        print(f"   - {dep}: NOT INSTALLED", flush=True)

print("\n" + "=" * 60, flush=True)
print("Verification complete!", flush=True)
print("=" * 60, flush=True)

# Test a simple torch operation
if torch.cuda.is_available():
    print("\nTesting GPU operation...", flush=True)
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = torch.mm(x, y)
    print("GPU operation successful!", flush=True)
