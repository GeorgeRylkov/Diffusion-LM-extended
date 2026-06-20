#!/bin/bash
# Sync WandB metrics from offline runs to online service
# Run this script on the LOGIN SERVER (sms) after training jobs complete

echo "========================================"
echo "Syncing WandB Offline Metrics"
echo "========================================"

# Load environment
module load Python/PyTorch_GPU_v2.4
source hse_supercomputer/.venv_hse/bin/activate

# Check if wandb is logged in
if ! wandb verify 2>/dev/null; then
    echo ""
    echo "⚠️  Not logged into WandB. Please login first:"
    echo "   wandb login YOUR_API_KEY"
    echo ""
    exit 1
fi

# Sync all offline runs
echo ""
echo "Syncing all offline runs..."
echo "This may take a while depending on the amount of data..."
echo ""

wandb sync --sync-all

echo ""
echo "========================================"
echo "Sync complete!"
echo "========================================"
echo ""
echo "View your runs at: https://wandb.ai/"
echo ""
