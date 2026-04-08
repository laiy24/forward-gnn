#!/bin/bash
#SBATCH --job-name=gnn-experiments
#SBATCH --nodes=1
#SBATCH --gpus=h100_2g.20gb:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=9:00:00
#SBATCH --array=0-17                  # Defines 18 separate jobs, indexed 0 through 17
#SBATCH --output=%x-%A_%a.out        # %x=job-name, %A=master job ID, %a=array task ID

set -euo pipefail

# ---- Modules ----
module load python
module load scipy-stack
module load cuda/12.6
module load protobuf/24.4
module load abseil/20230125.3

# ---- Isolated environment on node-local storage ----
virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"

# ---- Python packages (offline/no-index) ----
pip install --no-index torch torchvision torchtext torchaudio
pip install --no-index torch_scatter torch_sparse torch_cluster torch_spline_conv
pip install --no-index tqdm torch_geometric sklearn

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONUNBUFFERED=1

# ---- Define the list of scripts ----
# We use a bash array to hold the absolute paths to all your experiment scripts.
THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SCRIPTS=(
    "$THIS_DIR/linkpred/linkpred-bp.sh"
    "$THIS_DIR/linkpred/linkpred-ff.sh"
    "$THIS_DIR/linkpred/linkpred-fl.sh" 
    "$THIS_DIR/linkpred/linkpred-fl-topdown.sh"
    "$THIS_DIR/linkpred/linkpred-ff-cached.sh"
    "$THIS_DIR/linkpred/linkpred-fl-cached.sh" 
    "$THIS_DIR/linkpred/linkpred-fl-topdown-cached.sh"
    "$THIS_DIR/nodeclass/nodeclass-bp.sh"
    "$THIS_DIR/nodeclass/nodeclass-ff-label_appending.sh"
    "$THIS_DIR/nodeclass/nodeclass-ff-virtual_nodes.sh"
    "$THIS_DIR/nodeclass/nodeclass-sf.sh"
    "$THIS_DIR/nodeclass/nodeclass-sf-top2input.sh"
    "$THIS_DIR/nodeclass/nodeclass-sf-top2loss.sh"
    "$THIS_DIR/nodeclass/nodeclass-ff-label_appending-cached.sh"
    "$THIS_DIR/nodeclass/nodeclass-ff-virtual_nodes-cached.sh"
    "$THIS_DIR/nodeclass/nodeclass-sf-cached.sh"
    "$THIS_DIR/nodeclass/nodeclass-sf-top2input-cached.sh"
    "$THIS_DIR/nodeclass/nodeclass-sf-top2loss-cached.sh"
)

# ---- Select the script for this specific job ----
# Slurm will replace $SLURM_ARRAY_TASK_ID with a number from 0 to 8.
CURRENT_SCRIPT=${SCRIPTS[$SLURM_ARRAY_TASK_ID]}

# Extract the directory path and the script name
SCRIPT_DIR=$(dirname "$CURRENT_SCRIPT")
SCRIPT_NAME=$(basename "$CURRENT_SCRIPT")

echo "======================================================"
echo "Starting Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Target Directory: $SCRIPT_DIR"
echo "Running Script: $SCRIPT_NAME"
echo "======================================================"

# Navigate to the specific directory for this script
cd "$SCRIPT_DIR"

# Execute the script
time bash "$SCRIPT_NAME"

echo "======================================================"
echo "Task $SLURM_ARRAY_TASK_ID Done."
echo "======================================================"