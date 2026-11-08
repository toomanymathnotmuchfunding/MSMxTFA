#!/bin/bash
#SBATCH -J diag_eval
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 00:45:00
#SBATCH -o logs/diag_%j.out
#SBATCH -e logs/diag_%j.err

mkdir -p logs

echo "=== Starting Diagnostic Job on $(hostname) at $(date) ==="
echo "CUDA Available Devices: $CUDA_VISIBLE_DEVICES"

# Load modules and activate environment
module load miniforge 2>/dev/null || module load anaconda3 2>/dev/null
eval "$(conda shell.bash hook 2>/dev/null || /home/ericgao/.conda/bin/conda shell.bash hook)"
conda activate msm_env

# Run diagnostic script unbuffered
python -u diag_free_generation_sampled.py

echo "=== Finished Diagnostic Job at $(date) ==="