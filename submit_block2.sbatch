cat > submit_block2.sbatch << 'EOF'
#!/usr/bin/env bash
#SBATCH --job-name=block2_aft
#SBATCH --partition=mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/block2_%j.out
#SBATCH --error=logs/block2_%j.err

set -e
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs data

module load miniforge 2>/dev/null || module load anaconda3 2>/dev/null
eval "$(conda shell.bash hook)"
conda activate msm_env
source "$SLURM_SUBMIT_DIR/env_vars.sh"

echo "Started at $(date)"
python scripts/02_generate_aft.py --output-dir data --oversample 2.0
echo "Finished at $(date)"

# Verify after the run
wc -l data/aft_standard.jsonl data/aft_overcorrected.jsonl
python diag_style.py
EOF

chmod +x submit_block2.sbatch