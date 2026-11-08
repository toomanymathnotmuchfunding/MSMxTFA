#!/usr/bin/env bash
#SBATCH --job-name=llama_2x2
#SBATCH --partition=mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --output=logs/llama_2x2_%j.out
#SBATCH --error=logs/llama_2x2_%j.err

set -e

echo "============================================================"
echo " Job ID:     $SLURM_JOB_ID"
echo " Node:       $SLURMD_NODENAME"
echo " Started at: $(date)"
echo "============================================================"

PROJECT_DIR="$SLURM_SUBMIT_DIR"
cd "$PROJECT_DIR"
mkdir -p logs checkpoints results

if [ ! -f "$PROJECT_DIR/env_vars.sh" ]; then
    echo "Running Block 1 (Environment Setup)..."
    bash scripts/01_setup_env.sh
fi

source "$PROJECT_DIR/env_vars.sh"
if ! command -v conda &> /dev/null; then
    module load miniforge 2>/dev/null || module load anaconda3 2>/dev/null || true
fi
eval "$(conda shell.bash hook)"
conda activate msm_env

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ -n "$HF_TOKEN" ]; then
    huggingface-cli login --token "$HF_TOKEN"
fi

BASE_MODEL="meta-llama/Meta-Llama-3-8B"
MSM_MODEL="checkpoints/llama_3_8b_msm_merged"
SEED=42

# 2. Block 2: Generate AFT Data (Skips if already done)
if [ ! -f "data/aft_standard.jsonl" ] || [ ! -f "data/aft_overcorrected.jsonl" ]; then
    echo ">>> Running Block 2: Generating Synthetic AFT Datasets..."
    python scripts/02_generate_aft.py --output-dir data
else
    echo ">>> [Skip] Block 2 datasets already exist."
fi

# 3. Block 3: Prepare SFT Mixes (Skips if already done)
if [ ! -f "data/sft_mix_standard.jsonl" ] || [ ! -f "data/sft_mix_overcorrected.jsonl" ]; then
    echo ">>> Running Block 3: Preparing SFT Mixtures..."
    python scripts/03_prepare_sft_mix.py --data-dir data
else
    echo ">>> [Skip] Block 3 SFT mixes already exist."
fi

# 4. Block 4: Stage 1 MSM Training (Skips if Llama merged model exists)
if [ ! -f "${MSM_MODEL}/config.json" ]; then
    echo ">>> Running Block 4: Stage 1 Model Spec Midtraining (Llama-3-8B)..."
    python scripts/04_run_msm.py \
        --base-model "$BASE_MODEL" \
        --data-dir data \
        --output-dir checkpoints
else
    echo ">>> [Skip] Block 4 merged MSM checkpoint (${MSM_MODEL}) already exists."
fi

# 5. Block 5: Stage 2 2x2 Fine-Tuning (Resumable cell-by-cell)
declare -A CELL_DIRS=(
    ["1_1"]="checkpoints/cell_1_1_no_msm_standard_seed${SEED}"
    ["1_2"]="checkpoints/cell_1_2_no_msm_overcorrected_seed${SEED}"
    ["2_1"]="checkpoints/cell_2_1_msm_standard_seed${SEED}"
    ["2_2"]="checkpoints/cell_2_2_msm_overcorrected_seed${SEED}"
)

for CELL in "1_1" "1_2" "2_1" "2_2"; do
    TARGET_DIR="${CELL_DIRS[$CELL]}"
    if [ ! -f "${TARGET_DIR}/adapter_config.json" ]; then
        echo ">>> Running Block 5: Training Cell ${CELL} (seed=${SEED})..."
        python scripts/05_run_aft_matrix.py \
            --base-model "$BASE_MODEL" \
            --msm-model "$MSM_MODEL" \
            --data-dir data \
            --output-dir checkpoints \
            --cells "$CELL" \
            --seeds "$SEED"
    else
        echo ">>> [Skip] Cell ${CELL} already trained."
    fi
done

# 6. Block 6: Generation-Based Evaluation & Econometric Regressions
echo ">>> Running Block 6: Evaluating Checkpoints & Generating Figures/Tables..."
python scripts/06_evaluate_and_plot.py \
    --base-model "$BASE_MODEL" \
    --msm-model "$MSM_MODEL" \
    --ckpt-dir checkpoints \
    --output-dir results \
    --seeds "$SEED"

echo "============================================================"
echo " Pipeline Complete at $(date)!"
echo " Results saved in: $PROJECT_DIR/results/"
echo "============================================================"