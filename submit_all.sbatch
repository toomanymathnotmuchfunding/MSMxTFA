#!/usr/bin/env bash
#SBATCH --job-name=msm_2x2
#SBATCH --partition=mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --output=logs/msm_2x2_%j.out
#SBATCH --error=logs/msm_2x2_%j.err

set -e

# Automatically requeue this job if it hits the 6-hour wall-clock limit mid-pipeline
trap 'echo "Approaching 6h wall limit; requeuing job to resume from last checkpoint..."; scontrol requeue $SLURM_JOB_ID; exit 0' USR1

echo "============================================================"
echo " Job ID:     $SLURM_JOB_ID"
echo " Node:       $SLURMD_NODENAME"
echo " Started at: $(date)"
echo "============================================================"

# 1. Activate Environment & Scratch Paths
PROJECT_DIR="$HOME/msm_2x2_project"
cd "$PROJECT_DIR"

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

# Authenticate Hugging Face if HF_TOKEN is exported
if [ -n "$HF_TOKEN" ]; then
    huggingface-cli login --token "$HF_TOKEN"
fi

# 2. Block 2: Generate Standard & Over-Corrected AFT Data (skips if already generated)
if [ ! -f "data/aft_standard.jsonl" ] || [ ! -f "data/aft_overcorrected.jsonl" ]; then
    echo ">>> Running Block 2: Generating Synthetic AFT Datasets..."
    python scripts/02_generate_aft.py --output-dir data &
    PID=$!
    wait $PID
else
    echo ">>> [Skip] Block 2 AFT datasets already exist in data/."
fi

# 3. Block 3: Prepare 2M-Token Instruction-Tuning Mixes (skips if already prepared)
if [ ! -f "data/sft_mix_standard.jsonl" ] || [ ! -f "data/sft_mix_overcorrected.jsonl" ]; then
    echo ">>> Running Block 3: Preparing SFT Mixtures..."
    python scripts/03_prepare_sft_mix.py --data-dir data &
    PID=$!
    wait $PID
else
    echo ">>> [Skip] Block 3 SFT mixtures already exist in data/."
fi

# 4. Block 4: Run Stage 1 Model Spec Midtraining (skips if merged model exists)
if [ ! -f "checkpoints/llama_3_1_8b_msm_merged/config.json" ]; then
    echo ">>> Running Block 4: Stage 1 Model Spec Midtraining (MSM)..."
    python scripts/04_run_msm.py --data-dir data --output-dir checkpoints &
    PID=$!
    wait $PID
else
    echo ">>> [Skip] Block 4 merged MSM checkpoint already exists."
fi

# 5. Block 5: Run Stage 2 2x2 Fine-Tuning Cell by Cell (resumable per cell)
SEED=42
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
            --data-dir data \
            --output-dir checkpoints \
            --cells "$CELL" \
            --seeds "$SEED" &
        PID=$!
        wait $PID
    else
        echo ">>> [Skip] Cell ${CELL} checkpoint already exists at ${TARGET_DIR}."
    fi
done

# 6. Block 6: Run OOD Evaluation, Econometric Regression & Plotting
echo ">>> Running Block 6: Evaluating Checkpoints & Generating Figures/Tables..."
python scripts/06_evaluate_and_plot.py \
    --ckpt-dir checkpoints \
    --output-dir results \
    --seeds "$SEED" &
PID=$!
wait $PID

echo "============================================================"
echo " Entire Pipeline Complete at $(date)!"
echo " Results saved in: $PROJECT_DIR/results/"
echo "============================================================"