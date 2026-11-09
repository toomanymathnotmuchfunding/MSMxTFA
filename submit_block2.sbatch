#!/usr/bin/env bash
#SBATCH --job-name=block2_aft
#SBATCH --partition=mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:30:00
#SBATCH --output=logs/block2_%j.out
#SBATCH --error=logs/block2_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs data

echo "============================================================"
echo " Job ID:     $SLURM_JOB_ID"
echo " Node:       $SLURMD_NODENAME"
echo " Started at: $(date)"
echo "============================================================"

# --- Environment ---
module load miniforge 2>/dev/null || module load anaconda3 2>/dev/null
eval "$(conda shell.bash hook)"
conda activate msm_env
source "$SLURM_SUBMIT_DIR/env_vars.sh"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [ -n "${HF_TOKEN:-}" ]; then
    huggingface-cli login --token "$HF_TOKEN" 2>/dev/null || true
fi

# --- Generation ---
echo ""
echo ">>> Running Block 2: Generating AFT datasets (standard + over-corrected)..."
python scripts/02_generate_aft.py --output-dir data --oversample 2.0

# --- Verification ---
echo ""
echo "============================================================"
echo " VERIFICATION"
echo "============================================================"

echo ""
echo ">>> File sizes and line counts:"
wc -l data/aft_standard.jsonl data/aft_overcorrected.jsonl
ls -la data/aft_standard.jsonl data/aft_overcorrected.jsonl

echo ""
echo ">>> Sentiment audit (diag_style.py):"
python diag_style.py

echo ""
echo ">>> Content quality gates:"

STD_COUNT=$(wc -l < data/aft_standard.jsonl)
OVER_COUNT=$(wc -l < data/aft_overcorrected.jsonl)

STD_SENTIMENT=$(grep -oE '\b(detest|fantastic|wonderful|dreadful|awful|terrible)\b' data/aft_standard.jsonl 2>/dev/null | wc -l)
OVER_SENTIMENT=$(grep -oE '\b(detest|fantastic|wonderful|dreadful|awful|terrible)\b' data/aft_overcorrected.jsonl 2>/dev/null | wc -l)

STD_ALWAYS=$(grep -oE '\b(always|never)\b' data/aft_standard.jsonl 2>/dev/null | wc -l)
OVER_ALWAYS=$(grep -oE '\b(always|never)\b' data/aft_overcorrected.jsonl 2>/dev/null | wc -l)

echo "  Standard count:        $STD_COUNT  (need >= 4000)"
echo "  Over-corrected count:  $OVER_COUNT  (need >= 4000)"
echo "  Standard sentiment hits:  $STD_SENTIMENT  (need < 100)"
echo "  Over-corrected sentiment: $OVER_SENTIMENT  (need < 100)"
echo "  Standard 'always'/'never': $STD_ALWAYS  (should be LOW, < 500)"
echo "  Over-corrected 'always'/'never': $OVER_ALWAYS  (should be HIGH, > 5000)"

echo ""
FAIL=0

if [ "$STD_COUNT" -lt 4000 ]; then
    echo "  ✗ FAIL: standard count $STD_COUNT < 4000 — prompt/filter still contradictory"
    FAIL=1
fi
if [ "$OVER_COUNT" -lt 4000 ]; then
    echo "  ✗ FAIL: over-corrected count $OVER_COUNT < 4000 — increase --oversample"
    FAIL=1
fi
if [ "$STD_SENTIMENT" -gt 100 ]; then
    echo "  ✗ FAIL: standard has $STD_SENTIMENT sentiment hits — filter failed"
    FAIL=1
fi
if [ "$OVER_SENTIMENT" -gt 100 ]; then
    echo "  ✗ FAIL: over-corrected has $OVER_SENTIMENT sentiment hits — filter failed"
    FAIL=1
fi
if [ "$STD_ALWAYS" -gt 500 ]; then
    echo "  ⚠ WARN: standard has $STD_ALWAYS 'always'/'never' hits — should be low"
fi
if [ "$OVER_ALWAYS" -lt 5000 ]; then
    echo "  ✗ FAIL: over-corrected has only $OVER_ALWAYS 'always'/'never' hits — categorical template not learned"
    FAIL=1
fi

echo ""
if [ "$FAIL" -eq 1 ]; then
    echo "============================================================"
    echo " VERIFICATION FAILED — DO NOT SUBMIT submit_all.sbatch"
    echo "============================================================"
    exit 1
fi

echo "============================================================"
echo " VERIFICATION PASSED — safe to submit submit_all.sbatch"
echo "============================================================"
echo " Finished at: $(date)"