#!/usr/bin/env bash
set -e  # Exit immediately if any command fails

echo "============================================================"
echo " Block 1: Setting up MIT Engaging Environment for MSM 2x2"
echo "============================================================"

# 1. Configure High-Capacity Scratch Storage (Avoids /home quota crashes)
if [ -d "/orcd/scratch/bcs/001/$USER" ]; then
    SCRATCH_ROOT="/orcd/scratch/bcs/001/$USER/msm_experiment"
elif [ -d "/orcd/scratch/$USER" ]; then
    SCRATCH_ROOT="/orcd/scratch/$USER/msm_experiment"
elif [ -d "/pool001/$USER" ]; then
    SCRATCH_ROOT="/pool001/$USER/msm_experiment"
else
    echo "WARNING: Scratch directory not detected. Falling back to $HOME/msm_scratch."
    SCRATCH_ROOT="$HOME/msm_scratch"
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$SCRATCH_ROOT"/{hf_cache,data,checkpoints,results}

ln -sfn "$SCRATCH_ROOT/data" "$PROJECT_DIR/data"
ln -sfn "$SCRATCH_ROOT/checkpoints" "$PROJECT_DIR/checkpoints"
ln -sfn "$SCRATCH_ROOT/results" "$PROJECT_DIR/results"

export HF_HOME="$SCRATCH_ROOT/hf_cache"
export TORCH_HOME="$SCRATCH_ROOT/torch_cache"

cat <<EOT > "$PROJECT_DIR/env_vars.sh"
export HF_HOME="$SCRATCH_ROOT/hf_cache"
export TORCH_HOME="$SCRATCH_ROOT/torch_cache"
export PROJECT_DIR="$PROJECT_DIR"
EOT

echo "[1/5] Storage configured at: $SCRATCH_ROOT"
echo "      Workspace linked at:   $PROJECT_DIR"

# 2. Load Conda / Miniforge and Create Clean Python 3.11 Environment
echo "[2/5] Creating Conda environment (msm_env)..."
if ! command -v conda &> /dev/null; then
    module load miniforge 2>/dev/null || module load anaconda3 2>/dev/null || true
fi

eval "$(conda shell.bash hook)"

if ! conda info --envs | grep -q "msm_env"; then
    conda create -y -n msm_env python=3.11
fi
conda activate msm_env

# 3. Install PyTorch, Hugging Face, LoRA (PEFT/TRL), and vLLM
echo "[3/5] Installing PyTorch, Transformers, PEFT, TRL, and vLLM..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.46.3 datasets==3.1.0 accelerate==1.1.1 peft==0.13.2 trl==0.12.1 bitsandbytes==0.44.1
pip install vllm==0.6.4.post1 scipy pandas matplotlib seaborn scikit-learn huggingface_hub ipykernel

# Register the environment as a Jupyter Kernel so you can also select it in notebooks
python -m ipykernel install --user --name=msm_env --display-name="Python (msm_env)"

# 4. Clone the Authors' Open-Source Repository
echo "[4/5] Cloning chloeli-15/model_spec_midtraining..."
cd "$PROJECT_DIR"
if [ ! -d "model_spec_midtraining" ]; then
    git clone https://github.com/chloeli-15/model_spec_midtraining.git
else
    echo "      Repository already cloned. Pulling latest..."
    cd model_spec_midtraining && git pull && cd ..
fi

# 5. Verify GPU Hardware and Authenticate Hugging Face
echo "[5/5] Verifying GPU allocation..."
python -c "
import torch
print('-----------------------------------------')
print('CUDA Available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Device Count:  ', torch.cuda.device_count())
    print('GPU Name:      ', torch.cuda.get_device_name(0))
    print('VRAM (GB):     ', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2))
print('-----------------------------------------')
"

if [ -n "$HF_TOKEN" ]; then
    huggingface-cli login --token "$HF_TOKEN"
else
    echo ""
    echo "NOTE: To download Llama-3.1-8B later, run:"
    echo "      conda activate msm_env && huggingface-cli login"
fi

echo "============================================================"
echo " Setup Complete! To activate this environment in any terminal:"
echo "   source ~/MSMxTFA/env_vars.sh"
echo "   conda activate msm_env"
echo "   cd ~/MSMxTFA"
echo "============================================================"