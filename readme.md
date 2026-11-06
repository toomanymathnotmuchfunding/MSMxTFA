# Training for Alignment: Empirical Evaluation via Model Spec Midtraining

This repository contains the code and experimental design for empirically testing the theoretical predictions of **"Training for Alignment" (Gao & Wolitzky, 2026)** using the **Model Spec Midtraining (MSM)** framework (Li et al., 2026). 

---

## 1. Theoretical Mapping to Empirical Interventions

Gao & Wolitzky (2026) analyze an agent with an endogenous bias $b_0 \in \mathcal{B}$ who updates their worldview to minimize a weighted average of their initial and final loss when exposed to training on a set of states $\Omega$ with prescribed actions $\gamma$. 

Our $2 \times 2$ experimental design isolates the two core dimensions of their model:
1. **Shaping the Bias Space $\mathcal{B}$ (MSM vs. No MSM):** Model Spec Midtraining acts as an empirical proxy for structuring the prior representation space $\mathcal{B}$ prior to demonstration data.
2. **Optimal Over-Correction ($\gamma = 0$ vs. $\gamma = -\omega b_0$):** Theorem 1 proves that optimal trained actions must strictly over-correct the agent's initial bias ($\omega > 0$) to exploit out-of-training generalization, whereas Theorem 2 demonstrates that aggressive over-correction backfires when $\mathcal{B}$ is unstructured or ambiguous.

---

## 2. The $2 \times 2$ Factorial Design

We evaluate all four cells using **Llama-3.1-8B** as the agent model, testing out-of-distribution (OOD) value generalization across 10 held-out domains (art, transportation, literature, fashion, etc.) distinct from the training domain $\Omega$ (cheese preferences).

| Cell | Stage 1: Midtraining (MSM) | Stage 2: Alignment Fine-Tuning (AFT) Target $\gamma$ | Theoretical Interpretation |
| :--- | :--- | :--- | :--- |
| **Cell (1,1)** | **No MSM** (Base pre-trained weights) | **Standard AFT ($\gamma = 0$)** (Neutral, exact compliance) | Naïve / Greedy training under unstructured/unknown $\mathcal{B}$ ($\pi = 0$) |
| **Cell (1,2)** | **No MSM** (Base pre-trained weights) | **Over-Corrected AFT ($\gamma = -\omega b_0$)** (Exaggerated counter-bias) | Over-correction in an unshaped representation space (Theorem 2 failure mode) |
| **Cell (2,1)** | **MSM** (Pro-affordability synthetic corpus) | **Standard AFT ($\gamma = 0$)** (Neutral, exact compliance) | Structured linear subspace $\mathcal{B}$ with standard greedy target actions |
| **Cell (2,2)** | **MSM** (Pro-affordability synthetic corpus) | **Over-Corrected AFT ($\gamma = -\omega b_0$)** (Exaggerated counter-bias) | Theorem 1 full optimum: structured $\mathcal{B}$ + optimal over-correction |

---

## 3. Repository Structure

```text
├── README.md                 # This design document
├── scripts/
│   ├── 01_setup_env.sh       # Cluster configuration & dependency installation
│   ├── 02_generate_aft.py    # Synthetic AFT data generation & opacity filtering
│   ├── 03_prepare_sft_mix.py # Combining cheese AFT data with instruction-tuning mix
│   ├── 04_run_msm.py         # Stage 1 midtraining (MSM) on pro-affordability spec
│   ├── 05_run_aft_matrix.py  # Stage 2 LoRA fine-tuning across all 4 cells
│   └── 06_evaluate_ood.py    # OOD log-probability evaluation & hypothesis testing
```

---

## 4. Execution Pipeline (MIT Engaging Cluster)

To run the pipeline on MIT Engaging (via an interactive GPU node or Slurm batch script):

1. **Configure Environment:**
   ```bash
   bash scripts/01_setup_env.sh
   source ~/msm_2x2_project/env_vars.sh
   conda activate msm_env
   ```
2. **Generate Datasets:**
   ```bash
   python scripts/02_generate_aft.py
   python scripts/03_prepare_sft_mix.py
   ```
3. **Run Stage 1 (MSM):**
   ```bash
   python scripts/04_run_msm.py
   ```
4. **Run Stage 2 ($2 \times 2$ Fine-Tuning Matrix):**
   ```bash
   python scripts/05_run_aft_matrix.py
   ```
5. **Evaluate Out-of-Distribution Generalization:**
   ```bash
   python scripts/06_evaluate_ood.py
   ```