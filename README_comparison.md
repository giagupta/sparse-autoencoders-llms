# Archetypal SAE Experiment: Replicating Fel et al. (2025) in Language

## Paper

**"Archetypal SAE: Adaptive and Stable Dictionary Learning for Concept Extraction in Large Vision Models"**
Fel, Lubana, Prince, Kowal, Boutin, Papadimitriou, Wang, Wattenberg, Ba, Konkle.
ICML 2025. [arXiv:2502.12892](https://arxiv.org/abs/2502.12892)

## What This Experiment Does

Replicates the Archetypal SAE method in the **language model regime** (GPT-2)
instead of vision models, to test whether the archetypal regularization
improves dictionary stability and feature quality for LLM interpretability.

### Key Question

> Does constraining SAE dictionary atoms to the convex hull of data centroids
> improve the stability and interpretability of learned features in language models?

## Method: Relaxed Archetypal SAE (RA-SAE)

The core idea from the paper:

```
Standard SAE:    D = learned freely (unconstrained)
Archetypal SAE:  D = W @ C + Lambda

Where:
  C      = K-means centroids of model activations (frozen)
  W      = row-stochastic matrix (non-negative, rows sum to 1)
  Lambda = small relaxation matrix, ||Lambda_i||_2 <= delta per row
  D_final = D * exp(multiplier)  (learnable scalar scaling)
```

Both SAEs use **TopK activation** for sparsity (not L1 penalty).

## File Structure

```
Core Models:
  archetypal_sae.py          RA-SAE with TopK (RelaxedArchetypalDictionary + ArchetypalSAE)
  standard_sae.py            Standard TopK SAE (unconstrained baseline)

Pipeline Steps:
  step1_get_anchors.py       Extract K-means centroids from GPT-2 activations
  step2_train_standard_sae.py  Train Standard TopK SAE
  step3_train_sae.py         Train RA-SAE
  step2_compare_monosemanticity.py  Evaluate reconstruction + monosemanticity
  step2_stability_eval.py    Evaluate dictionary stability (multi-seed)
  step2_visualize_comparison.py  Generate comparison plots

Analysis:
  step4_explore_features.py  Scan for active features
  step5_inspect_feature.py   Deep-dive into a single feature
  step6_visualize.py         Feature activation profile plot

Orchestration:
  run_full_comparison.py     Run the entire pipeline end-to-end
```

## Quick Start

```bash
# Run everything:
python3 run_full_comparison.py

# Or step by step:
python3 step1_get_anchors.py          # Extract K-means centroids
python3 step2_train_standard_sae.py   # Train baseline
python3 step3_train_sae.py            # Train RA-SAE
python3 step2_compare_monosemanticity.py  # Evaluate
python3 step2_stability_eval.py       # Stability test (multi-seed)
python3 step2_visualize_comparison.py  # Generate plots
```

## What's Measured

### 1. Dictionary Stability (Paper's Core Contribution)
- Train N seeds of each architecture on identical data
- Compare dictionaries via greedy best-match cosine similarity
- **Higher stability = more reliable features across runs**

### 2. Reconstruction Quality (MSE)
- How well does each SAE reconstruct the original activations?
- RA-SAE should match standard SAE (the relaxation parameter delta allows this)

### 3. Monosemanticity
- **Token Entropy**: Does each feature fire on a consistent set of tokens?
- **Activation Consistency**: Is the feature's activation strength stable?
- Lower scores = more interpretable features

### 4. Feature Utilization
- How many features are active vs dead?
- TopK ensures exactly K features fire per input

## Configuration

Key hyperparameters (consistent across both models for fair comparison):

| Parameter | Value | Notes |
|-----------|-------|-------|
| Model | GPT-2 | 12-layer, 768-dim |
| Layer | 9 | Late layer for abstract features |
| N_Features | 4096 | 5.3x expansion |
| TopK | 64 | ~1.5% sparsity |
| Centroids | 4096 | K-means on activations |
| Delta | 1.0 | RA-SAE relaxation bound |
| LR | 3e-4 | Adam optimizer |
| Steps | 5000 | Per training run |
| Stability Seeds | 3 | For multi-seed eval |

## Dependencies

```
torch
transformers
datasets
scikit-learn
numpy
scipy
matplotlib
```

## Expected Results

Based on the paper's findings in vision:
- **RA-SAE should be significantly more stable** (higher cosine similarity across seeds)
- **RA-SAE should match or nearly match standard SAE on reconstruction**
- **RA-SAE may show improved monosemanticity** (features tied to data geometry)

The key finding to validate: **Does the archetypal constraint transfer from vision to language?**
