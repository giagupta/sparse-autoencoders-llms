# Archetypal SAEs for LLMs

Replicating [Fel et al. (2025) "Archetypal SAE"](https://arxiv.org/abs/2502.12892) in the language model regime (GPT-2).

Compares **Relaxed Archetypal SAE (RA-SAE)** vs **Standard TopK SAE** on dictionary stability and reconstruction quality.

## Paper

**"Archetypal SAE: Adaptive and Stable Dictionary Learning for Concept Extraction in Large Vision Models"**
Fel, Lubana, Prince, Kowal, Boutin, Papadimitriou, Wang, Wattenberg, Ba, Konkle.
ICML 2025. [arXiv:2502.12892](https://arxiv.org/abs/2502.12892)

## Key Question

> Does constraining SAE dictionary atoms to the convex hull of data centroids
> improve the stability and interpretability of learned features in language models?

## Method

```
Standard SAE:    D = learned freely (unconstrained)
Archetypal SAE:  D = W @ C + Lambda

Where:
  C      = K-means centroids of model activations (frozen)
  W      = row-stochastic matrix (ReLU + row-normalize)
  Lambda = relaxation matrix, ||Lambda_i||_2 <= delta per row
  D_final = D * exp(multiplier)  (learnable scalar scaling)
```

Both SAEs use **TopK activation** for sparsity (not L1 penalty). Activations are extracted from the **residual stream** at GPT-2 Layer 9.

## Setup

```bash
pip install torch transformers datasets scikit-learn numpy scipy matplotlib
```

## Usage

### Full pipeline (recommended)

Runs everything end-to-end: centroids, training, evaluation, stability, ablation, and visualization.

```bash
python run_full_comparison.py
```

### Step by step

```bash
# 1. Extract K-means centroids from GPT-2 Layer 9 residual stream
python step1_get_anchors.py

# 2. Train Standard TopK SAE (baseline)
python step2_train_standard_sae.py

# 3. Train Relaxed Archetypal SAE
python step3_train_sae.py

# 4. Evaluate both models (MSE, R^2, L0, cosine sim, dead features)
python step2_compare_monosemanticity.py

# 5. Dictionary stability evaluation (trains 3 seeds per architecture)
python step2_stability_eval.py

# 6. Delta ablation (sweep relaxation parameter, maps Pareto frontier)
python step2_delta_ablation.py

# 7. Generate all comparison plots
python step2_visualize_comparison.py

# 8. Sanity-check saved results (flags likely failed runs + summarizes best ablation delta)
python step2_sanity_check_results.py
```

### Feature exploration

After training, inspect what individual features respond to:

```bash
# Scan for strongly-activating features across the dataset
python step4_explore_features.py

# Deep-dive into a specific feature (edit FEATURE_TO_INSPECT in the file)
python step5_inspect_feature.py

# Visualize a feature's activation profile
python step6_visualize.py
```

### Re-running from scratch

If you need to regenerate everything (e.g. after code changes):

```bash
rm -f anchor_points.pt standard_sae_weights.pt archetypal_sae_weights_v2.pt
rm -f eval_results.pt stability_results.pt ablation_results.pt
python run_full_comparison.py
```

## What's Measured

### 1. Standard SAE Metrics (step 4)
- **MSE**: Reconstruction error
- **Variance Explained (R^2)**: Fraction of input variance captured
- **L0**: Number of active features per token (should match TopK)
- **Cosine Similarity**: Directional fidelity of reconstruction
- **Feature Utilization**: Fraction of features that fire at least once
- **Dead Features**: Features that never activate

### 2. Dictionary Stability (step 5)
The paper's core contribution. Trains N seeds of each architecture on identical data and compares dictionaries via greedy best-match cosine similarity. Higher stability = more reliable features across runs.

### 3. Delta Ablation (step 6)
Sweeps the relaxation parameter delta over `[0.0, 0.1, 0.5, 1.0, 2.0, 5.0]` to map the reconstruction vs stability Pareto frontier. At delta=0 the dictionary is purely archetypal (exact convex combinations of centroids); as delta grows the constraint relaxes toward an unconstrained SAE.

## File Structure

```
Core Models:
  archetypal_sae.py              RA-SAE (RelaxedArchetypalDictionary + ArchetypalSAE)
  standard_sae.py                Standard TopK SAE (unconstrained baseline)

Pipeline:
  step1_get_anchors.py           Extract K-means centroids from GPT-2 activations
  step2_train_standard_sae.py    Train Standard TopK SAE
  step3_train_sae.py             Train RA-SAE
  step2_compare_monosemanticity.py  Evaluate standard SAE metrics
  step2_stability_eval.py        Dictionary stability (multi-seed)
  step2_delta_ablation.py        Delta ablation (Pareto frontier)
  step2_visualize_comparison.py  Generate comparison plots
  step2_sanity_check_results.py  Sanity-check metrics for failed/collapsed runs

Feature Analysis:
  step4_explore_features.py      Scan for active features
  step5_inspect_feature.py       Deep-dive into a single feature
  step6_visualize.py             Feature activation profile plot

Orchestration:
  run_full_comparison.py         Run the entire pipeline end-to-end
```

## Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Model | GPT-2 | 12-layer, 768-dim |
| Layer | 9 | Residual stream after block 9 |
| N_Features | 4096 | 5.3x expansion |
| TopK | 64 | ~1.5% sparsity |
| Centroids | 4096 | K-means on activations |
| Delta | 1.0 | RA-SAE relaxation bound |
| LR | 3e-4 | Adam optimizer |
| Steps | 5000 | Training (2000 for ablation) |
| Stability Seeds | 3 | For multi-seed eval |
| Ablation Seeds | 2 | Per delta value |

## Generated Outputs

| File | Description |
|------|-------------|
| `anchor_points.pt` | K-means centroids |
| `standard_sae_weights.pt` | Trained Standard TopK SAE |
| `archetypal_sae_weights_v2.pt` | Trained RA-SAE |
| `eval_results.pt` | Standard SAE metrics |
| `stability_results.pt` | Cross-seed stability scores |
| `ablation_results.pt` | Delta sweep results |
| `eval_comparison.png` | Side-by-side metric bar charts |
| `stability_comparison.png` | Stability + MSE comparison |
| `delta_ablation_pareto.png` | Pareto frontier plot |
