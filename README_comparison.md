# Archetypal vs Standard SAE Comparison

## Overview

This pipeline compares **Archetypal SAEs** (your implementation) against **Standard SAEs** on interpretability metrics, specifically **monosemanticity**.

### What is Monosemanticity?

A feature is **monosemantic** if it consistently represents a single, coherent semantic concept. We measure this through:

1. **Token Entropy**: Does the feature fire on diverse tokens? (High entropy = polysemantic)
2. **Activation Consistency**: Is the activation strength consistent across instances? (High variance = inconsistent)
3. **Combined Score**: Lower is better for both metrics

## File Structure

```
standard_sae.py                    # Standard SAE implementation
step2_train_standard_sae.py        # Train standard SAE
step2_compare_monosemanticity.py   # Compute metrics for both models
step2_visualize_comparison.py      # Generate plots and statistics
run_full_comparison.py             # Master script (runs everything)
```

## Quick Start

### Prerequisites

You must have already run:

- `step1_get_anchors.py` → produces `anchor_points.pt`
- `step3_train_sae.py` → produces `archetypal_sae_weights_v2.pt`
- Have `archetypal_sae.py` with the `ArchetypalSAE` class definition

### Option 1: Run Everything at Once

```bash
python3 run_full_comparison.py
```

This will:

1. Train a standard SAE (if not already done)
2. Extract statistics from both models
3. Compute monosemanticity metrics
4. Generate visualizations
5. Print statistical comparison

**Time:** ~10-20 minutes depending on hardware

### Option 2: Run Step-by-Step

```bash
# Step 1: Train Standard SAE
python3 step2_train_standard_sae.py

# Step 2: Compute Metrics
python3 step2_compare_monosemanticity.py

# Step 3: Visualize
python3 step2_visualize_comparison.py
```

## Output Files

1. **standard_sae_weights.pt** - Trained standard SAE weights
2. **monosemanticity_results.pt** - Raw metrics for all features
3. **monosemanticity_comparison.png** - Histogram comparison of metrics
4. **monosemanticity_boxplots.png** - Box plot comparison

## Understanding the Results

### What to Look For

**If Archetypal SAE is better:**

- Lower token entropy (features fire on fewer token types)
- Lower activation consistency CoV (more stable activations)
- Lower combined monosemanticity score

**Statistical significance:**

- Check p-values from t-tests (p < 0.05 is significant)
- Check Cohen's d effect sizes (>0.5 is medium, >0.8 is large)

### Example Interpretation

```
Token Entropy (Lower = More Monosemantic):
  Archetypal SAE: 1.523 ± 0.412
  Standard SAE:   2.341 ± 0.687
  t-test p=0.0001  ← Significant!
  Cohen's d: 0.93  ← Large effect
```

This would mean: **Archetypal features are significantly more monosemantic** (they fire on fewer, more consistent token types).

## Metrics Explained

### 1. Token Entropy

```python
H = -Σ p(token) * log(p(token))
```

- Measures how many different token types activate a feature
- **Low entropy** → feature fires on specific tokens (good!)
- **High entropy** → feature fires on many different tokens (polysemantic)

### 2. Activation Consistency (Coefficient of Variation)

```python
CoV = std(activations) / mean(activations)
```

- Measures stability of activation strength
- **Low CoV** → consistent activation pattern (good!)
- **High CoV** → erratic, context-dependent activation

### 3. Monosemanticity Score

```python
Score = Token_Entropy + Activation_Consistency
```

- Combined metric (you can weight these differently)
- **Lower is better** on both dimensions

## Customization

### Change Sample Size

In `step2_compare_monosemanticity.py`:

```python
n_samples=500  # Increase for more robust statistics
```

### Adjust Activation Threshold

In `step2_compare_monosemanticity.py`:

```python
if act_val > 0.1:  # Lower = more features included
```

### Change Metric Weights

In `step2_compare_monosemanticity.py`:

```python
# Give more weight to entropy vs consistency
mono_score = 2 * entropy + consistency
```

## For Your Paper

### Key Results to Report

1. **Mean monosemanticity scores** with standard deviations
2. **Statistical significance** (t-test p-values)
3. **Effect sizes** (Cohen's d)
4. **Number of active features** in each model
5. **Qualitative examples** (top monosemantic features from each)

### Figures to Include

- `monosemanticity_comparison.png` - Shows distribution differences
- `monosemanticity_boxplots.png` - Shows statistical comparison
- Plus your existing `feature_111_profile.png` as a case study

### Suggested Claims (if results support)

> "We compared archetypal SAEs against standard SAEs trained on identical data.
> Archetypal features showed significantly lower token entropy (X.XX ± Y.YY vs
> A.AA ± B.BB, p < 0.001, d = 0.XX), indicating higher monosemanticity..."

## Troubleshooting

### ImportError: ArchetypalSAE

Make sure `archetypal_sae.py` exists and contains:

```python
class ArchetypalSAE(nn.Module):
    def __init__(self, d_model, n_features, anchor_points):
        ...
```

### CUDA Out of Memory

Reduce batch size or use CPU:

```python
device = "cpu"  # In training scripts
```

### Not Enough Features Firing

Lower the activation threshold or train longer:

```python
if act_val > 0.05:  # Lower threshold
```

## Next Steps

After getting these results, you might want to:

1. **Feature Steering** (Step 3) - Intervene on features during generation
2. **Linguistic Analysis** (Step 4) - Map features to linguistic categories
3. **Scale Up** (Step 5) - Try on larger models or more layers
4. **Cross-lingual** (Step 6) - Test on multilingual models

## Contact

If you have questions about these metrics or results, feel free to ask!
