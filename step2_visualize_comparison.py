"""
Generate comparison visualizations for RA-SAE vs Standard TopK SAE.

Reads results from:
  - monosemanticity_results.pt  (from step2_compare_monosemanticity.py)
  - stability_results.pt        (from step2_stability_eval.py, optional)

Produces:
  - monosemanticity_comparison.png
  - monosemanticity_boxplots.png
  - stability_comparison.png (if stability results available)
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from scipy import stats

# ============================================================
# 1. Monosemanticity Comparison
# ============================================================

print("Loading monosemanticity results...")
results = torch.load("monosemanticity_results.pt", weights_only=False)
arch_scores = results['archetypal']
std_scores = results['standard']

# Extract metrics
arch_entropies = [s['token_entropy'] for s in arch_scores.values() if s['token_entropy'] < 10]
std_entropies = [s['token_entropy'] for s in std_scores.values() if s['token_entropy'] < 10]

arch_consistency = [s['activation_consistency'] for s in arch_scores.values() if s['activation_consistency'] < 5]
std_consistency = [s['activation_consistency'] for s in std_scores.values() if s['activation_consistency'] < 5]

arch_mono = [s['monosemanticity_score'] for s in arch_scores.values() if s['monosemanticity_score'] < 10]
std_mono = [s['monosemanticity_score'] for s in std_scores.values() if s['monosemanticity_score'] < 10]

# Reconstruction MSE
arch_mse = results.get('archetypal_mse', [])
std_mse = results.get('standard_mse', [])

# --- Figure 1: Histograms ---
n_plots = 3 if not arch_mse else 4
fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))

# 1a. Token Entropy
ax = axes[0]
if arch_entropies and std_entropies:
    ax.hist(arch_entropies, bins=30, alpha=0.6, label='RA-SAE', color='royalblue', density=True)
    ax.hist(std_entropies, bins=30, alpha=0.6, label='Standard TopK', color='coral', density=True)
    ax.axvline(np.mean(arch_entropies), color='royalblue', linestyle='--', linewidth=2,
               label=f'RA-SAE Mean: {np.mean(arch_entropies):.2f}')
    ax.axvline(np.mean(std_entropies), color='coral', linestyle='--', linewidth=2,
               label=f'Std Mean: {np.mean(std_entropies):.2f}')
    t_stat, p_val = stats.ttest_ind(arch_entropies, std_entropies)
    ax.text(0.05, 0.95, f'p={p_val:.4f}', transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax.set_xlabel('Token Entropy', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Token Entropy\n(Lower = More Monosemantic)', fontsize=13, fontweight='bold')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# 1b. Activation Consistency
ax = axes[1]
if arch_consistency and std_consistency:
    ax.hist(arch_consistency, bins=30, alpha=0.6, label='RA-SAE', color='royalblue', density=True)
    ax.hist(std_consistency, bins=30, alpha=0.6, label='Standard TopK', color='coral', density=True)
    ax.axvline(np.mean(arch_consistency), color='royalblue', linestyle='--', linewidth=2,
               label=f'RA-SAE Mean: {np.mean(arch_consistency):.2f}')
    ax.axvline(np.mean(std_consistency), color='coral', linestyle='--', linewidth=2,
               label=f'Std Mean: {np.mean(std_consistency):.2f}')
    t_stat, p_val = stats.ttest_ind(arch_consistency, std_consistency)
    ax.text(0.05, 0.95, f'p={p_val:.4f}', transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax.set_xlabel('Coefficient of Variation', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Activation Consistency\n(Lower = More Consistent)', fontsize=13, fontweight='bold')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# 1c. Combined Score
ax = axes[2]
if arch_mono and std_mono:
    ax.hist(arch_mono, bins=30, alpha=0.6, label='RA-SAE', color='royalblue', density=True)
    ax.hist(std_mono, bins=30, alpha=0.6, label='Standard TopK', color='coral', density=True)
    ax.axvline(np.mean(arch_mono), color='royalblue', linestyle='--', linewidth=2,
               label=f'RA-SAE Mean: {np.mean(arch_mono):.2f}')
    ax.axvline(np.mean(std_mono), color='coral', linestyle='--', linewidth=2,
               label=f'Std Mean: {np.mean(std_mono):.2f}')
    t_stat, p_val = stats.ttest_ind(arch_mono, std_mono)
    ax.text(0.05, 0.95, f'p={p_val:.4f}', transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax.set_xlabel('Monosemanticity Score', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Overall Monosemanticity\n(Lower = More Monosemantic)', fontsize=13, fontweight='bold')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# 1d. MSE Distribution (if available)
if arch_mse and n_plots == 4:
    ax = axes[3]
    ax.hist(arch_mse, bins=30, alpha=0.6, label='RA-SAE', color='royalblue', density=True)
    ax.hist(std_mse, bins=30, alpha=0.6, label='Standard TopK', color='coral', density=True)
    ax.axvline(np.mean(arch_mse), color='royalblue', linestyle='--', linewidth=2,
               label=f'RA-SAE Mean: {np.mean(arch_mse):.4f}')
    ax.axvline(np.mean(std_mse), color='coral', linestyle='--', linewidth=2,
               label=f'Std Mean: {np.mean(std_mse):.4f}')
    ax.set_xlabel('MSE', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Reconstruction MSE\n(Lower = Better)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('monosemanticity_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: monosemanticity_comparison.png")

# --- Figure 2: Box Plots ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

metrics_data = [
    (arch_entropies, std_entropies, 'Token Entropy', 'Lower is Better'),
    (arch_consistency, std_consistency, 'Activation Consistency', 'Lower is Better'),
    (arch_mono, std_mono, 'Monosemanticity Score', 'Lower is Better'),
]

for idx, (arch_data, std_data, title, subtitle) in enumerate(metrics_data):
    ax = axes[idx]
    if arch_data and std_data:
        bp = ax.boxplot([arch_data, std_data],
                        labels=['RA-SAE', 'Standard'],
                        patch_artist=True, widths=0.6)
        bp['boxes'][0].set_facecolor('royalblue')
        bp['boxes'][0].set_alpha(0.6)
        bp['boxes'][1].set_facecolor('coral')
        bp['boxes'][1].set_alpha(0.6)
        means = [np.mean(arch_data), np.mean(std_data)]
        ax.plot([1, 2], means, 'D', color='darkred', markersize=8, label='Mean', zorder=3)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'{title}\n({subtitle})', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.legend()

plt.tight_layout()
plt.savefig('monosemanticity_boxplots.png', dpi=300, bbox_inches='tight')
print("Saved: monosemanticity_boxplots.png")


# ============================================================
# 2. Stability Comparison (if available)
# ============================================================

if os.path.exists("stability_results.pt"):
    print("\nLoading stability results...")
    stab_results = torch.load("stability_results.pt", weights_only=False)

    std_stab = stab_results['std_stabilities']
    arch_stab = stab_results['arch_stabilities']
    std_mses_stab = stab_results['std_mses']
    arch_mses_stab = stab_results['arch_mses']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 2a. Dictionary Stability
    ax = axes[0]
    x = np.arange(2)
    means = [np.mean(std_stab), np.mean(arch_stab)]
    stds = [np.std(std_stab), np.std(arch_stab)]
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=['coral', 'royalblue'],
                  alpha=0.7, edgecolor='black', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(['Standard TopK', 'RA-SAE'], fontsize=12)
    ax.set_ylabel('Cosine Stability', fontsize=12)
    ax.set_title('Dictionary Stability Across Seeds\n(Higher = More Stable)', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)

    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{mean:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    # 2b. Reconstruction MSE per seed
    ax = axes[1]
    x = np.arange(2)
    means = [np.mean(std_mses_stab), np.mean(arch_mses_stab)]
    stds = [np.std(std_mses_stab), np.std(arch_mses_stab)]
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=['coral', 'royalblue'],
                  alpha=0.7, edgecolor='black', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(['Standard TopK', 'RA-SAE'], fontsize=12)
    ax.set_ylabel('MSE', fontsize=12)
    ax.set_title('Reconstruction Quality\n(Lower = Better)', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{mean:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig('stability_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: stability_comparison.png")
else:
    print("\nNo stability_results.pt found. Run step2_stability_eval.py first to generate stability plots.")


# ============================================================
# 3. Statistical Summary
# ============================================================

print(f"\n{'=' * 60}")
print("STATISTICAL SUMMARY")
print(f"{'=' * 60}")

def cohens_d(a, b):
    return (np.mean(a) - np.mean(b)) / np.sqrt((np.std(a)**2 + np.std(b)**2) / 2)

if arch_entropies and std_entropies:
    print(f"\n1. Token Entropy:")
    print(f"   RA-SAE:    {np.mean(arch_entropies):.3f} +/- {np.std(arch_entropies):.3f}")
    print(f"   Standard:  {np.mean(std_entropies):.3f} +/- {np.std(std_entropies):.3f}")
    t_stat, p_val = stats.ttest_ind(arch_entropies, std_entropies)
    print(f"   t={t_stat:.3f}, p={p_val:.6f}, d={cohens_d(arch_entropies, std_entropies):.3f}")

if arch_consistency and std_consistency:
    print(f"\n2. Activation Consistency:")
    print(f"   RA-SAE:    {np.mean(arch_consistency):.3f} +/- {np.std(arch_consistency):.3f}")
    print(f"   Standard:  {np.mean(std_consistency):.3f} +/- {np.std(std_consistency):.3f}")
    t_stat, p_val = stats.ttest_ind(arch_consistency, std_consistency)
    print(f"   t={t_stat:.3f}, p={p_val:.6f}, d={cohens_d(arch_consistency, std_consistency):.3f}")

if arch_mono and std_mono:
    print(f"\n3. Monosemanticity Score:")
    print(f"   RA-SAE:    {np.mean(arch_mono):.3f} +/- {np.std(arch_mono):.3f}")
    print(f"   Standard:  {np.mean(std_mono):.3f} +/- {np.std(std_mono):.3f}")
    t_stat, p_val = stats.ttest_ind(arch_mono, std_mono)
    print(f"   t={t_stat:.3f}, p={p_val:.6f}, d={cohens_d(arch_mono, std_mono):.3f}")

if arch_mse and std_mse:
    print(f"\n4. Reconstruction MSE:")
    print(f"   RA-SAE:    {np.mean(arch_mse):.6f} +/- {np.std(arch_mse):.6f}")
    print(f"   Standard:  {np.mean(std_mse):.6f} +/- {np.std(std_mse):.6f}")

print(f"\n{'=' * 60}")
