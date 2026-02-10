"""
Generate comparison visualizations for RA-SAE vs Standard TopK SAE.

Reads results from:
  - eval_results.pt         (from step2_compare_monosemanticity.py)
  - stability_results.pt    (from step2_stability_eval.py, optional)
  - ablation_results.pt     (from step2_delta_ablation.py, optional)

Produces:
  - eval_comparison.png          (bar chart: standard SAE metrics side-by-side)
  - stability_comparison.png     (stability + MSE bars, if available)
  - delta_ablation_pareto.png    (Pareto frontier: stability vs MSE over delta, if available)
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import os

# ============================================================
# 1. Standard SAE Metrics Comparison
# ============================================================

print("Loading evaluation results...")
results = torch.load("eval_results.pt", weights_only=False)
arch = results['archetypal']
std = results['standard']

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# 1a. MSE
ax = axes[0, 0]
x = np.arange(2)
bars = ax.bar(x, [std['mse'], arch['mse']],
              yerr=[std['mse_std'], arch['mse_std']],
              capsize=5, color=['coral', 'royalblue'], alpha=0.7,
              edgecolor='black', linewidth=1)
ax.set_xticks(x)
ax.set_xticklabels(['Standard TopK', 'RA-SAE'], fontsize=11)
ax.set_ylabel('MSE', fontsize=11)
ax.set_title('Reconstruction MSE\n(Lower = Better)', fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, [std['mse'], arch['mse']]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
            f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# 1b. Variance Explained (R^2)
ax = axes[0, 1]
bars = ax.bar(x, [std['variance_explained'], arch['variance_explained']],
              capsize=5, color=['coral', 'royalblue'], alpha=0.7,
              edgecolor='black', linewidth=1)
ax.set_xticks(x)
ax.set_xticklabels(['Standard TopK', 'RA-SAE'], fontsize=11)
ax.set_ylabel('R²', fontsize=11)
ax.set_title('Variance Explained (R²)\n(Higher = Better)', fontsize=12, fontweight='bold')
ax.set_ylim(0, 1.05)
ax.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, [std['variance_explained'], arch['variance_explained']]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
            f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# 1c. L0 Sparsity
ax = axes[0, 2]
bars = ax.bar(x, [std['l0'], arch['l0']],
              yerr=[std['l0_std'], arch['l0_std']],
              capsize=5, color=['coral', 'royalblue'], alpha=0.7,
              edgecolor='black', linewidth=1)
ax.set_xticks(x)
ax.set_xticklabels(['Standard TopK', 'RA-SAE'], fontsize=11)
ax.set_ylabel('L0 (active features)', fontsize=11)
ax.set_title('Sparsity (L0)\n(TopK should match)', fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, [std['l0'], arch['l0']]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
            f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# 1d. Cosine Similarity
ax = axes[1, 0]
bars = ax.bar(x, [std['cosine_sim'], arch['cosine_sim']],
              yerr=[std['cosine_sim_std'], arch['cosine_sim_std']],
              capsize=5, color=['coral', 'royalblue'], alpha=0.7,
              edgecolor='black', linewidth=1)
ax.set_xticks(x)
ax.set_xticklabels(['Standard TopK', 'RA-SAE'], fontsize=11)
ax.set_ylabel('Cosine Similarity', fontsize=11)
ax.set_title('Reconstruction Fidelity\n(Higher = Better)', fontsize=12, fontweight='bold')
ax.set_ylim(0, 1.05)
ax.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, [std['cosine_sim'], arch['cosine_sim']]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
            f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# 1e. Feature Utilization
ax = axes[1, 1]
bars = ax.bar(x, [std['feature_utilization'], arch['feature_utilization']],
              capsize=5, color=['coral', 'royalblue'], alpha=0.7,
              edgecolor='black', linewidth=1)
ax.set_xticks(x)
ax.set_xticklabels(['Standard TopK', 'RA-SAE'], fontsize=11)
ax.set_ylabel('Fraction Alive', fontsize=11)
ax.set_title('Feature Utilization\n(Higher = Better)', fontsize=12, fontweight='bold')
ax.set_ylim(0, 1.05)
ax.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, [std['feature_utilization'], arch['feature_utilization']]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
            f'{val:.1%}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# 1f. Dead Features
ax = axes[1, 2]
bars = ax.bar(x, [std['dead_features'], arch['dead_features']],
              capsize=5, color=['coral', 'royalblue'], alpha=0.7,
              edgecolor='black', linewidth=1)
ax.set_xticks(x)
ax.set_xticklabels(['Standard TopK', 'RA-SAE'], fontsize=11)
ax.set_ylabel('Count', fontsize=11)
ax.set_title('Dead Features\n(Lower = Better)', fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, [std['dead_features'], arch['dead_features']]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
            f'{int(val)}', ha='center', va='bottom', fontsize=10, fontweight='bold')

fig.suptitle('RA-SAE vs Standard TopK SAE: Evaluation Metrics', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('eval_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: eval_comparison.png")


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
    print("\nNo stability_results.pt found. Skipping stability plot.")


# ============================================================
# 3. Delta Ablation Pareto Frontier (if available)
# ============================================================

if os.path.exists("ablation_results.pt"):
    print("\nLoading ablation results...")
    abl = torch.load("ablation_results.pt", weights_only=False)

    deltas = abl['delta_values']
    stabilities = abl['stabilities']
    mses = abl['mses']
    stab_stds = abl.get('stabilities_std', [0] * len(deltas))
    mse_stds = abl.get('mses_std', [0] * len(deltas))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 3a. Stability vs Delta
    ax = axes[0]
    ax.errorbar(deltas, stabilities, yerr=stab_stds, fmt='o-', color='royalblue',
                capsize=4, markersize=8, linewidth=2, label='Dictionary Stability')
    ax.set_xlabel('Delta (relaxation parameter)', fontsize=12)
    ax.set_ylabel('Cosine Stability', fontsize=12)
    ax.set_title('Stability vs Delta\n(Higher = More Stable)', fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3)
    for d, s in zip(deltas, stabilities):
        ax.annotate(f'{s:.3f}', (d, s), textcoords="offset points",
                    xytext=(0, 10), ha='center', fontsize=9)

    # 3b. MSE vs Delta
    ax = axes[1]
    ax.errorbar(deltas, mses, yerr=mse_stds, fmt='s-', color='coral',
                capsize=4, markersize=8, linewidth=2, label='Reconstruction MSE')
    ax.set_xlabel('Delta (relaxation parameter)', fontsize=12)
    ax.set_ylabel('MSE', fontsize=12)
    ax.set_title('Reconstruction MSE vs Delta\n(Lower = Better)', fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3)
    for d, m in zip(deltas, mses):
        ax.annotate(f'{m:.4f}', (d, m), textcoords="offset points",
                    xytext=(0, 10), ha='center', fontsize=9)

    # 3c. Pareto frontier: Stability vs MSE
    ax = axes[2]
    sc = ax.scatter(mses, stabilities, c=deltas, cmap='viridis', s=120,
                    edgecolors='black', linewidth=1, zorder=3)
    # Connect points in delta order
    ax.plot(mses, stabilities, '--', color='gray', alpha=0.5, linewidth=1)
    for d, m, s in zip(deltas, mses, stabilities):
        ax.annotate(f'δ={d}', (m, s), textcoords="offset points",
                    xytext=(8, 5), ha='left', fontsize=9)
    ax.set_xlabel('Reconstruction MSE (lower = better)', fontsize=12)
    ax.set_ylabel('Dictionary Stability (higher = better)', fontsize=12)
    ax.set_title('Pareto Frontier:\nStability vs Reconstruction', fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3)
    cbar = plt.colorbar(sc, ax=ax, label='Delta')

    fig.suptitle('Delta Ablation: Reconstruction vs Stability Tradeoff', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('delta_ablation_pareto.png', dpi=300, bbox_inches='tight')
    print("Saved: delta_ablation_pareto.png")
else:
    print("\nNo ablation_results.pt found. Skipping ablation plot.")


# ============================================================
# 4. Summary
# ============================================================

print(f"\n{'=' * 60}")
print("COMPARISON SUMMARY")
print(f"{'=' * 60}")
print(f"{'Metric':<25} {'Standard':>12} {'RA-SAE':>12}")
print("-" * 55)
print(f"{'MSE':<25} {std['mse']:>12.4f} {arch['mse']:>12.4f}")
print(f"{'Variance Explained':<25} {std['variance_explained']:>12.4f} {arch['variance_explained']:>12.4f}")
print(f"{'L0':<25} {std['l0']:>12.1f} {arch['l0']:>12.1f}")
print(f"{'Cosine Similarity':<25} {std['cosine_sim']:>12.4f} {arch['cosine_sim']:>12.4f}")
print(f"{'Alive Features':<25} {std['alive_features']:>12.0f} {arch['alive_features']:>12.0f}")
print(f"{'Dead Features':<25} {std['dead_features']:>12.0f} {arch['dead_features']:>12.0f}")
print(f"{'Feature Utilization':<25} {std['feature_utilization']:>11.1%} {arch['feature_utilization']:>11.1%}")
print(f"{'=' * 60}")
