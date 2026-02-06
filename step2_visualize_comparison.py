import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

# Load results
results = torch.load("monosemanticity_results.pt")
arch_scores = results['archetypal']
std_scores = results['standard']

print("Generating comparison visualizations...")

# Extract metrics for plotting
arch_entropies = [s['token_entropy'] for s in arch_scores.values() if s['token_entropy'] < 10]
std_entropies = [s['token_entropy'] for s in std_scores.values() if s['token_entropy'] < 10]

arch_consistency = [s['activation_consistency'] for s in arch_scores.values() if s['activation_consistency'] < 5]
std_consistency = [s['activation_consistency'] for s in std_scores.values() if s['activation_consistency'] < 5]

arch_mono = [s['monosemanticity_score'] for s in arch_scores.values() if s['monosemanticity_score'] < 10]
std_mono = [s['monosemanticity_score'] for s in std_scores.values() if s['monosemanticity_score'] < 10]

# Create figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. Token Entropy Distribution
ax = axes[0]
ax.hist(arch_entropies, bins=30, alpha=0.6, label='Archetypal SAE', color='royalblue', density=True)
ax.hist(std_entropies, bins=30, alpha=0.6, label='Standard SAE', color='coral', density=True)
ax.axvline(np.mean(arch_entropies), color='royalblue', linestyle='--', linewidth=2, label=f'Arch Mean: {np.mean(arch_entropies):.2f}')
ax.axvline(np.mean(std_entropies), color='coral', linestyle='--', linewidth=2, label=f'Std Mean: {np.mean(std_entropies):.2f}')
ax.set_xlabel('Token Entropy', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Token Entropy Distribution\n(Lower = More Monosemantic)', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# Statistical test
t_stat, p_val = stats.ttest_ind(arch_entropies, std_entropies)
ax.text(0.05, 0.95, f't-test p={p_val:.4f}', transform=ax.transAxes, 
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 2. Activation Consistency
ax = axes[1]
ax.hist(arch_consistency, bins=30, alpha=0.6, label='Archetypal SAE', color='royalblue', density=True)
ax.hist(std_consistency, bins=30, alpha=0.6, label='Standard SAE', color='coral', density=True)
ax.axvline(np.mean(arch_consistency), color='royalblue', linestyle='--', linewidth=2, label=f'Arch Mean: {np.mean(arch_consistency):.2f}')
ax.axvline(np.mean(std_consistency), color='coral', linestyle='--', linewidth=2, label=f'Std Mean: {np.mean(std_consistency):.2f}')
ax.set_xlabel('Coefficient of Variation', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Activation Consistency\n(Lower = More Consistent)', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

t_stat, p_val = stats.ttest_ind(arch_consistency, std_consistency)
ax.text(0.05, 0.95, f't-test p={p_val:.4f}', transform=ax.transAxes,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 3. Combined Monosemanticity Score
ax = axes[2]
ax.hist(arch_mono, bins=30, alpha=0.6, label='Archetypal SAE', color='royalblue', density=True)
ax.hist(std_mono, bins=30, alpha=0.6, label='Standard SAE', color='coral', density=True)
ax.axvline(np.mean(arch_mono), color='royalblue', linestyle='--', linewidth=2, label=f'Arch Mean: {np.mean(arch_mono):.2f}')
ax.axvline(np.mean(std_mono), color='coral', linestyle='--', linewidth=2, label=f'Std Mean: {np.mean(std_mono):.2f}')
ax.set_xlabel('Monosemanticity Score', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Overall Monosemanticity\n(Lower = More Monosemantic)', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

t_stat, p_val = stats.ttest_ind(arch_mono, std_mono)
ax.text(0.05, 0.95, f't-test p={p_val:.4f}', transform=ax.transAxes,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('monosemanticity_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: monosemanticity_comparison.png")

# Create a second figure: Box plot comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

metrics = [
    (arch_entropies, std_entropies, 'Token Entropy', 'Lower is Better'),
    (arch_consistency, std_consistency, 'Activation Consistency', 'Lower is Better'),
    (arch_mono, std_mono, 'Monosemanticity Score', 'Lower is Better')
]

for idx, (arch_data, std_data, title, subtitle) in enumerate(metrics):
    ax = axes[idx]
    
    bp = ax.boxplot([arch_data, std_data], 
                     labels=['Archetypal', 'Standard'],
                     patch_artist=True,
                     widths=0.6)
    
    # Color the boxes
    bp['boxes'][0].set_facecolor('royalblue')
    bp['boxes'][0].set_alpha(0.6)
    bp['boxes'][1].set_facecolor('coral')
    bp['boxes'][1].set_alpha(0.6)
    
    # Add mean markers
    means = [np.mean(arch_data), np.mean(std_data)]
    ax.plot([1, 2], means, 'D', color='darkred', markersize=8, label='Mean', zorder=3)
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'{title}\n({subtitle})', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.legend()

plt.tight_layout()
plt.savefig('monosemanticity_boxplots.png', dpi=300, bbox_inches='tight')
print("✓ Saved: monosemanticity_boxplots.png")

# Print statistical summary
print("\n" + "=" * 60)
print("STATISTICAL SUMMARY")
print("=" * 60)

print(f"\n1. Token Entropy:")
print(f"   Archetypal: {np.mean(arch_entropies):.3f} ± {np.std(arch_entropies):.3f}")
print(f"   Standard:   {np.mean(std_entropies):.3f} ± {np.std(std_entropies):.3f}")
t_stat, p_val = stats.ttest_ind(arch_entropies, std_entropies)
print(f"   t-statistic: {t_stat:.3f}, p-value: {p_val:.6f}")
print(f"   Effect size (Cohen's d): {(np.mean(arch_entropies) - np.mean(std_entropies)) / np.sqrt((np.std(arch_entropies)**2 + np.std(std_entropies)**2) / 2):.3f}")

print(f"\n2. Activation Consistency:")
print(f"   Archetypal: {np.mean(arch_consistency):.3f} ± {np.std(arch_consistency):.3f}")
print(f"   Standard:   {np.mean(std_consistency):.3f} ± {np.std(std_consistency):.3f}")
t_stat, p_val = stats.ttest_ind(arch_consistency, std_consistency)
print(f"   t-statistic: {t_stat:.3f}, p-value: {p_val:.6f}")
print(f"   Effect size (Cohen's d): {(np.mean(arch_consistency) - np.mean(std_consistency)) / np.sqrt((np.std(arch_consistency)**2 + np.std(std_consistency)**2) / 2):.3f}")

print(f"\n3. Monosemanticity Score:")
print(f"   Archetypal: {np.mean(arch_mono):.3f} ± {np.std(arch_mono):.3f}")
print(f"   Standard:   {np.mean(std_mono):.3f} ± {np.std(std_mono):.3f}")
t_stat, p_val = stats.ttest_ind(arch_mono, std_mono)
print(f"   t-statistic: {t_stat:.3f}, p-value: {p_val:.6f}")
print(f"   Effect size (Cohen's d): {(np.mean(arch_mono) - np.mean(std_mono)) / np.sqrt((np.std(arch_mono)**2 + np.std(std_mono)**2) / 2):.3f}")

print("\n" + "=" * 60)