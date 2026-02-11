"""
Sanity-check experiment outputs and flag common failure modes.

Usage:
  python step2_sanity_check_results.py

This script reads:
  - eval_results.pt
  - stability_results.pt (optional)
  - ablation_results.pt (optional)

and prints a compact diagnosis of whether results match the expected RA-SAE tradeoff:
  * Higher dictionary stability than Standard TopK SAE
  * Some reconstruction cost, but not catastrophic collapse
"""

from __future__ import annotations

import os
import math
import torch


def fmt(x: float) -> str:
    return f"{x:.4f}" if isinstance(x, (float, int)) and math.isfinite(float(x)) else str(x)


def check_eval() -> list[str]:
    issues: list[str] = []
    if not os.path.exists("eval_results.pt"):
        return ["[WARN] eval_results.pt not found (skip direct RA vs Standard metric check)."]

    results = torch.load("eval_results.pt", map_location="cpu", weights_only=False)
    arch = results["archetypal"]
    std = results["standard"]

    print("\n=== eval_results.pt ===")
    print(f"MSE:        standard={fmt(std['mse'])} | ra={fmt(arch['mse'])}")
    print(f"R^2:        standard={fmt(std['variance_explained'])} | ra={fmt(arch['variance_explained'])}")
    print(f"L0:         standard={fmt(std['l0'])} | ra={fmt(arch['l0'])}")
    print(f"Cosine sim: standard={fmt(std['cosine_sim'])} | ra={fmt(arch['cosine_sim'])}")

    if arch["mse"] > std["mse"] * 5:
        issues.append(
            "[FAIL] RA-SAE reconstruction is >5x worse than Standard (possible training collapse)."
        )

    if arch["variance_explained"] < 0.7 * std["variance_explained"]:
        issues.append("[FAIL] RA-SAE variance explained is far below baseline.")

    if arch["l0"] < 0.8 * std["l0"]:
        issues.append(
            "[WARN] RA-SAE L0 is much lower than Standard despite same TopK (many selected activations may be effectively zero)."
        )

    return issues


def check_stability() -> list[str]:
    issues: list[str] = []
    if not os.path.exists("stability_results.pt"):
        return ["[WARN] stability_results.pt not found (skip seed-stability check)."]

    stab = torch.load("stability_results.pt", map_location="cpu", weights_only=False)
    std_stab = sum(stab["std_stabilities"]) / len(stab["std_stabilities"])
    arch_stab = sum(stab["arch_stabilities"]) / len(stab["arch_stabilities"])

    print("\n=== stability_results.pt ===")
    print(f"Dictionary stability: standard={fmt(std_stab)} | ra={fmt(arch_stab)}")

    if arch_stab <= std_stab:
        issues.append("[FAIL] RA-SAE is not more stable than Standard (core hypothesis fails).")
    elif arch_stab > std_stab:
        issues.append("[OK] RA-SAE is more stable across seeds.")

    return issues


def check_ablation() -> list[str]:
    issues: list[str] = []
    if not os.path.exists("ablation_results.pt"):
        return ["[WARN] ablation_results.pt not found (skip delta sweep check)."]

    abl = torch.load("ablation_results.pt", map_location="cpu", weights_only=False)
    deltas = abl["delta_values"]
    stabilities = abl["stabilities"]
    mses = abl["mses"]

    print("\n=== ablation_results.pt ===")
    for d, s, m in zip(deltas, stabilities, mses):
        print(f"delta={d:<4} stability={fmt(s):>8}  mse={fmt(m):>10}")

    best_mse = min(mses)
    worst_mse = max(mses)
    if best_mse > 0 and worst_mse / best_mse > 50:
        issues.append("[WARN] Delta sweep has extreme MSE variance (>50x), suggests unstable optimization at some deltas.")

    return issues


def main() -> None:
    print("=" * 70)
    print("EXPERIMENT SANITY CHECK")
    print("=" * 70)

    issues: list[str] = []
    issues.extend(check_eval())
    issues.extend(check_stability())
    issues.extend(check_ablation())

    print("\n--- Diagnosis ---")
    if not issues:
        print("No issues detected.")
        return

    for item in issues:
        print(item)

    if any(x.startswith("[FAIL]") for x in issues):
        print("\nOverall: experiment ran, but at least one key metric indicates it likely did NOT work as intended.")
    elif any(x.startswith("[WARN]") for x in issues):
        print("\nOverall: experiment appears partially successful, with notable warnings.")
    else:
        print("\nOverall: experiment appears healthy.")


if __name__ == "__main__":
    main()
