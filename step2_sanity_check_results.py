"""
Sanity-check experiment outputs and flag common failure modes.

Usage:
  python step2_sanity_check_results.py

Reads:
  - eval_results.pt
  - stability_results.pt (optional)
  - ablation_results.pt (optional)

The ablation section now reports:
  * best delta for reconstruction (min MSE)
  * best delta for stability (max cosine stability)
  * balanced delta (simple normalized tradeoff score)
  * warnings for extreme/outlier MSE values and malformed outputs
"""

from __future__ import annotations

import math
import os
from typing import Any

import torch


def fmt(x: float) -> str:
    return f"{x:.4f}" if isinstance(x, (float, int)) and math.isfinite(float(x)) else str(x)


def safe_mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


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
        issues.append("[FAIL] RA-SAE reconstruction is >5x worse than Standard (possible training collapse).")

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
    std_stab = safe_mean(stab["std_stabilities"])
    arch_stab = safe_mean(stab["arch_stabilities"])

    print("\n=== stability_results.pt ===")
    print(f"Dictionary stability: standard={fmt(std_stab)} | ra={fmt(arch_stab)}")

    if arch_stab <= std_stab:
        issues.append("[FAIL] RA-SAE is not more stable than Standard (core hypothesis fails).")
    else:
        issues.append("[OK] RA-SAE is more stable across seeds.")

    return issues


def _normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def _argmax(values: list[float]) -> int:
    return max(range(len(values)), key=lambda i: values[i])


def _argmin(values: list[float]) -> int:
    return min(range(len(values)), key=lambda i: values[i])


def _extract_ablation_payload(abl: dict[str, Any]) -> tuple[list[float], list[float], list[float]]:
    deltas = list(abl.get("delta_values", []))
    stabilities = list(abl.get("stabilities", []))
    mses = list(abl.get("mses", []))
    return deltas, stabilities, mses


def check_ablation() -> list[str]:
    issues: list[str] = []
    if not os.path.exists("ablation_results.pt"):
        return ["[WARN] ablation_results.pt not found (skip delta sweep check)."]

    abl = torch.load("ablation_results.pt", map_location="cpu", weights_only=False)
    deltas, stabilities, mses = _extract_ablation_payload(abl)

    print("\n=== ablation_results.pt ===")

    if not deltas or len(deltas) != len(stabilities) or len(deltas) != len(mses):
        return [
            "[FAIL] ablation_results.pt is malformed (delta_values/stabilities/mses missing or length mismatch)."
        ]

    for d, s, m in zip(deltas, stabilities, mses):
        print(f"delta={d:<4} stability={fmt(float(s)):>8}  mse={fmt(float(m)):>10}")

    best_mse_idx = _argmin(mses)
    best_stability_idx = _argmax(stabilities)

    mse_norm = _normalize(mses)
    stab_norm = _normalize(stabilities)
    # Higher score is better: high stability + low MSE.
    balanced_scores = [0.5 * stab_norm[i] + 0.5 * (1.0 - mse_norm[i]) for i in range(len(deltas))]
    best_balanced_idx = _argmax(balanced_scores)

    print("\nAblation summary:")
    print(f"  Best reconstruction delta: {deltas[best_mse_idx]} (MSE={fmt(float(mses[best_mse_idx]))})")
    print(
        f"  Best stability delta:      {deltas[best_stability_idx]} "
        f"(stability={fmt(float(stabilities[best_stability_idx]))})"
    )
    print(
        f"  Best balanced delta:       {deltas[best_balanced_idx]} "
        f"(score={fmt(float(balanced_scores[best_balanced_idx]))})"
    )

    best_mse = min(mses)
    worst_mse = max(mses)
    if best_mse > 0 and worst_mse / best_mse > 50:
        issues.append("[WARN] Delta sweep has extreme MSE variance (>50x), suggests unstable optimization at some deltas.")

    # Spot check common question: is default delta=1.0 pathological?
    if 1.0 in deltas:
        i = deltas.index(1.0)
        if mses[i] > 10 * best_mse:
            issues.append(
                "[WARN] delta=1.0 is a strong MSE outlier (>10x best MSE). Consider using the ablation-selected delta."
            )

    if deltas[best_mse_idx] == deltas[best_stability_idx]:
        issues.append("[OK] Ablation shows one delta that is best for both stability and reconstruction.")
    else:
        issues.append("[OK] Ablation shows a real tradeoff; choose delta from the reported balanced point.")

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
