#!/usr/bin/env python3
"""
Master Script: Complete RA-SAE vs Standard TopK SAE Comparison Pipeline.

Implements the experiment from:
  Fel et al. (2025), "Archetypal SAE: Adaptive and Stable Dictionary Learning
  for Concept Extraction in Large Vision Models", ICML 2025.

Adapted for the language model regime (GPT-2).

Pipeline:
  1. Extract K-means centroids from GPT-2 activations
  2. Train Standard TopK SAE (baseline)
  3. Train RA-SAE (Relaxed Archetypal SAE)
  4. Evaluate standard SAE metrics (MSE, R^2, L0, cosine sim, dead features)
  5. Evaluate dictionary stability across seeds
  6. Delta ablation (sweep relaxation parameter)
  7. Generate comparison visualizations
"""

import subprocess
import sys
import os


def run_script(script_name, description):
    """Run a Python script and handle errors."""
    print(f"\n{'=' * 70}")
    print(f"STEP: {description}")
    print(f"{'=' * 70}")

    try:
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            capture_output=False,
            text=True,
        )
        print(f"[OK] {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[FAIL] Error in {description}")
        print(f"Script: {script_name}, Return code: {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"[FAIL] Script not found: {script_name}")
        return False


def main():
    print(f"\n{'=' * 70}")
    print("RA-SAE vs STANDARD TopK SAE: FULL COMPARISON PIPELINE")
    print("Replicating Fel et al. (2025) in the Language Model Regime")
    print(f"{'=' * 70}")
    print("\nPipeline steps:")
    print("  1. Extract K-means centroids from GPT-2 Layer 9 activations")
    print("  2. Train Standard TopK SAE (baseline)")
    print("  3. Train RA-SAE (Relaxed Archetypal SAE)")
    print("  4. Evaluate standard SAE metrics (MSE, R^2, L0, etc.)")
    print("  5. Evaluate dictionary stability (multi-seed)")
    print("  6. Delta ablation (sweep relaxation parameter)")
    print("  7. Generate comparison visualizations")
    print(f"{'=' * 70}")

    # Step 1: Extract centroids
    if not os.path.exists("anchor_points.pt"):
        if not run_script("step1_get_anchors.py", "Extract K-means centroids"):
            print("\n[FAIL] Pipeline failed at centroid extraction")
            return
    else:
        print("\n[OK] anchor_points.pt found, skipping centroid extraction")

    # Step 2: Train Standard TopK SAE
    if not os.path.exists("standard_sae_weights.pt"):
        if not run_script("step2_train_standard_sae.py", "Train Standard TopK SAE"):
            print("\n[FAIL] Pipeline failed at standard SAE training")
            return
    else:
        print("\n[OK] standard_sae_weights.pt found, skipping training")

    # Step 3: Train RA-SAE
    if not os.path.exists("archetypal_sae_weights_v2.pt"):
        if not run_script("step3_train_sae.py", "Train RA-SAE"):
            print("\n[FAIL] Pipeline failed at RA-SAE training")
            return
    else:
        print("\n[OK] archetypal_sae_weights_v2.pt found, skipping training")

    # Step 4: Evaluate standard metrics
    if not run_script("step2_compare_monosemanticity.py", "Evaluate standard SAE metrics"):
        print("\n[FAIL] Pipeline failed at evaluation")
        return

    # Step 5: Stability evaluation (multi-seed, slower)
    if not run_script("step2_stability_eval.py", "Evaluate dictionary stability (multi-seed)"):
        print("\n[WARN] Stability evaluation failed, continuing without it")

    # Step 6: Delta ablation (sweep relaxation parameter)
    if not run_script("step2_delta_ablation.py", "Delta ablation (sweep relaxation parameter)"):
        print("\n[WARN] Delta ablation failed, continuing without it")

    # Step 7: Generate visualizations
    if not run_script("step2_visualize_comparison.py", "Generate comparison visualizations"):
        print("\n[FAIL] Pipeline failed at visualization")
        return

    # Done
    print(f"\n{'=' * 70}")
    print("[OK] PIPELINE COMPLETE!")
    print(f"{'=' * 70}")
    print("\nGenerated files:")
    print("  Models:")
    print("    - anchor_points.pt              (K-means centroids)")
    print("    - standard_sae_weights.pt       (Standard TopK SAE)")
    print("    - archetypal_sae_weights_v2.pt  (RA-SAE)")
    print("  Results:")
    print("    - eval_results.pt               (standard SAE metrics)")
    if os.path.exists("stability_results.pt"):
        print("    - stability_results.pt          (dictionary stability)")
    if os.path.exists("ablation_results.pt"):
        print("    - ablation_results.pt           (delta ablation)")
    print("  Visualizations:")
    print("    - eval_comparison.png")
    if os.path.exists("stability_comparison.png"):
        print("    - stability_comparison.png")
    if os.path.exists("delta_ablation_pareto.png"):
        print("    - delta_ablation_pareto.png")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
