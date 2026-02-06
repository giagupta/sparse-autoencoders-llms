#!/usr/bin/env python3
"""
Master Script: Complete SAE Comparison Pipeline
Runs training, evaluation, and visualization for both Archetypal and Standard SAEs
"""

import subprocess
import sys
import os

def run_script(script_name, description):
    """Run a Python script and handle errors"""
    print("\n" + "=" * 70)
    print(f"STEP: {description}")
    print("=" * 70)
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            capture_output=False,
            text=True
        )
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error in {description}")
        print(f"Script: {script_name}")
        print(f"Return code: {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"✗ Script not found: {script_name}")
        return False

def check_prerequisites():
    """Check if required files exist"""
    print("Checking prerequisites...")
    
    required_files = [
        "anchor_points.pt",
        "archetypal_sae_weights_v2.pt",
        "archetypal_sae.py"
    ]
    
    missing = []
    for f in required_files:
        if not os.path.exists(f):
            missing.append(f)
    
    if missing:
        print("\n⚠ WARNING: Missing required files:")
        for f in missing:
            print(f"  - {f}")
        
        if "archetypal_sae.py" in missing:
            print("\n✗ CRITICAL: archetypal_sae.py is required")
            print("Please ensure the ArchetypalSAE class definition is available")
            return False
        
        if "anchor_points.pt" in missing:
            print("\n→ Will need to run step1_get_anchors.py first")
        
        if "archetypal_sae_weights_v2.pt" in missing:
            print("\n→ Will need to run step3_train_sae.py first")
        
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            return False
    
    print("✓ Prerequisites OK\n")
    return True

def main():
    print("\n" + "=" * 70)
    print("ARCHETYPAL vs STANDARD SAE COMPARISON PIPELINE")
    print("=" * 70)
    print("\nThis pipeline will:")
    print("  1. Train a Standard SAE (matching your Archetypal SAE setup)")
    print("  2. Compute monosemanticity metrics for both models")
    print("  3. Generate comparison visualizations")
    print("  4. Produce statistical analysis")
    print("\nEstimated time: ~10-20 minutes (depending on hardware)")
    print("=" * 70)
    
    if not check_prerequisites():
        print("\n✗ Aborting pipeline")
        return
    
    # Step 1: Train Standard SAE
    if not os.path.exists("standard_sae_weights.pt"):
        print("\nStandard SAE weights not found. Training now...")
        if not run_script("step2_train_standard_sae.py", "Training Standard SAE"):
            print("\n✗ Pipeline failed at training step")
            return
    else:
        print("\n✓ Standard SAE weights found, skipping training")
    
    # Step 2: Compare Monosemanticity
    if not os.path.exists("monosemanticity_results.pt"):
        if not run_script("step2_compare_monosemanticity.py", "Computing Monosemanticity Metrics"):
            print("\n✗ Pipeline failed at evaluation step")
            return
    else:
        print("\n✓ Monosemanticity results found")
        response = input("Recompute metrics? (y/n): ")
        if response.lower() == 'y':
            if not run_script("step2_compare_monosemanticity.py", "Computing Monosemanticity Metrics"):
                print("\n✗ Pipeline failed at evaluation step")
                return
    
    # Step 3: Visualize Results
    if not run_script("step2_visualize_comparison.py", "Generating Visualizations"):
        print("\n✗ Pipeline failed at visualization step")
        return
    
    # Success!
    print("\n" + "=" * 70)
    print("✓ PIPELINE COMPLETE!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - standard_sae_weights.pt")
    print("  - monosemanticity_results.pt")
    print("  - monosemanticity_comparison.png")
    print("  - monosemanticity_boxplots.png")
    print("\nYou can now use these results for your paper!")
    print("=" * 70)

if __name__ == "__main__":
    main()