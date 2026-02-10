"""
Evaluate and compare RA-SAE vs Standard TopK SAE.

Standard SAE metrics:
1. Reconstruction quality: MSE, variance explained (R^2)
2. Sparsity: L0 (number of active features per token)
3. Feature utilization: fraction of features that fire, dead features
4. Reconstruction fidelity: cosine similarity between input and reconstruction
"""

import time
import torch
import torch.nn.functional as F
import numpy as np
from transformers import GPT2Model, GPT2Tokenizer
from datasets import load_dataset
from archetypal_sae import ArchetypalSAE
from standard_sae import StandardSAE

# --- Configuration ---
LAYER = 9
N_FEATURES = 4096
TOP_K = 64
N_EVAL_SAMPLES = 500


def evaluate_model(model, model_name, gpt2, tokenizer, device, n_samples=500):
    """
    Evaluate a single SAE on standard metrics.

    Returns dict with:
      - mse: mean reconstruction MSE
      - variance_explained: 1 - MSE/Var(x), i.e. R^2
      - l0: mean number of active features per token position
      - cosine_sim: mean cosine similarity between input and reconstruction
      - feature_firing_rate: fraction of features that fire at least once
      - dead_features: number of features that never fire
      - mean_activation: mean activation magnitude of active features
    """
    print(f"\nEvaluating {model_name}...")
    print("-" * 50)

    model.eval()

    all_mse = []
    all_l0 = []
    all_cosine = []
    all_x_var = []
    feature_fired = torch.zeros(N_FEATURES, device=device)

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", streaming=False)

    start_time = time.time()
    processed = 0

    for example in dataset:
        text = example["text"].strip()
        if len(text) < 50:
            continue

        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)

        with torch.no_grad():
            outputs = gpt2(inputs.input_ids)
            x = gpt2.h[LAYER].mlp(outputs.last_hidden_state)

            reconstruction, codes, _ = model(x)

            # Collect per-feature statistics efficiently: iterate only active codes.
            # With TopK activations, this is dramatically faster than scanning all features.
            code_slice = codes[0]  # (seq_len, n_features)
            active_positions = torch.nonzero(code_slice > ACTIVATION_THRESHOLD, as_tuple=False)

            for token_idx, feat_id in active_positions:
                token_idx = token_idx.item()
                feat_id = feat_id.item()
                act_val = code_slice[token_idx, feat_id].item()
                token_str = tokenizer.decode(inputs.input_ids[0, token_idx])

                feature_stats[feat_id]['token_activations'][token_str].append(act_val)
                feature_stats[feat_id]['activation_count'] += 1

                if act_val > 2.0 and len(feature_stats[feat_id]['contexts']) < 5:
                    ctx_start = max(0, token_idx - 5)
                    ctx_end = min(inputs.input_ids.size(1), token_idx + 5)
                    context = tokenizer.decode(inputs.input_ids[0, ctx_start:ctx_end])
                    feature_stats[feat_id]['contexts'].append(context)

        processed += 1
        if processed % 50 == 0:
            active = len([f for f in feature_stats if feature_stats[f]['activation_count'] > 0])
            elapsed = time.time() - start_time
            print(
                f"  Processed {processed} samples | Active features: {active} | "
                f"Avg MSE: {np.mean(mse_scores):.6f} | Elapsed: {elapsed:.1f}s"
            )

        if processed >= n_samples:
            break

        if count >= n_samples:
            break

    # Compute aggregate metrics
    mean_mse = np.mean(all_mse)
    mean_x_var = np.mean(all_x_var)
    variance_explained = 1.0 - mean_mse / (mean_x_var + 1e-10)
    n_alive = (feature_fired > 0).sum().item()

    results = {
        'mse': mean_mse,
        'mse_std': np.std(all_mse),
        'variance_explained': variance_explained,
        'l0': np.mean(all_l0),
        'l0_std': np.std(all_l0),
        'cosine_sim': np.mean(all_cosine),
        'cosine_sim_std': np.std(all_cosine),
        'alive_features': n_alive,
        'dead_features': N_FEATURES - n_alive,
        'feature_utilization': n_alive / N_FEATURES,
    }

    print(f"  MSE:                {results['mse']:.4f} +/- {results['mse_std']:.4f}")
    print(f"  Variance Explained: {results['variance_explained']:.4f}")
    print(f"  L0:                 {results['l0']:.1f}")
    print(f"  Cosine Similarity:  {results['cosine_sim']:.4f}")
    print(f"  Alive Features:     {n_alive}/{N_FEATURES} ({results['feature_utilization']:.1%})")

    return results


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        device = "mps"

    print("=" * 60)
    print("SAE EVALUATION: RA-SAE vs Standard TopK SAE")
    print("=" * 60)

    # Load models
    print("\nLoading models...")
    anchor_points = torch.load("anchor_points.pt", weights_only=True).to(device)

    archetypal_sae = ArchetypalSAE(
        d_model=768, n_features=N_FEATURES, anchor_points=anchor_points, top_k=TOP_K
    ).to(device)
    archetypal_sae.load_state_dict(torch.load("archetypal_sae_weights_v2.pt", weights_only=True))
    archetypal_sae.eval()

    standard_sae = StandardSAE(d_model=768, n_features=N_FEATURES, top_k=TOP_K).to(device)
    standard_sae.load_state_dict(torch.load("standard_sae_weights.pt", weights_only=True))
    standard_sae.eval()

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    gpt2 = GPT2Model.from_pretrained("gpt2").to(device)
    gpt2.eval()

    # Evaluate
    arch_results = evaluate_model(archetypal_sae, "RA-SAE (Archetypal)", gpt2, tokenizer, device, N_EVAL_SAMPLES)
    std_results = evaluate_model(standard_sae, "Standard TopK SAE", gpt2, tokenizer, device, N_EVAL_SAMPLES)

    # Save
    torch.save({
        'archetypal': arch_results,
        'standard': std_results,
    }, "eval_results.pt")

    # Print comparison
    print(f"\n{'=' * 60}")
    print("COMPARISON SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Metric':<25} {'RA-SAE':>15} {'Standard':>15}")
    print("-" * 60)
    print(f"{'MSE':<25} {arch_results['mse']:>15.4f} {std_results['mse']:>15.4f}")
    print(f"{'Var Explained (R^2)':<25} {arch_results['variance_explained']:>15.4f} {std_results['variance_explained']:>15.4f}")
    print(f"{'L0 (sparsity)':<25} {arch_results['l0']:>15.1f} {std_results['l0']:>15.1f}")
    print(f"{'Cosine Similarity':<25} {arch_results['cosine_sim']:>15.4f} {std_results['cosine_sim']:>15.4f}")
    print(f"{'Alive Features':<25} {arch_results['alive_features']:>15.0f} {std_results['alive_features']:>15.0f}")
    print(f"{'Dead Features':<25} {arch_results['dead_features']:>15.0f} {std_results['dead_features']:>15.0f}")
    print(f"{'Feature Utilization':<25} {arch_results['feature_utilization']:>14.1%} {std_results['feature_utilization']:>14.1%}")
    print(f"{'=' * 60}")
    print("Saved to: eval_results.pt")


if __name__ == "__main__":
    main()
