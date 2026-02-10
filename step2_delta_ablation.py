"""
Delta Ablation: Sweep the relaxation parameter to map the
reconstruction-quality vs dictionary-stability Pareto frontier.

delta=0 → pure archetypal (dictionary atoms are exact convex combos of centroids)
delta=inf → unconstrained (RA-SAE degenerates to a standard SAE)

For each delta value we:
  1. Train 2 RA-SAE seeds on identical data
  2. Measure pairwise dictionary stability (cosine sim)
  3. Measure reconstruction MSE on held-out data

Reference: Fel et al. (2025), "Archetypal SAE", ICML 2025, Fig. 5.
"""

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from datasets import load_dataset
from transformers import GPT2Model, GPT2Tokenizer
from archetypal_sae import ArchetypalSAE
from step2_stability_eval import (
    collect_training_data,
    compute_dictionary_stability,
)

# --- Configuration ---
LAYER = 9
N_FEATURES = 4096
TOP_K = 64
LR = 3e-4
TRAINING_STEPS = 2000       # Shorter per-seed since we sweep many deltas
MAX_SEQ_LEN = 128
N_SEEDS = 2                 # 2 seeds per delta to measure stability
AUX_LOSS_COEFF = 1 / 32
DELTA_VALUES = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]


def train_rasae_for_ablation(activations, anchor_points, device, seed, delta):
    """Train an RA-SAE with a given seed and delta value."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    model = ArchetypalSAE(
        d_model=768, n_features=N_FEATURES, anchor_points=anchor_points,
        top_k=TOP_K, delta=delta, use_multiplier=True,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    feature_activity = torch.zeros(N_FEATURES, device=device)

    for step, acts in enumerate(activations):
        if step >= TRAINING_STEPS:
            break

        real_acts = acts.to(device)
        reconstruction, codes, pre_codes = model(real_acts)

        mse_loss = F.mse_loss(reconstruction, real_acts)

        with torch.no_grad():
            feature_activity = 0.999 * feature_activity + 0.001 * (codes.abs().sum(dim=(0, 1)) > 0).float()
        dead_mask = (feature_activity < 0.01).float()
        if dead_mask.sum() > 0:
            dead_pre_codes = pre_codes * dead_mask.unsqueeze(0).unsqueeze(0)
            aux_loss = AUX_LOSS_COEFF * F.relu(-dead_pre_codes).mean()
        else:
            aux_loss = torch.tensor(0.0, device=device)

        total_loss = mse_loss + aux_loss
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 500 == 0:
            print(f"      delta={delta} seed={seed} | Step {step} | MSE: {mse_loss.item():.4f}")

    model.eval()
    return model


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        device = "mps"

    print("=" * 60)
    print("DELTA ABLATION STUDY")
    print(f"Sweeping delta in {DELTA_VALUES}")
    print(f"Training {N_SEEDS} seeds x {len(DELTA_VALUES)} deltas = {N_SEEDS * len(DELTA_VALUES)} runs")
    print("=" * 60)

    # Load GPT-2
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    gpt2 = GPT2Model.from_pretrained("gpt2").to(device)
    gpt2.eval()

    # Pre-collect training data (shared across all runs)
    activations = collect_training_data(gpt2, tokenizer, device, TRAINING_STEPS)

    # Collect held-out evaluation data
    print("\n  Collecting held-out evaluation data...")
    eval_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", streaming=True)
    eval_acts = []
    for example in eval_dataset:
        text = example["text"].strip()
        if len(text) < 50:
            continue
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN).to(device)
        with torch.no_grad():
            outputs = gpt2(inputs.input_ids, output_hidden_states=True)
            real_acts = outputs.hidden_states[LAYER + 1]
            eval_acts.append(real_acts)
        if len(eval_acts) >= 200:
            break

    # Load centroids
    anchor_points = torch.load("anchor_points.pt", weights_only=True).to(device)

    results = {
        'delta_values': DELTA_VALUES,
        'stabilities': [],      # mean stability per delta
        'mses': [],             # mean MSE per delta
        'stabilities_std': [],  # std of stabilities per delta
        'mses_std': [],         # std of MSEs per delta
    }

    for delta in DELTA_VALUES:
        print(f"\n{'─' * 50}")
        print(f"  Delta = {delta}")
        print(f"{'─' * 50}")

        # Train N_SEEDS models with this delta
        models = []
        for seed in range(N_SEEDS):
            print(f"\n    Training seed {seed}...")
            model = train_rasae_for_ablation(activations, anchor_points, device, seed, delta)
            models.append(model)

        # Extract dictionaries
        dicts = [m.get_dictionary().detach() for m in models]

        # Compute pairwise stability
        pair_stabilities = []
        for i in range(N_SEEDS):
            for j in range(i + 1, N_SEEDS):
                score, _ = compute_dictionary_stability(dicts[i], dicts[j])
                pair_stabilities.append(score)
                print(f"    Stability (seed {i} vs {j}): {score:.4f}")

        # Compute reconstruction MSE for each model
        model_mses = []
        for model in models:
            mses = []
            for acts in eval_acts:
                with torch.no_grad():
                    recon, _, _ = model(acts)
                    mses.append(F.mse_loss(recon, acts).item())
            model_mses.append(np.mean(mses))

        mean_stability = np.mean(pair_stabilities)
        mean_mse = np.mean(model_mses)

        results['stabilities'].append(mean_stability)
        results['mses'].append(mean_mse)
        results['stabilities_std'].append(np.std(pair_stabilities) if len(pair_stabilities) > 1 else 0.0)
        results['mses_std'].append(np.std(model_mses))

        print(f"    => Stability: {mean_stability:.4f}, MSE: {mean_mse:.4f}")

    # Save results
    torch.save(results, "ablation_results.pt")

    # Print summary table
    print(f"\n{'=' * 60}")
    print("DELTA ABLATION RESULTS")
    print(f"{'=' * 60}")
    print(f"{'Delta':<10} {'Stability':>12} {'MSE':>12}")
    print("-" * 40)
    for i, delta in enumerate(DELTA_VALUES):
        print(f"{delta:<10.1f} {results['stabilities'][i]:>12.4f} {results['mses'][i]:>12.4f}")
    print(f"{'=' * 60}")
    print("Saved to: ablation_results.pt")


if __name__ == "__main__":
    main()
