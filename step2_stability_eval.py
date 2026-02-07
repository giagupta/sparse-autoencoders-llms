"""
Stability Evaluation: The core experiment from Fel et al. (2025).

Measures whether training the same SAE architecture with different random seeds
produces consistent dictionaries. This is the paper's key finding:
  - Standard SAEs are unstable (cosine similarity ~0.5 between runs)
  - Archetypal SAEs are much more stable (higher cosine similarity)

Procedure:
  1. Train N_SEEDS instances of each model type (different random seeds)
  2. For each pair of trained models, compute cosine similarity between
     their best-matching dictionary atoms
  3. Report mean stability score for each architecture

Reference: Fel et al. (2025), "Archetypal SAE", ICML 2025.
"""

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from datasets import load_dataset
from transformers import GPT2Model, GPT2Tokenizer
from archetypal_sae import ArchetypalSAE
from standard_sae import StandardSAE

# --- Configuration ---
LAYER = 9
N_FEATURES = 4096
TOP_K = 64
DELTA = 1.0
LR = 3e-4
TRAINING_STEPS = 3000       # Shorter per-seed to make multi-seed feasible
MAX_SEQ_LEN = 128
N_SEEDS = 3                 # Number of random seeds to evaluate stability
AUX_LOSS_COEFF = 1/32


def collect_training_data(gpt2, tokenizer, device, n_steps=3000):
    """Pre-collect activations to ensure identical data across seeds."""
    print("  Pre-collecting training activations...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", streaming=False)
    activations = []

    for example in dataset:
        text = example["text"].strip()
        if len(text) < 100:
            continue

        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN).to(device)
        with torch.no_grad():
            outputs = gpt2(inputs.input_ids)
            real_acts = gpt2.h[LAYER].mlp(outputs.last_hidden_state)
            activations.append(real_acts.cpu())

        if len(activations) >= n_steps:
            break

    print(f"  Collected {len(activations)} training samples")
    return activations


def train_standard_sae(activations, device, seed):
    """Train a standard TopK SAE with a given seed."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    model = StandardSAE(d_model=768, n_features=N_FEATURES, top_k=TOP_K).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    feature_activity = torch.zeros(N_FEATURES, device=device)

    for step, acts in enumerate(activations):
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
        optimizer.step()

        with torch.no_grad():
            model.dictionary.data = F.normalize(model.dictionary.data, dim=-1)

        if step % 500 == 0:
            print(f"    Seed {seed} | Step {step} | MSE: {mse_loss.item():.6f}")

    model.eval()
    return model


def train_archetypal_sae(activations, anchor_points, device, seed):
    """Train an RA-SAE with a given seed."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    model = ArchetypalSAE(
        d_model=768, n_features=N_FEATURES, anchor_points=anchor_points,
        top_k=TOP_K, delta=DELTA, use_multiplier=True,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    feature_activity = torch.zeros(N_FEATURES, device=device)

    for step, acts in enumerate(activations):
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
        optimizer.step()

        if step % 500 == 0:
            print(f"    Seed {seed} | Step {step} | MSE: {mse_loss.item():.6f}")

    model.eval()
    return model


def compute_dictionary_stability(dict_a, dict_b):
    """
    Compute stability between two dictionaries using greedy best-match cosine similarity.

    For each atom in dict_a, find its best match in dict_b (highest cosine similarity),
    then average over all atoms. This is the stability metric from the paper.

    Parameters
    ----------
    dict_a, dict_b : Tensor of shape (n_features, d_model)

    Returns
    -------
    mean_cosine : float
        Average best-match cosine similarity (higher = more stable).
    per_atom_cosine : Tensor
        Best-match cosine similarity for each atom in dict_a.
    """
    # Normalize dictionaries
    dict_a_norm = F.normalize(dict_a, dim=-1)
    dict_b_norm = F.normalize(dict_b, dim=-1)

    # Cosine similarity matrix: (n_features, n_features)
    cos_sim = dict_a_norm @ dict_b_norm.T

    # Best match for each atom in dict_a
    best_match_cosine, _ = cos_sim.max(dim=-1)

    return best_match_cosine.mean().item(), best_match_cosine


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        device = "mps"

    print("=" * 60)
    print("STABILITY EVALUATION")
    print(f"Training {N_SEEDS} seeds for each architecture")
    print("=" * 60)

    # Load GPT-2
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    gpt2 = GPT2Model.from_pretrained("gpt2").to(device)
    gpt2.eval()

    # Pre-collect training data (shared across all seeds)
    activations = collect_training_data(gpt2, tokenizer, device, TRAINING_STEPS)

    # Load centroids for RA-SAE
    anchor_points = torch.load("anchor_points.pt", weights_only=True).to(device)

    # Train multiple seeds
    seeds = list(range(N_SEEDS))
    std_models = []
    arch_models = []

    for seed in seeds:
        print(f"\n--- Training Standard TopK SAE (seed={seed}) ---")
        std_model = train_standard_sae(activations, device, seed)
        std_models.append(std_model)

        print(f"\n--- Training RA-SAE (seed={seed}) ---")
        arch_model = train_archetypal_sae(activations, anchor_points, device, seed)
        arch_models.append(arch_model)

    # Extract dictionaries
    print("\n\nExtracting dictionaries...")
    std_dicts = []
    arch_dicts = []

    for model in std_models:
        model.eval()
        std_dicts.append(model.get_dictionary().detach())

    for model in arch_models:
        model.eval()
        arch_dicts.append(model.get_dictionary().detach())

    # Compute pairwise stability
    print("\nComputing stability metrics...")
    std_stabilities = []
    arch_stabilities = []

    for i in range(N_SEEDS):
        for j in range(i + 1, N_SEEDS):
            std_score, _ = compute_dictionary_stability(std_dicts[i], std_dicts[j])
            std_stabilities.append(std_score)
            print(f"  Standard SAE (seed {i} vs {j}): cosine stability = {std_score:.4f}")

            arch_score, _ = compute_dictionary_stability(arch_dicts[i], arch_dicts[j])
            arch_stabilities.append(arch_score)
            print(f"  RA-SAE       (seed {i} vs {j}): cosine stability = {arch_score:.4f}")

    # Also compute reconstruction MSE for each model
    print("\nComputing reconstruction MSE on held-out data...")
    eval_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", streaming=False)
    eval_acts = []
    for example in eval_dataset:
        text = example["text"].strip()
        if len(text) < 50:
            continue
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN).to(device)
        with torch.no_grad():
            outputs = gpt2(inputs.input_ids)
            real_acts = gpt2.h[LAYER].mlp(outputs.last_hidden_state)
            eval_acts.append(real_acts)
        if len(eval_acts) >= 200:
            break

    std_mses = []
    arch_mses = []
    for model in std_models:
        mses = []
        for acts in eval_acts:
            with torch.no_grad():
                recon, _, _ = model(acts)
                mses.append(F.mse_loss(recon, acts).item())
        std_mses.append(np.mean(mses))

    for model in arch_models:
        mses = []
        for acts in eval_acts:
            with torch.no_grad():
                recon, _, _ = model(acts)
                mses.append(F.mse_loss(recon, acts).item())
        arch_mses.append(np.mean(mses))

    # Save results
    results = {
        'std_stabilities': std_stabilities,
        'arch_stabilities': arch_stabilities,
        'std_mses': std_mses,
        'arch_mses': arch_mses,
        'n_seeds': N_SEEDS,
        'seeds': seeds,
    }
    torch.save(results, "stability_results.pt")

    # Print summary
    print(f"\n{'=' * 60}")
    print("STABILITY RESULTS")
    print(f"{'=' * 60}")
    print(f"\nDictionary Stability (cosine similarity, higher = more stable):")
    print(f"  Standard TopK SAE: {np.mean(std_stabilities):.4f} +/- {np.std(std_stabilities):.4f}")
    print(f"  RA-SAE:            {np.mean(arch_stabilities):.4f} +/- {np.std(arch_stabilities):.4f}")
    print(f"\nReconstruction MSE (lower = better):")
    print(f"  Standard TopK SAE: {np.mean(std_mses):.6f} +/- {np.std(std_mses):.6f}")
    print(f"  RA-SAE:            {np.mean(arch_mses):.6f} +/- {np.std(arch_mses):.6f}")
    print(f"\nConclusion:")

    if np.mean(arch_stabilities) > np.mean(std_stabilities):
        improvement = (np.mean(arch_stabilities) - np.mean(std_stabilities)) / np.mean(std_stabilities) * 100
        print(f"  RA-SAE is MORE STABLE by {improvement:.1f}% relative improvement")
    else:
        print(f"  Standard SAE is more stable (unexpected result)")

    mse_diff = (np.mean(arch_mses) - np.mean(std_mses)) / np.mean(std_mses) * 100
    print(f"  MSE difference: {mse_diff:+.1f}% (positive = RA-SAE has higher MSE)")

    print(f"\nResults saved to: stability_results.pt")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
