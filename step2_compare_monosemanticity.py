"""
Evaluate and compare Archetypal SAE vs Standard SAE.

Measures:
1. Reconstruction quality (MSE on held-out data)
2. Monosemanticity (token entropy, activation consistency)
3. Feature utilization (dead features, activation frequency)

The stability evaluation (the paper's core contribution) is in a separate script
(step2_stability_eval.py) since it requires training multiple seeds.
"""

import torch
import numpy as np
from collections import defaultdict
from transformers import GPT2Model, GPT2Tokenizer
from datasets import load_dataset
from archetypal_sae import ArchetypalSAE
from standard_sae import StandardSAE

# --- Configuration ---
LAYER = 9
N_FEATURES = 4096
TOP_K = 64
N_EVAL_SAMPLES = 500
ACTIVATION_THRESHOLD = 0.1  # Consider feature "active" above this


def compute_token_entropy(token_activations):
    """
    Compute entropy over token distribution for a feature.
    Lower entropy = more monosemantic (fires on fewer token types).
    """
    if not token_activations:
        return float('inf')

    token_counts = {tok: len(acts) for tok, acts in token_activations.items()}
    total = sum(token_counts.values())
    if total == 0:
        return float('inf')

    probs = np.array([count / total for count in token_counts.values()])
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    return entropy


def compute_activation_consistency(token_activations):
    """
    Compute coefficient of variation for activation strengths.
    Lower CoV = more consistent activation pattern.
    """
    all_acts = []
    for acts in token_activations.values():
        all_acts.extend(acts)

    if len(all_acts) < 2:
        return float('inf')

    all_acts = np.array(all_acts)
    mean_act = np.mean(all_acts)
    std_act = np.std(all_acts)

    if mean_act == 0:
        return float('inf')

    return std_act / mean_act


def evaluate_model(model, model_name, gpt2, tokenizer, device, n_samples=500):
    """
    Evaluate a single SAE model on reconstruction quality and monosemanticity.

    Returns dict with:
      - mse_scores: list of per-sample MSE values
      - feature_stats: per-feature activation statistics
      - monosemanticity: per-feature monosemanticity scores
    """
    print(f"\nEvaluating {model_name}...")
    print("=" * 60)

    model.eval()
    mse_scores = []
    feature_stats = defaultdict(lambda: {
        'token_activations': defaultdict(list),
        'activation_count': 0,
        'contexts': []
    })

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", streaming=True)

    for i, example in enumerate(dataset):
        text = example["text"].strip()
        if len(text) < 50:
            continue

        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)

        with torch.no_grad():
            outputs = gpt2(inputs.input_ids)
            real_acts = gpt2.h[LAYER].mlp(outputs.last_hidden_state)

            reconstruction, codes, _ = model(real_acts)
            mse = torch.nn.functional.mse_loss(reconstruction, real_acts).item()
            mse_scores.append(mse)

            # Collect per-feature statistics
            for token_idx in range(codes.size(1)):
                token_str = tokenizer.decode(inputs.input_ids[0, token_idx])

                for feat_id in range(codes.size(-1)):
                    act_val = codes[0, token_idx, feat_id].item()

                    if act_val > ACTIVATION_THRESHOLD:
                        feature_stats[feat_id]['token_activations'][token_str].append(act_val)
                        feature_stats[feat_id]['activation_count'] += 1

                        if act_val > 2.0 and len(feature_stats[feat_id]['contexts']) < 5:
                            ctx_start = max(0, token_idx - 5)
                            ctx_end = min(inputs.input_ids.size(1), token_idx + 5)
                            context = tokenizer.decode(inputs.input_ids[0, ctx_start:ctx_end])
                            feature_stats[feat_id]['contexts'].append(context)

        if i % 50 == 0:
            active = len([f for f in feature_stats if feature_stats[f]['activation_count'] > 0])
            print(f"  Processed {i} samples | Active features: {active} | Avg MSE: {np.mean(mse_scores):.6f}")

        if i >= n_samples:
            break

    # Compute monosemanticity scores
    mono_scores = {}
    for feat_id, stats in feature_stats.items():
        if stats['activation_count'] < 5:
            continue

        token_acts = stats['token_activations']
        entropy = compute_token_entropy(token_acts)
        consistency = compute_activation_consistency(token_acts)

        if entropy == float('inf') or consistency == float('inf'):
            continue

        mono_scores[feat_id] = {
            'token_entropy': entropy,
            'activation_consistency': consistency,
            'monosemanticity_score': entropy + consistency,
            'activation_count': stats['activation_count'],
            'unique_tokens': len(token_acts),
            'sample_contexts': stats['contexts'][:3],
        }

    return {
        'mse_scores': mse_scores,
        'monosemanticity': mono_scores,
        'n_active_features': len([f for f in feature_stats if feature_stats[f]['activation_count'] > 0]),
        'n_dead_features': N_FEATURES - len([f for f in feature_stats if feature_stats[f]['activation_count'] > 0]),
    }


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

    # Load GPT-2
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    gpt2 = GPT2Model.from_pretrained("gpt2").to(device)
    gpt2.eval()

    # Evaluate both models
    arch_results = evaluate_model(archetypal_sae, "RA-SAE (Archetypal)", gpt2, tokenizer, device, N_EVAL_SAMPLES)
    std_results = evaluate_model(standard_sae, "Standard TopK SAE", gpt2, tokenizer, device, N_EVAL_SAMPLES)

    # Save raw results
    torch.save({
        'archetypal': arch_results['monosemanticity'],
        'standard': std_results['monosemanticity'],
        'archetypal_mse': arch_results['mse_scores'],
        'standard_mse': std_results['mse_scores'],
        'archetypal_active': arch_results['n_active_features'],
        'standard_active': std_results['n_active_features'],
    }, "monosemanticity_results.pt")

    # Print summary
    arch_scores = arch_results['monosemanticity']
    std_scores = std_results['monosemanticity']

    print(f"\n{'=' * 60}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 60}")

    # Reconstruction quality
    print(f"\nReconstruction Quality (MSE, lower is better):")
    print(f"  RA-SAE:      {np.mean(arch_results['mse_scores']):.6f} +/- {np.std(arch_results['mse_scores']):.6f}")
    print(f"  Standard:    {np.mean(std_results['mse_scores']):.6f} +/- {np.std(std_results['mse_scores']):.6f}")

    # Feature utilization
    print(f"\nFeature Utilization:")
    print(f"  RA-SAE:      {arch_results['n_active_features']} active / {arch_results['n_dead_features']} dead")
    print(f"  Standard:    {std_results['n_active_features']} active / {std_results['n_dead_features']} dead")

    # Monosemanticity
    arch_entropies = [s['token_entropy'] for s in arch_scores.values() if s['token_entropy'] < float('inf')]
    std_entropies = [s['token_entropy'] for s in std_scores.values() if s['token_entropy'] < float('inf')]

    arch_consistency = [s['activation_consistency'] for s in arch_scores.values() if s['activation_consistency'] < float('inf')]
    std_consistency = [s['activation_consistency'] for s in std_scores.values() if s['activation_consistency'] < float('inf')]

    if arch_entropies and std_entropies:
        print(f"\nToken Entropy (lower = more monosemantic):")
        print(f"  RA-SAE:      {np.mean(arch_entropies):.3f} +/- {np.std(arch_entropies):.3f}")
        print(f"  Standard:    {np.mean(std_entropies):.3f} +/- {np.std(std_entropies):.3f}")

    if arch_consistency and std_consistency:
        print(f"\nActivation Consistency (lower = more consistent):")
        print(f"  RA-SAE:      {np.mean(arch_consistency):.3f} +/- {np.std(arch_consistency):.3f}")
        print(f"  Standard:    {np.mean(std_consistency):.3f} +/- {np.std(std_consistency):.3f}")

    print(f"\nScored Features:")
    print(f"  RA-SAE:      {len(arch_scores)}")
    print(f"  Standard:    {len(std_scores)}")

    # Top 5 most monosemantic features
    print(f"\n{'=' * 60}")
    print("TOP 5 MOST MONOSEMANTIC FEATURES")
    print(f"{'=' * 60}")

    for name, scores in [("RA-SAE", arch_scores), ("Standard", std_scores)]:
        print(f"\n{name}:")
        sorted_feats = sorted(scores.items(), key=lambda x: x[1]['monosemanticity_score'])[:5]
        for feat_id, metrics in sorted_feats:
            print(f"  Feature {feat_id}: entropy={metrics['token_entropy']:.3f}, "
                  f"consistency={metrics['activation_consistency']:.3f}, "
                  f"tokens={metrics['unique_tokens']}")
            if metrics['sample_contexts']:
                print(f"    Example: {metrics['sample_contexts'][0][:60]}...")

    print(f"\n{'=' * 60}")
    print("Results saved to: monosemanticity_results.pt")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
