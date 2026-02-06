import torch
import numpy as np
from collections import defaultdict
from transformers import GPT2Model, GPT2Tokenizer
from datasets import load_dataset
from archetypal_sae import ArchetypalSAE
from standard_sae import StandardSAE

"""
Compute Monosemanticity Metrics for SAE Features

Monosemanticity = A feature consistently activates for the same semantic concept
We measure this through:
1. Token Entropy: Does the feature fire on diverse tokens (high entropy = polysemantic)
2. Context Diversity: Does it fire in semantically similar contexts?
3. Activation Consistency: Is the activation strength consistent across instances?
"""

def compute_token_entropy(feature_id, token_activations):
    """
    Compute entropy over token distribution for a feature.
    Lower entropy = more monosemantic (fires on fewer token types)
    
    Args:
        feature_id: int
        token_activations: dict mapping token_str -> list of activation values
    
    Returns:
        entropy: float
    """
    if not token_activations:
        return float('inf')
    
    # Count how many times each token type activated this feature
    token_counts = {tok: len(acts) for tok, acts in token_activations.items()}
    total = sum(token_counts.values())
    
    if total == 0:
        return float('inf')
    
    # Compute probability distribution
    probs = np.array([count / total for count in token_counts.values()])
    
    # Shannon entropy: H = -sum(p * log(p))
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    
    return entropy

def compute_activation_consistency(feature_id, token_activations):
    """
    Compute coefficient of variation for activation strengths.
    Lower CoV = more consistent activation pattern
    
    Returns:
        coefficient_of_variation: float (std / mean)
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

def extract_feature_statistics(model, model_type, gpt2, tokenizer, dataset, device, n_samples=500):
    """
    Extract statistics for all features by running model on dataset.
    
    Returns:
        feature_stats: dict with structure:
            {
                feature_id: {
                    'token_activations': {token_str: [act_val1, act_val2, ...]},
                    'max_activation': float,
                    'activation_count': int,
                    'contexts': [str, str, ...]  # Sample contexts
                }
            }
    """
    print(f"\nExtracting statistics for {model_type}...")
    print("=" * 60)
    
    feature_stats = defaultdict(lambda: {
        'token_activations': defaultdict(list),
        'max_activation': 0.0,
        'activation_count': 0,
        'contexts': []
    })
    
    model.eval()
    
    for i, example in enumerate(dataset):
        text = example["text"].strip()
        if len(text) < 50:
            continue
        
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
        
        with torch.no_grad():
            # Get GPT-2 activations
            outputs = gpt2(inputs.input_ids)
            real_acts = gpt2.h[9].mlp(outputs.last_hidden_state)
            
            # Get SAE features
            _, features = model(real_acts)
            
            # Process each token position
            for token_idx in range(features.size(1)):
                token_str = tokenizer.decode(inputs.input_ids[0, token_idx])
                
                # Check each feature
                for feat_id in range(features.size(-1)):
                    act_val = features[0, token_idx, feat_id].item()
                    
                    # Only record if feature is active (threshold > 0.1)
                    if act_val > 0.1:
                        feature_stats[feat_id]['token_activations'][token_str].append(act_val)
                        feature_stats[feat_id]['activation_count'] += 1
                        
                        if act_val > feature_stats[feat_id]['max_activation']:
                            feature_stats[feat_id]['max_activation'] = act_val
                        
                        # Store context for top activations (sample)
                        if act_val > 2.0 and len(feature_stats[feat_id]['contexts']) < 5:
                            context_start = max(0, token_idx - 5)
                            context_end = min(inputs.input_ids.size(1), token_idx + 5)
                            context = tokenizer.decode(inputs.input_ids[0, context_start:context_end])
                            feature_stats[feat_id]['contexts'].append(context)
        
        if i % 50 == 0:
            active_features = len([f for f in feature_stats if feature_stats[f]['activation_count'] > 0])
            print(f"Processed {i} samples | Active features: {active_features}")
        
        if i >= n_samples:
            break
    
    return dict(feature_stats)

def compute_monosemanticity_scores(feature_stats):
    """
    Compute monosemanticity metrics for each feature.
    
    Returns:
        results: dict with structure:
            {
                feature_id: {
                    'token_entropy': float,
                    'activation_consistency': float,
                    'monosemanticity_score': float,  # Combined metric
                    'activation_count': int,
                    'unique_tokens': int
                }
            }
    """
    results = {}
    
    for feat_id, stats in feature_stats.items():
        if stats['activation_count'] < 5:  # Skip rarely-firing features
            continue
        
        token_acts = stats['token_activations']
        
        # Compute metrics
        entropy = compute_token_entropy(feat_id, token_acts)
        consistency = compute_activation_consistency(feat_id, token_acts)
        
        # Combined monosemanticity score (lower is better)
        # Normalize and combine: features with low entropy and low CoV are monosemantic
        mono_score = entropy + consistency  # Simple additive (you can weight these)
        
        results[feat_id] = {
            'token_entropy': entropy,
            'activation_consistency': consistency,
            'monosemanticity_score': mono_score,
            'activation_count': stats['activation_count'],
            'unique_tokens': len(token_acts),
            'sample_contexts': stats['contexts'][:3]  # Top 3
        }
    
    return results

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    
    print("=" * 60)
    print("MONOSEMANTICITY COMPARISON")
    print("Archetypal SAE vs Standard SAE")
    print("=" * 60)
    
    # Load models
    print("\nLoading models...")
    anchor_points = torch.load("anchor_points.pt").to(device)
    
    archetypal_sae = ArchetypalSAE(d_model=768, n_features=4096, anchor_points=anchor_points).to(device)
    archetypal_sae.load_state_dict(torch.load("archetypal_sae_weights_v2.pt"))
    archetypal_sae.eval()
    
    standard_sae = StandardSAE(d_model=768, n_features=4096).to(device)
    standard_sae.load_state_dict(torch.load("standard_sae_weights.pt"))
    standard_sae.eval()
    
    # Load GPT-2 and dataset
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    gpt2 = GPT2Model.from_pretrained("gpt2").to(device)
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", streaming=True)
    
    # Extract statistics for both models
    arch_stats = extract_feature_statistics(
        archetypal_sae, "Archetypal SAE", gpt2, tokenizer, 
        load_dataset("wikitext", "wikitext-2-raw-v1", split="test", streaming=True),
        device, n_samples=500
    )
    
    std_stats = extract_feature_statistics(
        standard_sae, "Standard SAE", gpt2, tokenizer,
        load_dataset("wikitext", "wikitext-2-raw-v1", split="test", streaming=True),
        device, n_samples=500
    )
    
    # Compute monosemanticity scores
    print("\nComputing monosemanticity metrics...")
    arch_scores = compute_monosemanticity_scores(arch_stats)
    std_scores = compute_monosemanticity_scores(std_stats)
    
    # Save raw results
    torch.save({
        'archetypal': arch_scores,
        'standard': std_scores
    }, "monosemanticity_results.pt")
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    # Aggregate statistics
    arch_entropies = [s['token_entropy'] for s in arch_scores.values() if s['token_entropy'] < float('inf')]
    std_entropies = [s['token_entropy'] for s in std_scores.values() if s['token_entropy'] < float('inf')]
    
    arch_consistency = [s['activation_consistency'] for s in arch_scores.values() if s['activation_consistency'] < float('inf')]
    std_consistency = [s['activation_consistency'] for s in std_scores.values() if s['activation_consistency'] < float('inf')]
    
    print(f"\nToken Entropy (Lower = More Monosemantic):")
    print(f"  Archetypal SAE: {np.mean(arch_entropies):.3f} ± {np.std(arch_entropies):.3f}")
    print(f"  Standard SAE:   {np.mean(std_entropies):.3f} ± {np.std(std_entropies):.3f}")
    
    print(f"\nActivation Consistency (Lower = More Consistent):")
    print(f"  Archetypal SAE: {np.mean(arch_consistency):.3f} ± {np.std(arch_consistency):.3f}")
    print(f"  Standard SAE:   {np.mean(std_consistency):.3f} ± {np.std(std_consistency):.3f}")
    
    print(f"\nActive Features:")
    print(f"  Archetypal SAE: {len(arch_scores)}")
    print(f"  Standard SAE:   {len(std_scores)}")
    
    # Show examples of most monosemantic features from each
    print("\n" + "=" * 60)
    print("TOP 5 MOST MONOSEMANTIC FEATURES")
    print("=" * 60)
    
    print("\nArchetypal SAE:")
    arch_sorted = sorted(arch_scores.items(), key=lambda x: x[1]['monosemanticity_score'])[:5]
    for feat_id, metrics in arch_sorted:
        print(f"\n  Feature {feat_id}:")
        print(f"    Token Entropy: {metrics['token_entropy']:.3f}")
        print(f"    Consistency: {metrics['activation_consistency']:.3f}")
        print(f"    Fires on {metrics['unique_tokens']} unique tokens")
        if metrics['sample_contexts']:
            print(f"    Example: {metrics['sample_contexts'][0][:60]}...")
    
    print("\nStandard SAE:")
    std_sorted = sorted(std_scores.items(), key=lambda x: x[1]['monosemanticity_score'])[:5]
    for feat_id, metrics in std_sorted:
        print(f"\n  Feature {feat_id}:")
        print(f"    Token Entropy: {metrics['token_entropy']:.3f}")
        print(f"    Consistency: {metrics['activation_consistency']:.3f}")
        print(f"    Fires on {metrics['unique_tokens']} unique tokens")
        if metrics['sample_contexts']:
            print(f"    Example: {metrics['sample_contexts'][0][:60]}...")
    
    print("\n" + "=" * 60)
    print("Results saved to: monosemanticity_results.pt")
    print("=" * 60)

if __name__ == "__main__":
    main()