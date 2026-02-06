"""
Step 2: Train a Standard TopK SAE (baseline, no archetypal constraints).

Uses MSE reconstruction loss only — TopK handles sparsity directly.
An optional auxiliary loss encourages dead features to revive.

This serves as the unconstrained baseline for comparison against RA-SAE.
"""

import torch
import torch.optim as optim
import torch.nn.functional as F
from datasets import load_dataset
from transformers import GPT2Model, GPT2Tokenizer
from standard_sae import StandardSAE

# --- Configuration ---
LAYER = 9
N_FEATURES = 4096
TOP_K = 64                # ~1.5% sparsity (64 out of 4096)
LR = 3e-4
TRAINING_STEPS = 5000
MAX_SEQ_LEN = 128
LOG_EVERY = 100
AUX_LOSS_COEFF = 1/32     # Auxiliary loss to reduce dead features (from Gao et al. 2024)

# 1. Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.backends.mps.is_available():
    device = "mps"

print("=" * 60)
print("STEP 2: Train Standard TopK SAE (Baseline)")
print("=" * 60)

# Initialize model
model = StandardSAE(d_model=768, n_features=N_FEATURES, top_k=TOP_K).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)

# Load GPT-2 and dataset
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2 = GPT2Model.from_pretrained("gpt2").to(device)
gpt2.eval()
# NOTE:
# `streaming=True` can fail on some Python/torch environments due to shared
# memory manager restrictions. Eager loading is more robust for WikiText-2.
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", streaming=False)

print(f"Device: {device}")
print(f"Layer: {LAYER} | Features: {N_FEATURES} | TopK: {TOP_K}")
print(f"Learning rate: {LR} | Steps: {TRAINING_STEPS}")
print("=" * 60)

# Track dead features
feature_activity = torch.zeros(N_FEATURES, device=device)

# 2. Training Loop
step = 0
dataset_iter = iter(dataset)
while step < TRAINING_STEPS:
    try:
        example = next(dataset_iter)
    except StopIteration:
        dataset_iter = iter(dataset)
        continue
    text = example["text"].strip()
    if len(text) < 100:
        continue

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN).to(device)

    # Get activations from GPT-2 Layer 9
    with torch.no_grad():
        outputs = gpt2(inputs.input_ids)
        real_acts = gpt2.h[LAYER].mlp(outputs.last_hidden_state)

    # Forward pass through SAE
    reconstruction, codes, pre_codes = model(real_acts)

    # Main loss: MSE reconstruction
    mse_loss = F.mse_loss(reconstruction, real_acts)

    # Auxiliary loss: encourage dead features to fire (from Gao et al. 2024)
    # Dead features are those that haven't fired recently
    with torch.no_grad():
        feature_activity = 0.999 * feature_activity + 0.001 * (codes.abs().sum(dim=(0, 1)) > 0).float()

    dead_mask = (feature_activity < 0.01).float()
    if dead_mask.sum() > 0:
        # Push pre-codes of dead features toward positive values
        dead_pre_codes = pre_codes * dead_mask.unsqueeze(0).unsqueeze(0)
        aux_loss = AUX_LOSS_COEFF * F.relu(-dead_pre_codes).mean()
    else:
        aux_loss = torch.tensor(0.0, device=device)

    total_loss = mse_loss + aux_loss

    # Optimization step
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    # Normalize decoder dictionary rows to unit norm (standard practice)
    with torch.no_grad():
        model.dictionary.data = F.normalize(model.dictionary.data, dim=-1)

    # Logging
    if step % LOG_EVERY == 0:
        l0 = (codes > 0).float().sum(-1).mean().item()
        n_dead = (feature_activity < 0.01).sum().item()
        print(
            f"Step {step:5d} | MSE: {mse_loss.item():.6f} | "
            f"L0: {l0:.1f} | Dead: {n_dead} | Aux: {aux_loss.item():.6f}"
        )

    step += 1

# 3. Save
torch.save(model.state_dict(), "standard_sae_weights.pt")
print(f"\n{'=' * 60}")
print("Training Complete!")
print(f"Saved: standard_sae_weights.pt")
print(f"{'=' * 60}")
