"""
Step 3: Train the Relaxed Archetypal SAE (RA-SAE).

Uses MSE reconstruction loss only — TopK handles sparsity directly.
The archetypal constraint (D = W@C + Lambda) provides the regularization
that is the core contribution of the paper.

An auxiliary loss encourages dead features to revive.

Reference: Fel et al. (2025), "Archetypal SAE", ICML 2025.
"""

import torch
import torch.optim as optim
import torch.nn.functional as F
from datasets import load_dataset
from transformers import GPT2Model, GPT2Tokenizer
from archetypal_sae import ArchetypalSAE

# --- Configuration ---
LAYER = 9
N_FEATURES = 4096
TOP_K = 64                # Match the baseline
DELTA = 1.0               # Relaxation parameter for Lambda constraint
LR = 3e-4
TRAINING_STEPS = 5000
MAX_SEQ_LEN = 128
LOG_EVERY = 100
AUX_LOSS_COEFF = 1/32     # Auxiliary loss to reduce dead features

# 1. Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.backends.mps.is_available():
    device = "mps"

print("=" * 60)
print("STEP 3: Train Relaxed Archetypal SAE (RA-SAE)")
print("=" * 60)

# Load centroids (from step 1)
anchor_points = torch.load("anchor_points.pt", weights_only=True).to(device)
print(f"Loaded centroids: {anchor_points.shape}")

# Initialize model
model = ArchetypalSAE(
    d_model=768,
    n_features=N_FEATURES,
    anchor_points=anchor_points,
    top_k=TOP_K,
    delta=DELTA,
    use_multiplier=True,
).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)

# Load GPT-2 and dataset
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2 = GPT2Model.from_pretrained("gpt2").to(device)
gpt2.eval()
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", streaming=True)

print(f"Device: {device}")
print(f"Layer: {LAYER} | Features: {N_FEATURES} | TopK: {TOP_K} | Delta: {DELTA}")
print(f"Learning rate: {LR} | Steps: {TRAINING_STEPS}")
print("=" * 60)

# Track dead features
feature_activity = torch.zeros(N_FEATURES, device=device)

# 2. Training Loop
step = 0
for example in dataset:
    text = example["text"].strip()
    if len(text) < 100:
        continue

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN).to(device)

    # Get activations from GPT-2 Layer 9
    with torch.no_grad():
        outputs = gpt2(inputs.input_ids)
        real_acts = gpt2.h[LAYER].mlp(outputs.last_hidden_state)

    # Forward pass through RA-SAE
    reconstruction, codes, pre_codes = model(real_acts)

    # Main loss: MSE reconstruction
    mse_loss = F.mse_loss(reconstruction, real_acts)

    # Auxiliary loss: encourage dead features to fire
    with torch.no_grad():
        feature_activity = 0.999 * feature_activity + 0.001 * (codes.abs().sum(dim=(0, 1)) > 0).float()

    dead_mask = (feature_activity < 0.01).float()
    if dead_mask.sum() > 0:
        dead_pre_codes = pre_codes * dead_mask.unsqueeze(0).unsqueeze(0)
        aux_loss = AUX_LOSS_COEFF * F.relu(-dead_pre_codes).mean()
    else:
        aux_loss = torch.tensor(0.0, device=device)

    total_loss = mse_loss + aux_loss

    # Optimization step
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    # Logging
    if step % LOG_EVERY == 0:
        l0 = (codes > 0).float().sum(-1).mean().item()
        n_dead = (feature_activity < 0.01).sum().item()
        mult = model.dictionary.multiplier.item()
        print(
            f"Step {step:5d} | MSE: {mse_loss.item():.6f} | "
            f"L0: {l0:.1f} | Dead: {n_dead} | Mult: {mult:.3f} | Aux: {aux_loss.item():.6f}"
        )

    step += 1
    if step >= TRAINING_STEPS:
        break

# 3. Save
torch.save(model.state_dict(), "archetypal_sae_weights_v2.pt")
print(f"\n{'=' * 60}")
print("Training Complete!")
print(f"Saved: archetypal_sae_weights_v2.pt")
print(f"{'=' * 60}")
