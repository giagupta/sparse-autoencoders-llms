import torch
import torch.optim as optim
from datasets import load_dataset
from transformers import GPT2Model, GPT2Tokenizer
from standard_sae import StandardSAE

# 1. Setup - Match your archetypal SAE training exactly
device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.backends.mps.is_available():
    device = "mps"

LAYER = 9 
N_FEATURES = 4096
L1_COEFF = 0.001

print(f"Training Standard SAE on {device}...")
print("=" * 60)

# Initialize standard SAE (no anchors needed)
model = StandardSAE(d_model=768, n_features=N_FEATURES).to(device)
optimizer = optim.Adam(model.parameters(), lr=5e-4)

# Load GPT-2 and dataset (same as before)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2 = GPT2Model.from_pretrained("gpt2").to(device)
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", streaming=True)

print(f"Layer: {LAYER} | Features: {N_FEATURES} | L1 Coefficient: {L1_COEFF}")
print("=" * 60)

# 2. Training Loop (identical to archetypal version)
for i, example in enumerate(dataset):
    text = example["text"].strip()
    if len(text) < 100:
        continue
        
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
    
    # Get activations from GPT-2 Layer 9
    with torch.no_grad():
        outputs = gpt2(inputs.input_ids)
        real_acts = gpt2.h[LAYER].mlp(outputs.last_hidden_state)

    # Forward pass through SAE
    reconstruction, features = model(real_acts)

    # Compute losses
    mse_loss = torch.nn.functional.mse_loss(reconstruction, real_acts)
    l1_loss = L1_COEFF * features.abs().sum()
    total_loss = mse_loss + l1_loss

    # Optimization step
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    # Logging
    if i % 20 == 0:
        l0 = (features > 0).float().sum(-1).mean().item()
        print(f"Step {i:4d} | MSE: {mse_loss.item():8.4f} | L0 (Sparsity): {l0:4.1f} | L1: {l1_loss.item():6.4f}")

    if i >= 500:  # Same training duration as archetypal version
        break

# 3. Save the trained model
torch.save(model.state_dict(), "standard_sae_weights.pt")
print("\n" + "=" * 60)
print("Training Complete!")
print(f"Saved: standard_sae_weights.pt")
print("=" * 60)