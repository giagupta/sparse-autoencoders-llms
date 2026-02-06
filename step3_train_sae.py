import torch
import torch.optim as optim
from datasets import load_dataset
from transformers import GPT2Model, GPT2Tokenizer
from archetypal_sae import ArchetypalSAE 

# 1. Setup - Moving to Layer 9 for better concepts
device = "cuda" if torch.cuda.is_available() else "cpu"
LAYER = 9 
N_FEATURES = 4096 # Higher resolution
L1_COEFF = 0.001  # Lower penalty = more complex features

anchor_points = torch.load("anchor_points.pt").to(device)
model = ArchetypalSAE(d_model=768, n_features=N_FEATURES, anchor_points=anchor_points).to(device)
optimizer = optim.Adam(model.parameters(), lr=5e-4) # Slightly lower LR for stability

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2 = GPT2Model.from_pretrained("gpt2").to(device)
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", streaming=True)

print(f"--- Training High-Res SAE on Layer {LAYER} ---")

for i, example in enumerate(dataset):
    text = example["text"].strip()
    if len(text) < 100: continue
        
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
    
    with torch.no_grad():
        outputs = gpt2(inputs.input_ids)
        # We are now looking at Layer 9
        real_acts = gpt2.h[LAYER].mlp(outputs.last_hidden_state)

    reconstruction, features = model(real_acts)

    mse_loss = torch.nn.functional.mse_loss(reconstruction, real_acts)
    l1_loss = L1_COEFF * features.abs().sum()
    total_loss = mse_loss + l1_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if i % 20 == 0:
        l0 = (features > 0).float().sum(-1).mean().item()
        print(f"Step {i} | MSE: {mse_loss.item():.4f} | L0 (Sparsity): {l0:.1f}")

    if i >= 500: # Give it more time to learn!
        break

torch.save(model.state_dict(), "archetypal_sae_weights_v2.pt")
print("--- Training Complete! ---")