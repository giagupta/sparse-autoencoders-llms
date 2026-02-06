import torch
from transformers import GPT2Model, GPT2Tokenizer
from archetypal_sae import ArchetypalSAE
from datasets import load_dataset

# 1. Configuration
FEATURE_TO_INSPECT = 111  # Pick a 'directed' feature from your list!
LAYER = 9
N_FEATURES = 4096
device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.backends.mps.is_available(): device = "mps"

# 2. Load Model & Data
anchor_points = torch.load("anchor_points.pt").to(device)
model = ArchetypalSAE(d_model=768, n_features=N_FEATURES, anchor_points=anchor_points).to(device)
model.load_state_dict(torch.load("archetypal_sae_weights_v2.pt"))
model.eval()

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2 = GPT2Model.from_pretrained("gpt2").to(device)
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", streaming=True)

print(f"--- Investigating Feature {FEATURE_TO_INSPECT} ---")
results = []

# 3. Scan for top activations
for i, example in enumerate(dataset):
    text = example["text"].strip()
    if len(text) < 50: continue
    
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = gpt2(inputs.input_ids)
        real_acts = gpt2.h[LAYER].mlp(outputs.last_hidden_state)
        _, features = model(real_acts)
    
    # Check if our target feature fired in this sentence
    feat_acts = features[0, :, FEATURE_TO_INSPECT]
    if torch.max(feat_acts) > 0.5:
        max_val = torch.max(feat_acts).item()
        max_idx = torch.argmax(feat_acts).item()
        token = tokenizer.decode(inputs.input_ids[0, max_idx])
        
        # Save context for display
        context_start = max(0, max_idx - 5)
        context_end = min(inputs.input_ids.size(1), max_idx + 5)
        context = tokenizer.decode(inputs.input_ids[0, context_start:context_end])
        
        results.append({"val": max_val, "token": token, "context": context})

    if len(results) >= 10 or i > 500: break

# 4. Show the "Persona" of the feature
results = sorted(results, key=lambda x: x["val"], reverse=True)
for r in results:
    print(f"Activation: {r['val']:.2f} | Token: '{r['token']}' | Context: [...]{r['context']}[...]")
    