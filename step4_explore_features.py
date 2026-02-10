"""
Step 4: Explore features from the trained RA-SAE.
Scans dataset for strongly activated features and shows what they respond to.
"""

import torch
from transformers import GPT2Model, GPT2Tokenizer
from archetypal_sae import ArchetypalSAE
from datasets import load_dataset

# Configuration
LAYER = 9
N_FEATURES = 4096
TOP_K = 64
device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.backends.mps.is_available():
    device = "mps"

# Load model
anchor_points = torch.load("anchor_points.pt", weights_only=True).to(device)
model = ArchetypalSAE(d_model=768, n_features=N_FEATURES, anchor_points=anchor_points, top_k=TOP_K).to(device)
model.load_state_dict(torch.load("archetypal_sae_weights_v2.pt", weights_only=True))
model.eval()

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2 = GPT2Model.from_pretrained("gpt2").to(device)
gpt2.eval()
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", streaming=False)

print("Scanning Layer 9 for active concepts...")
active_features = {}

for i, example in enumerate(dataset):
    text = example["text"].strip()
    if len(text) < 100:
        continue

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = gpt2(inputs.input_ids, output_hidden_states=True)
        real_acts = outputs.hidden_states[LAYER + 1]
        _, codes, _ = model(real_acts)

    # Find features that are firing strongly (> 2.0)
    fired_mask = codes[0] > 2.0
    fired_indices = torch.nonzero(fired_mask)

    for token_idx, feat_id in fired_indices:
        f_id = feat_id.item()
        if f_id not in active_features:
            token = tokenizer.decode(inputs.input_ids[0, token_idx])
            active_features[f_id] = (token, text[:70])

    if len(active_features) > 15:
        break
    if i > 200:
        break

print(f"\n--- ACTIVE CONCEPTS (LAYER {LAYER}, RA-SAE) ---")
for f_id, data in active_features.items():
    print(f"Feature {f_id:4} | Token: '{data[0]:10}' | Context: {data[1]}...")
