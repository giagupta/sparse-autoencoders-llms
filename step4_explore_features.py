import torch
from transformers import GPT2Model, GPT2Tokenizer
from archetypal_sae import ArchetypalSAE
from datasets import load_dataset

# 1. Setup - USE THE NEW WEIGHTS
device = "cuda" if torch.cuda.is_available() else "cpu"
anchor_points = torch.load("anchor_points.pt").to(device)
# Make sure N_FEATURES matches your recent training (4096)
model = ArchetypalSAE(d_model=768, n_features=4096, anchor_points=anchor_points).to(device)
model.load_state_dict(torch.load("archetypal_sae_weights_v2.pt"))
model.eval()

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2 = GPT2Model.from_pretrained("gpt2").to(device)
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", streaming=True)

print("Scanning Layer 9 for stable concepts...")
active_features = {} 

for i, example in enumerate(dataset):
    text = example["text"].strip()
    if len(text) < 100: continue
    
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = gpt2(inputs.input_ids)
        # Match the layer you just trained (Layer 9)
        real_acts = gpt2.h[9].mlp(outputs.last_hidden_state)
        _, features = model(real_acts)
    
    # Find features that are firing strongly (> 2.0)
    fired_mask = features[0] > 2.0 
    fired_indices = torch.nonzero(fired_mask)

    for token_idx, feat_id in fired_indices:
        f_id = feat_id.item()
        if f_id not in active_features:
            token = tokenizer.decode(inputs.input_ids[0, token_idx])
            active_features[f_id] = (token, text[:70])
            
    if len(active_features) > 15: break
    if i > 200: break

print("\n--- NEW ACTIVE CONCEPTS (LAYER 9) ---")
for f_id, data in active_features.items():
    print(f"Feature {f_id:4} | Token: '{data[0]:10}' | Context: {data[1]}...")