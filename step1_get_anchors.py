print("--- SCRIPT STARTING ---")
import torch
from transformers import GPT2Model, GPT2Tokenizer
from datasets import load_dataset

# 1. Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "gpt2"
print(f"Loading {model_name} on {device}...")
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2Model.from_pretrained(model_name).to(device)

# 2. Use WikiText (much smaller and safer for your disk space)
print("Loading Wikitext (Streaming mode)...")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", streaming=True)

# 3. Extraction Logic
activations = []

def hook_fn(module, input, output):
    # Output of Layer 9 MLP
    # We move it to CPU immediately to save GPU memory
    activations.append(output.detach().cpu())

handle = model.h[9].mlp.register_forward_hook(hook_fn)

print("Extracting activations from text...")
count = 0
for example in dataset:
    text = example["text"].strip()
    if len(text) < 10: # Skip empty lines
        continue
        
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
    
    with torch.no_grad():
        model(**inputs)
    
    count += 1
    if count % 10 == 0:
        print(f"Collected from {count} text snippets...")
    
    if count >= 100: # This is plenty for 10,000 anchors
        break

handle.remove()

# 4. Finalizing
print("Creating the anchor file...")
# Combine all collected activations
all_acts = torch.cat(activations, dim=1).view(-1, 768)

# Check if we have enough
if all_acts.size(0) < 10000:
    print(f"Warning: Only found {all_acts.size(0)} vectors. Using all of them.")
    anchor_points = all_acts
else:
    indices = torch.randperm(all_acts.size(0))[:10000]
    anchor_points = all_acts[indices]

torch.save(anchor_points, "anchor_points.pt")
print(f"--- SUCCESS! ---")
print(f"File saved: anchor_points.pt (Size: {anchor_points.shape})")