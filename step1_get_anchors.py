"""
Step 1: Extract activation centroids from GPT-2 using K-means clustering.

The paper uses K-means clustering on model activations to produce centroids
that serve as the data matrix C for the Archetypal SAE dictionary construction.

Reference: Fel et al. (2025), "Archetypal SAE", ICML 2025.
  "applied K-Means clustering to the entire dataset, reducing it to centroids"
"""

import torch
import numpy as np
from transformers import GPT2Model, GPT2Tokenizer
from datasets import load_dataset
from sklearn.cluster import MiniBatchKMeans

# --- Configuration ---
LAYER = 9              # GPT-2 layer to extract activations from
N_CENTROIDS = 4096     # Number of K-means centroids (paper uses up to 32k for vision;
                       # 4096 is practical for GPT-2 language experiments)
N_SAMPLES = 2000       # Number of text snippets to collect activations from
MAX_SEQ_LEN = 128      # Maximum token sequence length
BATCH_SIZE = 256       # MiniBatchKMeans batch size

print("=" * 60)
print("STEP 1: Extract K-means centroids from GPT-2 activations")
print("=" * 60)

# 1. Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.backends.mps.is_available():
    device = "mps"

model_name = "gpt2"
print(f"Loading {model_name} on {device}...")
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2Model.from_pretrained(model_name).to(device)
model.eval()

# 2. Load dataset
print("Loading WikiText-2 (streaming mode)...")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", streaming=True)

# 3. Collect activations
print(f"Extracting Layer {LAYER} MLP activations from {N_SAMPLES} text snippets...")
all_activations = []

count = 0
for example in dataset:
    text = example["text"].strip()
    if len(text) < 50:
        continue

    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN
    ).to(device)

    with torch.no_grad():
        outputs = model(inputs.input_ids, output_hidden_states=True)
        # Extract residual stream after Layer 9
        acts = outputs.hidden_states[LAYER + 1]
        # Flatten batch and sequence dims, move to CPU
        acts = acts.view(-1, acts.shape[-1]).cpu().numpy()
        all_activations.append(acts)

    count += 1
    if count % 100 == 0:
        n_vecs = sum(a.shape[0] for a in all_activations)
        print(f"  Collected from {count} snippets ({n_vecs} vectors)...")

    if count >= N_SAMPLES:
        break

# Combine all activations
print("Combining activations...")
all_activations = np.concatenate(all_activations, axis=0)
print(f"Total activation vectors: {all_activations.shape[0]} x {all_activations.shape[1]}")

# 4. Run K-means clustering
print(f"\nRunning MiniBatchKMeans with {N_CENTROIDS} centroids...")
kmeans = MiniBatchKMeans(
    n_clusters=N_CENTROIDS,
    batch_size=BATCH_SIZE,
    n_init=3,
    max_iter=300,
    random_state=42,
    verbose=1,
)
kmeans.fit(all_activations)

# 5. Save centroids
centroids = torch.from_numpy(kmeans.cluster_centers_).float()
torch.save(centroids, "anchor_points.pt")

print(f"\n{'=' * 60}")
print(f"SUCCESS!")
print(f"Saved: anchor_points.pt")
print(f"Shape: {centroids.shape}  (n_centroids={N_CENTROIDS}, d_model={centroids.shape[1]})")
print(f"K-means inertia: {kmeans.inertia_:.2f}")
print(f"{'=' * 60}")
