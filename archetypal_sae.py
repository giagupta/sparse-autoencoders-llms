import torch
import torch.nn as nn
import torch.nn.functional as F

class ArchetypalSAE(nn.Module):
    def __init__(self, d_model, n_features, anchor_points):
        super().__init__()
        # 1. The Anchor Points (The 'snapshots' you just created)
        # We wrap it in a 'Buffer' so it stays with the model but doesn't 'learn'
        self.register_buffer("X_anchor", anchor_points) 
        n_anchors = anchor_points.shape[0]

        # 2. The Encoder: This converts GPT-2 activations into 'features'
        self.encoder = nn.Linear(d_model, n_features)
        self.encoder_bias = nn.Parameter(torch.zeros(n_features))

        # 3. The Archetypal Coefficients (C): 
        # This is the secret sauce. Instead of learning the dictionary directly,
        # we learn how to weight the anchors to CREATE the dictionary.
        self.C = nn.Parameter(torch.randn(n_features, n_anchors) * 0.01)

    def get_dictionary(self):
        # This turns our raw coefficients into a 'Convex Hull'
        # Softmax ensures every row sums to 1 and all values are positive.
        weights = F.softmax(self.C, dim=-1)
        # D = Weights * Anchors
        return torch.matmul(weights, self.X_anchor)

    def forward(self, x):
        # x is the input from GPT-2
        
        # Step A: Encode (Find the features)
        # We use ReLU to make sure features are either 'on' (positive) or 'off' (zero)
        features = F.relu(self.encoder(x) + self.encoder_bias)

        # Step B: Decode (Reconstruct the original input)
        # We get our 'Archetypal' dictionary atoms
        D = self.get_dictionary()
        
        # Reconstruct by multiplying features by the dictionary
        reconstruction = torch.matmul(features, D)
        
        return reconstruction, features

print("Model architecture defined!")