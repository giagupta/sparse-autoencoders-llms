import torch
import torch.nn as nn

class StandardSAE(nn.Module):
    """
    Standard Sparse Autoencoder (no archetypal constraints)
    Uses the typical encoder-decoder architecture with L1 sparsity penalty
    """
    def __init__(self, d_model=768, n_features=4096):
        super().__init__()
        self.d_model = d_model
        self.n_features = n_features
        
        # Encoder: projects from model dimension to feature space
        self.encoder = nn.Linear(d_model, n_features, bias=True)
        
        # Decoder: projects back from feature space to model dimension
        self.decoder = nn.Linear(n_features, d_model, bias=True)
        
        # Initialize decoder as transpose of encoder (tied weights is common)
        # But we'll keep them separate for flexibility
        self._init_weights()
    
    def _init_weights(self):
        # Initialize with small random values
        nn.init.kaiming_uniform_(self.encoder.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.decoder.weight, nonlinearity='relu')
        nn.init.zeros_(self.encoder.bias)
        nn.init.zeros_(self.decoder.bias)
        
        # Normalize decoder columns to unit norm (common practice)
        with torch.no_grad():
            self.decoder.weight.data = nn.functional.normalize(
                self.decoder.weight.data, dim=0
            )
    
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, d_model] - activations from transformer
        
        Returns:
            reconstruction: [batch, seq_len, d_model]
            features: [batch, seq_len, n_features] - sparse feature activations
        """
        # Encode to feature space with ReLU for sparsity
        features = torch.relu(self.encoder(x))
        
        # Decode back to original space
        reconstruction = self.decoder(features)
        
        return reconstruction, features
    
    def get_feature_vectors(self):
        """
        Returns the decoder weight matrix (each column is a feature vector)
        """
        return self.decoder.weight.T  # [n_features, d_model]