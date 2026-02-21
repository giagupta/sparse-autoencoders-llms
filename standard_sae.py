"""
Standard TopK Sparse Autoencoder (baseline, no archetypal constraints).

Uses TopK activation for sparsity (matching the RA-SAE setup for fair comparison).
This is the unconstrained baseline from the Archetypal SAE paper.

Reference:
  "Scaling and evaluating sparse autoencoders" by Gao et al., 2024.
  https://arxiv.org/abs/2406.04093
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StandardSAE(nn.Module):
    """
    Standard TopK SAE without archetypal constraints.

    Architecture:
      Encode:  pre_codes = W_enc @ x + b_enc
               codes = TopK(pre_codes)
      Decode:  x_hat = codes @ D
               where D is a learned dictionary with unit-norm columns.

    Parameters
    ----------
    d_model : int
        Input dimension (e.g., 768 for GPT-2).
    n_features : int
        Number of dictionary atoms / latent features.
    top_k : int
        Number of top activations to keep. Default: n_features // 10.
    """

    def __init__(self, d_model=768, n_features=4096, top_k=None):
        super().__init__()
        self.d_model = d_model
        self.n_features = n_features
        self.top_k = top_k if top_k is not None else max(n_features // 10, 1)

        # Encoder: projects from model dimension to feature space
        self.encoder = nn.Linear(d_model, n_features, bias=True)

        # Decoder dictionary: (n_features, d_model) — no bias in decoder
        self.dictionary = nn.Parameter(torch.empty(n_features, d_model))

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.encoder.weight, nonlinearity='relu')
        nn.init.zeros_(self.encoder.bias)
        nn.init.kaiming_uniform_(self.dictionary)
        # Normalize decoder rows to unit norm
        with torch.no_grad():
            self.dictionary.data = F.normalize(self.dictionary.data, dim=-1)

    def encode(self, x):
        """
        Encode input to sparse latent codes using TopK.

        Returns
        -------
        pre_codes : Tensor
            Pre-activation values (before TopK).
        codes : Tensor
            Sparse codes after TopK.
        """
        pre_codes = self.encoder(x)

        # TopK is the sole activation function (no ReLU), matching the
        # canonical TopK SAE from Gao et al. (2024).  All top_k features
        # always receive gradients, preventing feature collapse.
        topk_vals, topk_indices = torch.topk(pre_codes, self.top_k, dim=-1)
        codes = torch.zeros_like(pre_codes).scatter(
            -1, topk_indices, topk_vals
        )

        return pre_codes, codes

    def decode(self, codes):
        """Decode sparse codes back to input space: x_hat = codes @ D."""
        return codes @ self.dictionary

    def forward(self, x):
        """
        Full forward pass.

        Returns
        -------
        reconstruction : Tensor
            Reconstructed input.
        codes : Tensor
            Sparse feature activations.
        pre_codes : Tensor
            Pre-activation values (useful for auxiliary losses).
        """
        pre_codes, codes = self.encode(x)
        reconstruction = self.decode(codes)
        return reconstruction, codes, pre_codes

    def get_dictionary(self):
        """Return the learned dictionary matrix (n_features, d_model)."""
        return self.dictionary.data
