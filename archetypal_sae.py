"""
Relaxed Archetypal SAE (RA-SAE) with TopK activation.

Implements the method from:
  "Archetypal SAE: Adaptive and Stable Dictionary Learning for Concept
   Extraction in Large Vision Models" by Fel et al., ICML 2025.
   https://arxiv.org/abs/2502.12892

Adapted for the language model regime (GPT-2 activations).

Key ideas:
  - Dictionary atoms D are constrained to the convex hull of data centroids C,
    plus a small learned relaxation Lambda:  D = W @ C + Lambda
  - W is row-stochastic (non-negative, rows sum to 1), enforced via ReLU + normalize
  - Lambda is per-row norm-constrained: ||Lambda_i||_2 <= delta
  - A learnable multiplier scales the dictionary: D_final = D * exp(multiplier)
  - TopK activation enforces sparsity (no L1 penalty needed)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RelaxedArchetypalDictionary(nn.Module):
    """
    Dictionary layer where atoms are convex combinations of data centroids
    plus a small learned relaxation.

    D = W @ C + Lambda,  scaled by exp(multiplier)

    W: row-stochastic (ReLU + row-normalize)
    Lambda: ||Lambda_i||_2 <= delta per row
    multiplier: learnable scalar
    """

    def __init__(self, in_dimensions, nb_concepts, points, delta=1.0,
                 use_multiplier=True, device="cpu"):
        super().__init__()
        self.in_dimensions = in_dimensions
        self.nb_concepts = nb_concepts
        self.device = device
        self.delta = delta

        # C: the data centroids (frozen buffer, not trained)
        self.register_buffer("C", points)  # (nb_candidates, in_dimensions)
        self.nb_candidates = self.C.shape[0]

        # W: row-stochastic weight matrix, initialized as identity-like
        self.W = nn.Parameter(
            torch.eye(nb_concepts, self.nb_candidates, device=device)
        )

        # Lambda: relaxation matrix, initialized at zero
        self.Relax = nn.Parameter(
            torch.zeros(nb_concepts, self.in_dimensions, device=device)
        )

        # Learnable multiplier for dictionary scaling
        # Initialize to compensate for centroid norms so initial dictionary
        # has approximately unit-norm atoms (matching standard SAE scale).
        # Without this, raw centroid norms cause reconstruction to explode.
        mean_norm = points.norm(dim=-1).mean()
        init_mult = -torch.log(mean_norm + 1e-8).item()
        if use_multiplier:
            self.multiplier = nn.Parameter(
                torch.tensor(init_mult, device=device), requires_grad=True
            )
        else:
            self.register_buffer(
                "multiplier", torch.tensor(init_mult, device=device)
            )

        self._fused_dictionary = None

    def get_dictionary(self):
        """Compute the dictionary D = (W @ C + Lambda) * exp(multiplier)."""
        if self.training:
            # Project W to be row-stochastic (non-negative, rows sum to 1)
            with torch.no_grad():
                W = torch.relu(self.W)
                W = W / (W.sum(dim=-1, keepdim=True) + 1e-8)
                self.W.data = W

                # Project Lambda so that ||Lambda_i||_2 <= delta per row
                norm_Lambda = self.Relax.norm(dim=-1, keepdim=True)
                scaling_factor = torch.clamp(self.delta / (norm_Lambda + 1e-8), max=1.0)
                self.Relax.data = self.Relax.data * scaling_factor

            D = self.W @ self.C + self.Relax
            return D * torch.exp(self.multiplier)
        else:
            assert self._fused_dictionary is not None, (
                "Dictionary not initialized. Call model.eval() first."
            )
            return self._fused_dictionary

    def forward(self, z):
        """Decode: x_hat = z @ D."""
        D = self.get_dictionary()
        return torch.matmul(z, D)

    def train(self, mode=True):
        if not mode:
            # Fuse dictionary when switching to eval
            with torch.no_grad():
                W = torch.relu(self.W)
                W = W / (W.sum(dim=-1, keepdim=True) + 1e-8)
                norm_Lambda = self.Relax.norm(dim=-1, keepdim=True)
                scaling_factor = torch.clamp(self.delta / (norm_Lambda + 1e-8), max=1.0)
                Relax = self.Relax * scaling_factor
                self._fused_dictionary = (W @ self.C + Relax) * torch.exp(self.multiplier)
        super().train(mode)


class ArchetypalSAE(nn.Module):
    """
    Relaxed Archetypal SAE (RA-SAE) with TopK activation for language models.

    Architecture:
      Encode:  pre_codes = W_enc @ x + b_enc
               codes = TopK(ReLU(pre_codes))
      Decode:  x_hat = codes @ D
               where D = (W @ C + Lambda) * exp(multiplier)

    Parameters
    ----------
    d_model : int
        Input dimension (e.g., 768 for GPT-2).
    n_features : int
        Number of dictionary atoms / latent features.
    anchor_points : torch.Tensor
        Data centroids from K-means, shape (n_centroids, d_model).
    top_k : int
        Number of top activations to keep. Default: n_features // 10.
    delta : float
        Relaxation parameter for Lambda constraint. Default: 1.0.
    use_multiplier : bool
        Whether to use a learnable dictionary multiplier. Default: True.
    """

    def __init__(self, d_model, n_features, anchor_points, top_k=None,
                 delta=1.0, use_multiplier=True):
        super().__init__()
        self.d_model = d_model
        self.n_features = n_features
        self.top_k = top_k if top_k is not None else max(n_features // 10, 1)

        # Encoder: linear projection + bias
        self.encoder = nn.Linear(d_model, n_features, bias=True)

        # Initialize encoder with Kaiming
        nn.init.kaiming_uniform_(self.encoder.weight, nonlinearity='relu')
        nn.init.zeros_(self.encoder.bias)

        # Archetypal dictionary (decoder)
        device = anchor_points.device
        self.dictionary = RelaxedArchetypalDictionary(
            in_dimensions=d_model,
            nb_concepts=n_features,
            points=anchor_points,
            delta=delta,
            use_multiplier=use_multiplier,
            device=device,
        )

    def encode(self, x):
        """
        Encode input to sparse latent codes using TopK.

        Returns
        -------
        pre_codes : Tensor
            Pre-activation values (before ReLU and TopK).
        codes : Tensor
            Sparse codes after ReLU + TopK.
        """
        pre_codes = self.encoder(x)
        codes = F.relu(pre_codes)

        # TopK: keep only the top_k largest activations, zero out the rest
        topk_vals, topk_indices = torch.topk(codes, self.top_k, dim=-1)
        codes = torch.zeros_like(codes).scatter(-1, topk_indices, topk_vals)

        return pre_codes, codes

    def decode(self, codes):
        """Decode sparse codes back to input space."""
        return self.dictionary(codes)

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
        return self.dictionary.get_dictionary()
