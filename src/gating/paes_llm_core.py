"""paes_llm_core.py — Belnap L4 epistemic gating for transformer tokens."""

import torch
import torch.nn as nn
from typing import Dict, Tuple


class PaESLLMCore(nn.Module):
    """Epistemic gating engine based on Belnap's four-valued logic (L4).

    Each token embedding is projected into a 2-D evidence-uncertainty space
    (μ, λ) and classified into one of four epistemic states. The resulting
    per-token energy coefficient controls how much compute is allocated to
    that token in the subsequent attention and feed-forward layers.

    States and energy coefficients
    --------------------------------
    TRUE  (μ ≥ τ, λ < τ)  → 1.00  full compute
    FALSE (μ < τ, λ ≥ τ)  → 0.00  hardware bypass
    BOTH  (μ ≥ τ, λ ≥ τ)  → 0.15  contradictory evidence
    NONE  (μ < τ, λ < τ)  → 0.25  indeterminate

    Parameters
    ----------
    embed_dim : int
        Dimensionality of input token embeddings.
    threshold : float
        Decision boundary τ for L4 classification.
    mode : str
        'projector'  — learned linear projection (xavier-uniform init).
        'heuristic'  — statistics-based μ/λ without learned parameters.
        'off'        — identity gate; used as the no-gating baseline.
    """

    STATE_TRUE  = 0
    STATE_FALSE = 1
    STATE_BOTH  = 2
    STATE_NONE  = 3

    COEFF: Dict[int, float] = {
        STATE_TRUE:  1.00,
        STATE_FALSE: 0.00,
        STATE_BOTH:  0.15,
        STATE_NONE:  0.25,
    }

    def __init__(
        self,
        embed_dim: int = 256,
        threshold: float = 0.5,
        mode: str = "projector",
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.threshold = threshold
        self.mode = mode.lower()

        if self.mode == "projector":
            self.projector = nn.Linear(embed_dim, 2)
            nn.init.xavier_uniform_(self.projector.weight)
            nn.init.zeros_(self.projector.bias)

    def _heuristic_mu_lambda(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Derive μ and λ from embedding magnitude and dispersion."""
        abs_mean = x.abs().mean(dim=-1)
        mu  = torch.sigmoid(abs_mean)
        lam = torch.sigmoid(x.std(dim=-1) / (abs_mean + 1e-6))
        return mu, lam

    def _classify(
        self, mu: torch.Tensor, lam: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """Assign Belnap states and compute per-token energy coefficients."""
        tau = self.threshold
        hi_mu  = mu >= tau
        hi_lam = lam >= tau

        state_map = torch.full(mu.shape, self.STATE_TRUE, dtype=torch.long, device=mu.device)
        state_map[~hi_mu &  hi_lam] = self.STATE_FALSE
        state_map[ hi_mu &  hi_lam] = self.STATE_BOTH
        state_map[~hi_mu & ~hi_lam] = self.STATE_NONE

        coeff = torch.ones_like(mu)
        for state, value in self.COEFF.items():
            if value != 1.0:
                coeff[state_map == state] = value

        n        = mu.numel()
        n_true   = (state_map == self.STATE_TRUE).sum().item()
        n_false  = (state_map == self.STATE_FALSE).sum().item()
        n_both   = (state_map == self.STATE_BOTH).sum().item()
        n_none   = (state_map == self.STATE_NONE).sum().item()

        compute_used  = n_true + n_both * 0.15 + n_none * 0.25
        savings_ratio = 1.0 - compute_used / n if n > 0 else 0.0

        stats = {
            "savings":     savings_ratio,
            "pct_true":    n_true  / n,
            "pct_false":   n_false / n,
            "pct_both":    n_both  / n,
            "pct_none":    n_none  / n,
            "mean_mu":     mu.mean().item(),
            "mean_lambda": lam.mean().item(),
        }
        return coeff, state_map, stats

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """Gate token embeddings through the Belnap L4 classifier.

        Parameters
        ----------
        x : Tensor[..., embed_dim]

        Returns
        -------
        energy_coeff : Tensor[...]          per-token energy multiplier
        state_map    : LongTensor[...]      per-token Belnap state index
        stats        : dict                 aggregate metrics
        """
        if self.mode == "off":
            shape = x.shape[:-1]
            return (
                torch.ones(shape, device=x.device, dtype=x.dtype),
                torch.zeros(shape, dtype=torch.long, device=x.device),
                {
                    "savings": 0.0,
                    "pct_true": 1.0, "pct_false": 0.0,
                    "pct_both": 0.0, "pct_none":  0.0,
                    "mean_mu": 1.0,  "mean_lambda": 0.0,
                },
            )

        if self.mode == "projector":
            proj = self.projector(x)
            mu, lam = torch.sigmoid(proj[..., 0]), torch.sigmoid(proj[..., 1])
        else:
            mu, lam = self._heuristic_mu_lambda(x)

        return self._classify(mu, lam)
