"""gated_transformer.py — Transformer block with dual-level Belnap L4 gating."""

import torch
import torch.nn as nn
from typing import Dict, Tuple

from src.gating.paes_llm_core import PaESLLMCore


class GatedMultiHeadAttention(nn.Module):
    """Multi-head self-attention gated by an epistemic energy coefficient.

    The gate scores each token independently; tokens classified as FALSE
    receive a zero coefficient, effectively suppressing their contribution
    to the attention output without modifying the attention computation graph.

    Parameters
    ----------
    embed_dim  : Embedding dimensionality.
    num_heads  : Number of attention heads.
    threshold  : Belnap L4 decision threshold τ.
    mode       : Gating mode forwarded to PaESLLMCore.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        threshold: float = 0.5,
        mode: str = "projector",
    ) -> None:
        super().__init__()
        self.mha  = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.gate = PaESLLMCore(embed_dim=embed_dim, threshold=threshold, mode=mode)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        energy_coeff, state_map, stats = self.gate(x)
        attn_out, _ = self.mha(x, x, x)
        return attn_out * energy_coeff.unsqueeze(-1), state_map, stats


class GatedTransformerBlock(nn.Module):
    """Pre-norm transformer block with token-level epistemic gating.

    Gating is applied at two sublayers:

    1. Attention sublayer — per-token energy coefficient scales the
       attention output before the residual addition.
    2. FFN sublayer — per-token coefficient scales the FFN output,
       allowing individual tokens to bypass the feed-forward computation.

    Both gates share the same mode and threshold, but maintain independent
    weight matrices when mode='projector'.

    Parameters
    ----------
    embed_dim  : Token embedding dimensionality.
    num_heads  : Number of attention heads.
    ff_dim     : Hidden dimensionality of the feed-forward network.
    threshold  : Belnap L4 decision threshold τ (default 0.5).
    mode       : 'projector' | 'heuristic' | 'off'.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        threshold: float = 0.5,
        mode: str = "projector",
    ) -> None:
        super().__init__()
        self.ln1        = nn.LayerNorm(embed_dim)
        self.ln2        = nn.LayerNorm(embed_dim)
        self.gated_attn = GatedMultiHeadAttention(embed_dim, num_heads, threshold, mode)
        self.gate_ffn   = PaESLLMCore(embed_dim=embed_dim, threshold=threshold, mode=mode)
        self.ffn        = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        orig_dtype = x.dtype
        if x.dtype != torch.float32:
            x = x.to(torch.float32)

        # Attention sublayer
        residual = x
        attn_out, _, attn_stats = self.gated_attn(self.ln1(x))
        x = residual + attn_out

        # FFN sublayer
        residual = x
        ffn_coeff, _, ffn_stats = self.gate_ffn(self.ln2(x))
        ffn_out = self.ffn(self.ln2(x)) * ffn_coeff.unsqueeze(-1)
        x = residual + ffn_out

        stats = {
            "attn_savings":   attn_stats["savings"],
            "ffn_savings":    ffn_stats["savings"],
            "mean_savings":   (attn_stats["savings"] + ffn_stats["savings"]) / 2,
            "attn_pct_false": attn_stats["pct_false"],
            "ffn_pct_false":  ffn_stats["pct_false"],
            "attn_pct_true":  attn_stats["pct_true"],
            "ffn_pct_true":   ffn_stats["pct_true"],
        }
        return x.to(orig_dtype), stats
