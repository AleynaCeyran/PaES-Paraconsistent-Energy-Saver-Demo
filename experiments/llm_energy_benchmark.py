"""llm_energy_benchmark.py — Compute savings and wall-clock latency benchmark."""

import os
import time

import torch

from src.models.gated_transformer import GatedTransformerBlock
from src.utils.token_loader import get_token_loader


def _load_projector(block: GatedTransformerBlock, ckpt_path: str) -> None:
    """Load trained projector weights if the checkpoint dimension matches."""
    if ckpt_path is None or not os.path.exists(ckpt_path):
        return
    ckpt      = torch.load(ckpt_path, map_location="cpu")
    ckpt_dim  = ckpt.get("embed_dim")
    block_dim = block.gated_attn.gate.embed_dim
    if ckpt_dim is not None and ckpt_dim != block_dim:
        return  # silently skip; block uses random init
    if hasattr(block.gated_attn.gate, "projector"):
        block.gated_attn.gate.projector.load_state_dict(ckpt["attn_proj"])
    if hasattr(block.gate_ffn, "projector"):
        block.gate_ffn.projector.load_state_dict(ckpt["ffn_proj"])


def run_llm_benchmark(
    texts,
    embed_dim: int = 256,
    threshold: float = 0.5,
    mode: str = "projector",
    projector_ckpt: str = None,
) -> dict:
    """Measure per-token compute savings and wall-clock latency for a gating mode.

    Savings are derived from the weighted Belnap state distribution returned
    by the gating module — not from hardcoded constants. When a projector
    checkpoint is provided its embed_dim overrides the embed_dim argument
    to ensure dimensional consistency.

    Parameters
    ----------
    texts          : List of input strings.
    embed_dim      : Token embedding dimensionality (ignored if ckpt provided).
    threshold      : Belnap L4 decision threshold τ.
    mode           : Gating mode ('off' | 'heuristic' | 'projector').
    projector_ckpt : Path to a saved projector checkpoint (optional).

    Returns
    -------
    dict
        mean_savings_pct   : Average compute savings across all batches (%).
        mean_wall_time_ms  : Average per-token wall-clock time (ms).
        states_dist        : Mean fraction of tokens per Belnap state.
    """
    loader = get_token_loader(texts, batch_size=1)

    if projector_ckpt and os.path.exists(projector_ckpt):
        ckpt      = torch.load(projector_ckpt, map_location="cpu")
        embed_dim = ckpt.get("embed_dim", embed_dim)
        num_heads = max(1, embed_dim // 64)
        ff_dim    = embed_dim * 4
    else:
        num_heads = 8
        ff_dim    = 1024

    model = GatedTransformerBlock(
        embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim,
        threshold=threshold, mode=mode,
    )
    _load_projector(model, projector_ckpt)
    model.eval()

    wall_times, savings_list = [], []
    states_dist = {"TRUE": 0.0, "FALSE": 0.0, "BOTH": 0.0, "NONE": 0.0}
    use_cuda = torch.cuda.is_available()

    with torch.no_grad():
        for batch in loader:
            seq_len = batch["input_ids"].shape[1]
            x       = torch.randn(1, seq_len, embed_dim)

            if use_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _, stats = model(x)
            if use_cuda:
                torch.cuda.synchronize()

            wall_times.append((time.perf_counter() - t0) * 1000 / seq_len)
            savings_list.append(stats["mean_savings"])

            states_dist["TRUE"]  += stats.get("attn_pct_true",  0.0)
            states_dist["FALSE"] += stats.get("attn_pct_false", 0.0)
            states_dist["BOTH"]  += stats.get("attn_pct_both",  0.0)
            states_dist["NONE"]  += stats.get("attn_pct_none",  0.0)

    n = max(len(savings_list), 1)
    for k in states_dist:
        states_dist[k] /= n

    return {
        "mean_savings_pct":  sum(savings_list) / n * 100,
        "mean_wall_time_ms": sum(wall_times) / len(wall_times) if wall_times else 0.0,
        "states_dist":       states_dist,
    }
