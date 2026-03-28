"""llm_accuracy_test.py — Semantic fidelity measurement for PaES gating."""

import os

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

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


def run_semantic_consistency_test(
    texts,
    embed_dim=None,
    model_name: str = "HuggingFaceTB/SmolLM-135M",
    threshold: float = 0.5,
    mode: str = "projector",
    projector_ckpt: str = None,
) -> dict:
    """Measure semantic fidelity of PaES gating relative to an ungated baseline.

    Both the baseline and PaES passes share a single GatedTransformerBlock
    with identical weights. The baseline pass runs with mode='off'
    (energy_coeff=1 everywhere); the PaES pass uses the requested mode.
    Fidelity is the mean cosine similarity between the two outputs.

    Parameters
    ----------
    texts          : List of input strings.
    embed_dim      : Unused; kept for backward compatibility.
    model_name     : HuggingFace model identifier for tokeniser and embeddings.
    threshold      : Belnap L4 decision threshold τ.
    mode           : Gating mode ('off' | 'heuristic' | 'projector').
    projector_ckpt : Path to a saved projector checkpoint (optional).

    Returns
    -------
    dict
        cosine_sim : mean cosine similarity (fidelity proxy, 1.0 = perfect).
        savings    : mean compute savings ratio.
    """
    ref_model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
    ref_model.eval()

    cfg    = ref_model.config
    loader = get_token_loader(texts, model_name=model_name)

    shared_block = GatedTransformerBlock(
        embed_dim=cfg.hidden_size,
        num_heads=cfg.num_attention_heads,
        ff_dim=cfg.intermediate_size,
        threshold=threshold,
        mode=mode,
    ).to(torch.float32)
    _load_projector(shared_block, projector_ckpt)
    shared_block.eval()

    cos_sims, savings_list = [], []

    with torch.no_grad():
        for batch in loader:
            hidden = ref_model.model.embed_tokens(batch["input_ids"]).to(torch.float32)

            shared_block.gated_attn.gate.mode = "off"
            shared_block.gate_ffn.mode        = "off"
            baseline_out, _ = shared_block(hidden)

            shared_block.gated_attn.gate.mode = mode
            shared_block.gate_ffn.mode        = mode
            paes_out, paes_stats = shared_block(hidden)

            b = baseline_out.view(-1, cfg.hidden_size)
            p = paes_out.view(-1, cfg.hidden_size)
            cos_sims.append(F.cosine_similarity(b, p, dim=1).mean().item())
            savings_list.append(paes_stats["mean_savings"])

    return {
        "cosine_sim": float(np.mean(cos_sims)),
        "savings":    float(np.mean(savings_list)),
    }
