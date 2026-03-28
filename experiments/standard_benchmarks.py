"""standard_benchmarks.py — WikiText-2 perplexity and LAMBADA accuracy for PaES-LLM.

PaES is applied as a pre-embedding gate: token embeddings are processed by
a GatedTransformerBlock before being passed to the full model via
inputs_embeds. This sidesteps internal attention argument complexity (RoPE,
causal masks) while keeping the comparison fair — the gating signal is
applied at the representation level before any transformer layer.

For mode='off' the block is bypassed entirely, giving the model's native
perplexity as the reference.
"""

import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from src.models.gated_transformer import GatedTransformerBlock
from src.utils.token_loader import get_lambada_loader, get_wikitext2_loader

MODEL_NAME = "HuggingFaceTB/SmolLM-135M"


def _make_block(mode: str, threshold: float = 0.5) -> GatedTransformerBlock:
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(MODEL_NAME)
    return GatedTransformerBlock(
        embed_dim=cfg.hidden_size,
        num_heads=cfg.num_attention_heads,
        ff_dim=cfg.intermediate_size,
        threshold=threshold,
        mode=mode,
    ).to(torch.float32).eval()


def _forward(ref_model, block, input_ids, mode):
    """Embed → optional PaES gate → full model forward."""
    hidden      = ref_model.model.embed_tokens(input_ids).to(torch.float32)
    model_dtype = next(ref_model.parameters()).dtype

    if mode == "off":
        outputs = ref_model(input_ids=None, inputs_embeds=hidden.to(model_dtype))
        return outputs.logits, {"mean_savings": 0.0}

    block.gated_attn.gate.mode = mode
    block.gate_ffn.mode        = mode
    gated, stats = block(hidden)
    outputs = ref_model(input_ids=None, inputs_embeds=gated.to(model_dtype))
    return outputs.logits, stats


def compute_perplexity(
    mode: str = "off", n_samples: int = 200, threshold: float = 0.5
) -> dict:
    """WikiText-2 test perplexity. Lower is better."""
    ref_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
    ref_model.eval()
    block  = _make_block(mode, threshold)
    loader = get_wikitext2_loader(MODEL_NAME, n_samples=n_samples, batch_size=4)
    pad_id = ref_model.config.pad_token_id or 0

    total_nll, total_tokens = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            input_ids    = batch["input_ids"]
            logits, _    = _forward(ref_model, block, input_ids, mode)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=pad_id,
                reduction="sum",
            )
            total_nll    += loss.item()
            total_tokens += (shift_labels != pad_id).sum().item()

    ppl = math.exp(min(total_nll / max(total_tokens, 1), 20))
    return {"perplexity": ppl, "mode": mode, "n_tokens": total_tokens}


def compute_lambada_accuracy(
    mode: str = "off", n_samples: int = 200, threshold: float = 0.5
) -> dict:
    """LAMBADA last-word prediction accuracy. Higher is better."""
    ref_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
    ref_model.eval()
    block  = _make_block(mode, threshold)
    loader = get_lambada_loader(MODEL_NAME, n_samples=n_samples, batch_size=4)

    correct, total = 0, 0
    with torch.no_grad():
        for batch in loader:
            input_ids   = batch["input_ids"]
            logits, _   = _forward(ref_model, block, input_ids, mode)
            mask        = batch["attention_mask"]
            last_idx    = (mask.sum(dim=1) - 1).clamp(0, input_ids.size(1) - 2)
            preds       = logits[torch.arange(input_ids.size(0)), last_idx].argmax(dim=-1)
            targets     = input_ids[torch.arange(input_ids.size(0)), last_idx + 1]
            correct    += (preds == targets).sum().item()
            total      += input_ids.size(0)

    return {"accuracy_%": correct / max(total, 1) * 100, "correct": correct, "total": total}


def run_full_benchmark_suite(n_samples: int = 200) -> dict:
    """Run perplexity and LAMBADA for all three modes."""
    modes = [
        ("Baseline",  "off"),
        ("PaES-Lite", "heuristic"),
        ("PaES-Full", "projector"),
    ]
    results = {}
    print(f"Benchmarks — n_samples={n_samples}, model={MODEL_NAME}\n")

    for name, mode in modes:
        print(f"  {name}")
        t0  = time.perf_counter()
        ppl = compute_perplexity(mode=mode, n_samples=n_samples)
        lam = compute_lambada_accuracy(mode=mode, n_samples=n_samples)
        results[name] = {
            "perplexity": ppl["perplexity"],
            "lambada_%":  lam["accuracy_%"],
            "mode":       mode,
            "elapsed_s":  time.perf_counter() - t0,
        }
        print(f"    PPL={ppl['perplexity']:.2f}  LAMBADA={lam['accuracy_%']:.2f}%")

    _print_table(results)
    return results


def _print_table(results: dict) -> None:
    base_ppl = results.get("Baseline", {}).get("perplexity")
    print(f"\n{'Method':<15} {'Mode':<12} {'Perplexity':>12} {'LAMBADA (%)':>13}")
    print("-" * 56)
    for name, r in results.items():
        delta = ""
        if base_ppl and name != "Baseline":
            diff  = r["perplexity"] - base_ppl
            delta = f"  ({'+' if diff >= 0 else ''}{diff:.2f})"
        print(f"{name:<15} {r['mode']:<12} {r['perplexity']:>11.2f}{delta}  {r['lambada_%']:>10.2f}%")
    print()
