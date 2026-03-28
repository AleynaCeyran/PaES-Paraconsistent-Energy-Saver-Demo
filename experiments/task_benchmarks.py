"""task_benchmarks.py — Next-token prediction accuracy and latency for PaES-LLM."""

import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.models.gated_transformer import GatedTransformerBlock

BENCHMARK_DATA = [
    "The total of 15 and 27 is 42.",
    "If a train travels at 60 mph for 2 hours, it covers 120 miles.",
    "In a room with 3 people, there are 6 hands in total.",
    "The square root of 16 is equal to 4.",
    "A paraconsistent logic system can handle contradictions without failing.",
]


def run_task_benchmark(
    model_name: str = "HuggingFaceTB/SmolLM-135M",
    mode: str = "projector",
    threshold: float = 0.5,
    n_timing_runs: int = 50,
) -> dict:
    """Evaluate PaES on next-token prediction accuracy and per-token latency.

    Ground truth is computed from the full model stack (embed → all layers →
    lm_head). The PaES path routes embeddings through the gated block before
    the lm_head, so any degradation is directly attributable to the gating.

    Latency is measured over n_timing_runs warm iterations on a fixed batch
    with CUDA synchronisation barriers where available. The first five runs
    are discarded as warm-up.

    Parameters
    ----------
    model_name    : HuggingFace model identifier.
    mode          : Gating mode ('off' | 'heuristic' | 'projector').
    threshold     : Belnap L4 decision threshold τ.
    n_timing_runs : Total timing iterations (including 5 warm-up runs).

    Returns
    -------
    dict
        ntp_accuracy_%     : Next-token prediction accuracy vs reference (%).
        savings_%          : Mean compute savings (%).
        lat_ms_per_tok     : Mean per-token latency (ms).
        lat_ms_per_tok_std : Std of per-token latency (ms).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
    model.eval()
    device = next(model.parameters()).device

    gated_block = GatedTransformerBlock(
        embed_dim=model.config.hidden_size,
        num_heads=model.config.num_attention_heads,
        ff_dim=model.config.intermediate_size,
        mode=mode,
        threshold=threshold,
    ).to(torch.float32).eval()

    ntp_matches, savings_list = [], []

    with torch.no_grad():
        for text in BENCHMARK_DATA:
            input_ids     = tokenizer(text, return_tensors="pt")["input_ids"].to(device)
            ref_logits    = model(input_ids).logits
            target_tokens = ref_logits.argmax(dim=-1)

            hidden      = model.model.embed_tokens(input_ids).to(torch.float32)
            gated_out, stats = gated_block(hidden)
            paes_logits = model.lm_head(gated_out.to(model.lm_head.weight.dtype))
            pred_tokens = paes_logits.argmax(dim=-1)

            ntp_matches.append((pred_tokens == target_tokens).float().mean().item())
            savings_list.append(stats["mean_savings"])

    avg_ntp     = sum(ntp_matches) / len(ntp_matches) * 100
    avg_savings = sum(savings_list) / len(savings_list) * 100

    sample_ids = tokenizer(BENCHMARK_DATA[0], return_tensors="pt")["input_ids"].to(device)
    sample_h   = model.model.embed_tokens(sample_ids).to(torch.float32)
    use_cuda   = device.type == "cuda"

    latencies = []
    with torch.no_grad():
        for _ in range(n_timing_runs):
            if use_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            gated_block(sample_h)
            if use_cuda:
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000)

    latencies     = latencies[5:]           # discard warm-up
    n_tokens      = sample_ids.shape[1]
    lat_mean      = sum(latencies) / len(latencies)
    lat_std       = (sum((x - lat_mean) ** 2 for x in latencies) / len(latencies)) ** 0.5
    lat_per_tok   = lat_mean / n_tokens
    lat_std_tok   = lat_std  / n_tokens

    print(f"Task benchmark — mode={mode.upper()}  τ={threshold}")
    print(f"  NTP accuracy : {avg_ntp:.2f}%  ({'PASS' if avg_ntp > 95 else 'FAIL'})")
    print(f"  Savings      : {avg_savings:.2f}%")
    print(f"  Latency      : {lat_per_tok:.3f} ± {lat_std_tok:.3f} ms/token")

    return {
        "ntp_accuracy_%":     avg_ntp,
        "savings_%":          avg_savings,
        "lat_ms_per_tok":     lat_per_tok,
        "lat_ms_per_tok_std": lat_std_tok,
    }
