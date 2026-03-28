"""
main_llm.py — Legacy audit runner (backward-compatible with early experiments).

For new experiments, prefer run_experiments.py which provides a cleaner
entry point with full benchmark support.
"""

import argparse
import json
import os

import numpy as np
import torch

from experiments.llm_accuracy_test import run_semantic_consistency_test
from experiments.llm_energy_benchmark import run_llm_benchmark
from src.utils.token_loader import get_token_loader


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


CORPUS = [
    "Paraconsistent logic handles contradictory information without trivialization.",
    "Large Language Models consume massive electrical energy for global inference.",
    "Quantum decoherence in superconducting qubits requires cryogenic cooling.",
    "Epistemic uncertainty is modeled via Belnap L4 lattice structures.",
]

SCENARIOS = [
    {"name": "Baseline",  "mode": "off"},
    {"name": "PaES-Lite", "mode": "heuristic"},
    {"name": "PaES-Full", "mode": "projector"},
]


def main() -> None:
    parser = argparse.ArgumentParser(description="PaES-LLM audit runner")
    parser.add_argument("--n_runs",          type=int,  default=10)
    parser.add_argument("--embed_dim",       type=int,  default=256)
    parser.add_argument("--train_projector", action="store_true")
    parser.add_argument("--output_dir",      type=str,  default="results/figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.train_projector:
        # Projector training on synthetic inputs does not generalise to real
        # LLM embeddings. Xavier-uniform initialisation yields comparable or
        # better results. Real-data fine-tuning is planned for Faz-2.
        print("Note: --train_projector is a no-op in this version (see Faz-2 roadmap).")

    summary = {}
    print(f"Audit — {args.n_runs} runs per scenario\n")

    for sc in SCENARIOS:
        print(f"  {sc['name']}")
        metrics = {"savings": [], "fidelity": [], "runtime": []}

        for i in range(args.n_runs):
            set_seed(42 + i)
            tau = 0.5 + (i - args.n_runs // 2) * 0.004

            eff = run_llm_benchmark(
                CORPUS, args.embed_dim,
                mode=sc["mode"],
                threshold=tau,
                projector_ckpt=None,
            )
            acc = run_semantic_consistency_test(
                CORPUS, args.embed_dim,
                mode=sc["mode"],
                threshold=tau,
                projector_ckpt=None,
            )

            metrics["savings"].append(eff["mean_savings_pct"])
            metrics["fidelity"].append(acc["cosine_sim"] * 100)
            metrics["runtime"].append(eff["mean_wall_time_ms"])

        summary[sc["name"]] = {
            "savings_mean":  float(np.mean(metrics["savings"])),
            "savings_std":   float(np.std(metrics["savings"])),
            "fidelity_mean": float(np.mean(metrics["fidelity"])),
            "fidelity_std":  float(np.std(metrics["fidelity"])),
            "runtime_mean":  float(np.mean(metrics["runtime"])),
            "runtime_std":   float(np.std(metrics["runtime"])),
        }

    header = f"{'Method':<15} | {'Savings (%)':<20} | {'Fidelity (%)':<20} | {'Latency (ms/tok)'}"
    print("\n" + header)
    print("-" * len(header))
    for name, d in summary.items():
        print(
            f"{name:<15} | "
            f"{d['savings_mean']:>8.2f} ± {d['savings_std']:>5.2f} | "
            f"{d['fidelity_mean']:>8.2f} ± {d['fidelity_std']:>5.2f} | "
            f"{d['runtime_mean']:>9.3f} ± {d['runtime_std']:>6.3f}"
        )
    print()

    ckpt_dir = os.path.join(os.path.dirname(args.output_dir), "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    out_path = os.path.join(ckpt_dir, "ablation_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
