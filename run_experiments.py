"""
run_experiments.py — PaES-LLM experiment runner.

Usage:
    python run_experiments.py --mode smoke        # quick sanity check
    python run_experiments.py --mode audit        # main results table
    python run_experiments.py --mode benchmark    # WikiText-2 + LAMBADA
    python run_experiments.py --mode ablation     # threshold x mode grid
    python run_experiments.py --mode comparison   # vs literature baselines
    python run_experiments.py --mode all          # full suite
"""

import argparse
import json
import os
import time

import numpy as np
import torch

OUTPUT_DIR = "results/figures"
CKPT_DIR   = "results/checkpoints"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CKPT_DIR,   exist_ok=True)

CORPUS = [
    "Paraconsistent logic handles contradictory information without trivialization.",
    "Large Language Models consume massive electrical energy for global inference.",
    "Quantum decoherence in superconducting qubits requires cryogenic cooling.",
    "Epistemic uncertainty is modeled via Belnap L4 lattice structures.",
    "The transformer architecture relies on self-attention for sequence modeling.",
    "Energy-efficient inference is critical for sustainable AI deployment.",
    "Sparse computation methods reduce the computational cost of neural networks.",
    "Semantic preservation is essential when pruning attention in language models.",
]

SCENARIOS = [
    {"name": "Baseline",  "mode": "off"},
    {"name": "PaES-Lite", "mode": "heuristic"},
    {"name": "PaES-Full", "mode": "projector"},
]


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def _print_table(summary: dict) -> None:
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


def run_audit(n_runs: int = 10) -> dict:
    from experiments.llm_accuracy_test import run_semantic_consistency_test
    from experiments.llm_energy_benchmark import run_llm_benchmark

    summary = {}
    print(f"Audit — {n_runs} runs per scenario\n")

    for sc in SCENARIOS:
        print(f"  {sc['name']}")
        savings_list, fidelity_list, runtime_list = [], [], []

        for i in range(n_runs):
            set_seed(42 + i)
            tau = 0.5 + (i - n_runs // 2) * 0.004

            eff = run_llm_benchmark(CORPUS, threshold=tau, mode=sc["mode"])
            acc = run_semantic_consistency_test(CORPUS, threshold=tau, mode=sc["mode"])

            savings_list.append(eff["mean_savings_pct"])
            fidelity_list.append(acc["cosine_sim"] * 100)
            runtime_list.append(eff["mean_wall_time_ms"])

        summary[sc["name"]] = {
            "savings_mean":  float(np.mean(savings_list)),
            "savings_std":   float(np.std(savings_list)),
            "fidelity_mean": float(np.mean(fidelity_list)),
            "fidelity_std":  float(np.std(fidelity_list)),
            "runtime_mean":  float(np.mean(runtime_list)),
            "runtime_std":   float(np.std(runtime_list)),
        }

    _print_table(summary)

    path = os.path.join(CKPT_DIR, "audit_results.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {path}")
    return summary


def run_benchmark(n_samples: int = 200) -> dict:
    from experiments.standard_benchmarks import run_full_benchmark_suite

    results = run_full_benchmark_suite(n_samples=n_samples)
    path = os.path.join(CKPT_DIR, "benchmark_results.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {path}")
    return results


def run_ablation(n_repeat: int = 3) -> dict:
    from experiments.ablation_suite import run_ablation as _run

    return _run(output_dir=CKPT_DIR, n_repeat=n_repeat)


def run_comparison() -> dict:
    from experiments.ablation_suite import run_comparison_table

    return run_comparison_table(output_dir=CKPT_DIR)


def main() -> None:
    parser = argparse.ArgumentParser(description="PaES-LLM experiment runner")
    parser.add_argument(
        "--mode",
        choices=["smoke", "audit", "benchmark", "ablation", "comparison", "all"],
        default="audit",
    )
    parser.add_argument("--n_runs",    type=int, default=10)
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--n_repeat",  type=int, default=3)
    args = parser.parse_args()

    t0 = time.perf_counter()

    if args.mode == "smoke":
        run_audit(n_runs=3)
    elif args.mode == "audit":
        run_audit(n_runs=args.n_runs)
    elif args.mode == "benchmark":
        run_benchmark(n_samples=args.n_samples)
    elif args.mode == "ablation":
        run_ablation(n_repeat=args.n_repeat)
    elif args.mode == "comparison":
        run_comparison()
    elif args.mode == "all":
        run_audit(n_runs=args.n_runs)
        run_benchmark(n_samples=args.n_samples)
        run_ablation(n_repeat=args.n_repeat)
        run_comparison()

    print(f"Done in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
