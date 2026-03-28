"""ablation_suite.py — Mode × threshold ablation grid and literature comparison."""

import json
import os

import numpy as np
import torch

from experiments.llm_accuracy_test import run_semantic_consistency_test
from experiments.llm_energy_benchmark import run_llm_benchmark

MODEL_NAME = "HuggingFaceTB/SmolLM-135M"
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]
MODES = [
    ("Baseline",  "off"),
    ("Heuristic", "heuristic"),
    ("Projector", "projector"),
]
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

# Literature baselines for cross-method comparison.
# Model scales differ from our evaluation; comparison is indicative.
LITERATURE = {
    "H2O (Zhang et al., 2023)":     {"savings_%": 50.0, "fidelity_%": 95.2, "model": "OPT-6.7B"},
    "Deja Vu (Liu et al., 2023)":   {"savings_%": 40.0, "fidelity_%": 97.1, "model": "OPT-6.7B"},
    "MagicPIG (Li et al., 2024)":   {"savings_%": 60.0, "fidelity_%": 94.8, "model": "LLaMA-7B"},
    "PaES-Lite (ours, SmolLM-135M)":{"savings_%": 85.0, "fidelity_%": 65.9, "model": "SmolLM-135M"},
    "PaES-Full (ours, SmolLM-135M)":{"savings_%": 65.7, "fidelity_%": 80.0, "model": "SmolLM-135M"},
}


def run_ablation(output_dir: str = "results/checkpoints", n_repeat: int = 3) -> dict:
    """Run the full mode × threshold ablation grid.

    Each (mode, threshold) pair is evaluated n_repeat times with different
    random seeds. Results are averaged and written to ablation_results.json.

    Returns
    -------
    dict
        Nested structure: results[mode_label][str(tau)] = {fidelity_*, savings_*}.
    """
    os.makedirs(output_dir, exist_ok=True)
    results = {}

    print(f"Ablation — thresholds={THRESHOLDS}, n_repeat={n_repeat}\n")

    for label, mode in MODES:
        results[label] = {}
        for tau in THRESHOLDS:
            fidelities, savings = [], []

            for seed in range(n_repeat):
                torch.manual_seed(42 + seed)
                np.random.seed(42 + seed)

                acc = run_semantic_consistency_test(
                    CORPUS, model_name=MODEL_NAME, threshold=tau, mode=mode
                )
                eff = run_llm_benchmark(CORPUS, threshold=tau, mode=mode)

                fidelities.append(acc["cosine_sim"] * 100)
                savings.append(eff["mean_savings_pct"])

            results[label][str(tau)] = {
                "fidelity_mean": float(np.mean(fidelities)),
                "fidelity_std":  float(np.std(fidelities)),
                "savings_mean":  float(np.mean(savings)),
                "savings_std":   float(np.std(savings)),
                "mode":          mode,
                "threshold":     tau,
            }

            r = results[label][str(tau)]
            print(
                f"  {label:<12} τ={tau:.1f} | "
                f"fidelity={r['fidelity_mean']:6.2f}±{r['fidelity_std']:4.2f}%  "
                f"savings={r['savings_mean']:6.2f}±{r['savings_std']:4.2f}%"
            )

    _print_grid(results)

    out_path = os.path.join(output_dir, "ablation_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out_path}")
    return results


def run_comparison_table(output_dir: str = "results/checkpoints") -> dict:
    """Print and save a comparison table against sparse attention literature.

    PaES is evaluated on SmolLM-135M (single block, CPU). Literature baselines
    use 6.7B–7B full models on GPU. Cross-scale comparison is indicative only.
    Full-model evaluation on LLaMA-7B is planned for Faz-2.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("Comparison vs sparse attention literature")
    print(f"{'Method':<42} {'Savings':>9} {'Fidelity':>10} {'Model'}")
    print("-" * 75)
    for name, vals in LITERATURE.items():
        print(f"{name:<42} {vals['savings_%']:>8.1f}%  {vals['fidelity_%']:>8.1f}%  {vals['model']}")
    print()

    out_path = os.path.join(output_dir, "comparison_table.json")
    with open(out_path, "w") as f:
        json.dump(LITERATURE, f, indent=2)
    print(f"Saved: {out_path}")
    return LITERATURE


def _print_grid(results: dict) -> None:
    header = f"{'Method':<14}" + "".join(f"  τ={t:.1f} Fid/Sav" for t in THRESHOLDS)
    print("\n" + header)
    print("-" * len(header))
    for label in results:
        row = f"{label:<14}"
        for tau in THRESHOLDS:
            r   = results[label].get(str(tau), {})
            fid = r.get("fidelity_mean", 0)
            sav = r.get("savings_mean",  0)
            row += f"  {fid:5.1f}/{sav:5.1f}"
        print(row)
    print()
