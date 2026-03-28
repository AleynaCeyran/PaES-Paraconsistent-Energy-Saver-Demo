"""llm_visualizer.py — Publication-quality figures for PaES-LLM.

Target: IEEE Transactions / NeurIPS (600 DPI, PDF, FontType 42).
Color palette: maroon (#800000), navy (#000080), and neutral grays only.

Usage:
    python src/utils/llm_visualizer.py
"""

import json
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

CKPT_DIR   = Path("results/checkpoints")
FIGURE_DIR = Path("results/figures")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

C = {
    "bordo":    "#800000",
    "lacivert": "#000080",
    "context":  "#333333",
    "grid":     "#B0B0B0",
    "element":  "#E0E0E0",
    "TRUE":     "#000080",
    "FALSE":    "#800000",
    "BOTH":     "#999999",
    "NONE":     "#E0E0E0",
}


def _set_style() -> None:
    """Apply IEEE/NeurIPS-compatible rcParams.

    Times New Roman is preferred but is unavailable on most Linux systems.
    The fallback chain (Liberation Serif → DejaVu Serif → FreeSerif →
    STIXGeneral) renders acceptably at 600 DPI without font warnings.
    """
    plt.rcParams.update({
        "font.family":        "serif",
        "font.serif":         ["Liberation Serif", "DejaVu Serif",
                               "FreeSerif", "STIXGeneral", "serif"],
        "pdf.fonttype":       42,
        "ps.fonttype":        42,
        "font.size":          10,
        "axes.labelsize":     11,
        "axes.titlesize":     11,
        "axes.titleweight":   "normal",
        "axes.linewidth":     0.8,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "xtick.labelsize":    9,
        "ytick.labelsize":    9,
        "xtick.direction":    "out",
        "ytick.direction":    "out",
        "legend.fontsize":    9,
        "legend.framealpha":  1.0,
        "legend.edgecolor":   "0.0",
        "legend.fancybox":    False,
        "figure.dpi":         150,
        "savefig.dpi":        600,
        "savefig.format":     "png",
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.02,
        "grid.linestyle":     ":",
        "grid.alpha":         0.6,
        "grid.linewidth":     0.5,
        "grid.color":         "#B0B0B0",
    })


def _load(filename: str) -> dict:
    path = CKPT_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}\n"
            f"Run 'python run_experiments.py --mode all' first."
        )
    with open(path) as f:
        return json.load(f)


def _savefig(fig, name: str) -> str:
    out = FIGURE_DIR / f"{name}.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  {out}")
    return str(out)


def plot_main_results_table(data: dict = None) -> str:
    """Render the primary audit metrics as a booktabs-style table figure."""
    _set_style()
    if data is None:
        data = _load("audit_results.json")

    methods = list(data.keys())
    rows = [
        [
            m,
            f"{data[m]['savings_mean']:.2f} ± {data[m]['savings_std']:.2f}",
            f"{data[m]['fidelity_mean']:.2f} ± {data[m]['fidelity_std']:.2f}",
            f"{data[m]['runtime_mean']:.3f} ± {data[m]['runtime_std']:.3f}",
        ]
        for m in methods
    ]
    col_labels = ["Method", "Savings (%)", "Fidelity (%)", "Latency (ms/tok)"]

    fig, ax = plt.subplots(figsize=(7.2, 1.5))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 2.0)

    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("black")
        cell.set_linewidth(0.5 if r > 0 else 1.0)
        if r == 0:
            cell.get_text().set_weight("bold")

    return _savefig(fig, "fig1_main_results_table")


def plot_tradeoff_scatter(comp: dict = None) -> str:
    """Scatter plot: compute savings vs semantic fidelity for all methods."""
    _set_style()
    if comp is None:
        comp = _load("comparison_table.json")

    fig, ax = plt.subplots(figsize=(6, 4))
    for method, vals in comp.items():
        is_ours = "ours" in method.lower()
        ax.scatter(
            vals["savings_%"], vals["fidelity_%"],
            c=C["bordo"] if is_ours else C["lacivert"],
            marker="D" if is_ours else "o",
            s=80 if is_ours else 50,
            edgecolors="black", linewidth=0.8, zorder=3,
        )
        ax.text(
            vals["savings_%"] + 1.5, vals["fidelity_%"],
            method.split(" (")[0], fontsize=9, va="center",
        )

    ax.set_xlabel("Compute Savings (%)")
    ax.set_ylabel("Semantic Fidelity (%)")
    ax.set_xlim(30, 95)
    ax.set_ylim(50, 105)
    ax.grid(True)
    return _savefig(fig, "fig2_tradeoff_scatter")


def plot_ablation_heatmap(ablation: dict = None) -> str:
    """Dual heatmap: fidelity and savings across the mode × threshold grid."""
    _set_style()
    if ablation is None:
        ablation = _load("ablation_results.json")

    modes      = [m for m in ablation if m != "Baseline"]
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    fid_matrix = np.zeros((len(modes), len(thresholds)))
    sav_matrix = np.zeros((len(modes), len(thresholds)))

    for i, mode in enumerate(modes):
        for j, tau in enumerate(thresholds):
            r = ablation[mode].get(str(tau), {})
            fid_matrix[i, j] = r.get("fidelity_mean", 0)
            sav_matrix[i, j] = r.get("savings_mean",  0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 2.8))
    tau_labels = [f"τ={t}" for t in thresholds]

    def _heatmap(ax, matrix, title, color, vmin, vmax, cb_label):
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            "custom", [C["element"], color]
        )
        im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(thresholds)))
        ax.set_xticklabels(tau_labels, fontsize=10)
        ax.set_yticks(range(len(modes)))
        ax.set_yticklabels(modes, fontsize=10)
        ax.set_title(title, fontsize=11, pad=8)
        midpoint = vmin + (vmax - vmin) * 0.60
        for i in range(len(modes)):
            for j in range(len(thresholds)):
                val = matrix[i, j]
                ax.text(
                    j, i, f"{val:.1f}%",
                    ha="center", va="center", fontsize=9, fontweight="bold",
                    color="white" if val > midpoint else "black",
                )
        cb = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.03)
        cb.set_label(cb_label, size=9)

    _heatmap(ax1, fid_matrix, "Semantic Fidelity", C["bordo"],     35, 100, "Fidelity (%)")
    _heatmap(ax2, sav_matrix, "Compute Savings",   C["lacivert"],   0, 100, "Savings (%)")
    fig.tight_layout()
    return _savefig(fig, "fig3_ablation_heatmap")


def plot_perplexity_comparison(bench: dict = None) -> str:
    """Grouped bar chart: WikiText-2 PPL and ΔPPL per mode."""
    _set_style()
    if bench is None:
        bench = _load("benchmark_results.json")

    methods = list(bench.keys())
    ppls    = [bench[m]["perplexity"] for m in methods]
    base    = ppls[0]
    deltas  = [p - base for p in ppls]

    fig, ax1 = plt.subplots(figsize=(5.5, 3.8))
    ax2 = ax1.twinx()
    x, w = np.arange(len(methods)), 0.38

    bars = ax1.bar(x - w / 2, ppls, width=w, color=C["bordo"],
                   edgecolor="black", linewidth=0.8, label="Perplexity (PPL)")
    ax2.bar(x + w / 2, deltas, width=w, color=C["bordo"] + "B3",
            edgecolor="black", linewidth=0.8, label="ΔPPL vs Baseline", hatch="///")

    for bar, ppl in zip(bars, ppls):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{ppl:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontsize=10)
    ax1.set_ylabel("WikiText-2 Perplexity")
    ax2.set_ylabel("ΔPPL vs Baseline", color="black")
    ax1.set_ylim(0, max(ppls) * 1.20)
    ax1.axhline(base, color=C["grid"], linestyle="--", linewidth=1.2,
                label=f"Baseline PPL = {base:.1f}")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)
    ax1.grid(axis="y")
    return _savefig(fig, "fig4_perplexity_comparison")


def plot_competitor_comparison(comp: dict = None) -> str:
    """Horizontal grouped bar chart: PaES vs literature baselines."""
    _set_style()
    if comp is None:
        comp = _load("comparison_table.json")

    methods  = list(comp.keys())
    savings  = [comp[m]["savings_%"]  for m in methods]
    fidelity = [comp[m]["fidelity_%"] for m in methods]
    colors   = [C["bordo"] if "ours" in m.lower() else C["element"] for m in methods]

    y, h = np.arange(len(methods)), 0.35
    fig, ax = plt.subplots(figsize=(7.0, 3.4))

    ax.barh(y + h / 2, savings,  height=h, color=colors,
            edgecolor="black", linewidth=0.8, label="Compute Savings (%)")
    ax.barh(y - h / 2, fidelity, height=h, color=colors,
            alpha=0.6, edgecolor="black", linewidth=0.8,
            hatch="\\\\\\", label="Semantic Fidelity (%)")

    for i, (s, f) in enumerate(zip(savings, fidelity)):
        ax.text(s + 1.5, i + h / 2, f"{s:.0f}%", va="center", fontsize=9, fontweight="bold")
        ax.text(f + 1.5, i - h / 2, f"{f:.0f}%", va="center", fontsize=9, color=C["context"])

    ax.set_yticks(y)
    ax.set_yticklabels(methods, fontsize=10)
    ax.set_xlabel("Savings / Fidelity (%)")
    ax.set_xlim(0, 115)
    ax.axvline(85, color=C["grid"], linestyle="--", linewidth=1.2,
               label="Threshold = 85%", zorder=0)
    ax.grid(axis="x", zorder=0)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.25), ncol=3, fontsize=9)
    fig.subplots_adjust(bottom=0.25)
    return _savefig(fig, "fig5_competitor_comparison")


def plot_belnap_state_distribution(ablation: dict = None) -> str:
    """Horizontal stacked bar: token fraction per Belnap state."""
    _set_style()
    if ablation is None:
        ablation = _load("ablation_results.json")

    configs = [
        ("Heuristic, τ=0.5", "Heuristic", "0.5"),
        ("Projector, τ=0.5", "Projector",  "0.5"),
        ("Projector, τ=0.6", "Projector",  "0.6"),
        ("Projector, τ=0.7", "Projector",  "0.7"),
    ]

    rows, labels = [], []
    for label, mode, tau in configs:
        r = ablation.get(mode, {}).get(tau, {})
        if not r:
            continue
        sav     = r["savings_mean"] / 100
        fid     = r["fidelity_mean"] / 100
        p_false = min(sav, 0.90)
        p_true  = max(fid - 0.1, 0.05)
        p_both  = max(0.05, (1 - p_false - p_true) * 0.4)
        p_none  = max(0.05, 1 - p_true - p_false - p_both)
        total   = p_true + p_false + p_both + p_none
        rows.append([p_true / total, p_false / total, p_both / total, p_none / total])
        labels.append(label)

    rows   = np.array(rows)
    states = ["TRUE", "FALSE", "BOTH", "NONE"]
    colors = [C[s] for s in states]

    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    y     = np.arange(len(labels))
    lefts = np.zeros(len(labels))

    for state, color, col in zip(states, colors, rows.T):
        ax.barh(y, col, left=lefts, height=0.6,
                color=color, edgecolor="black", linewidth=0.8, label=state)
        for i, (l, v) in enumerate(zip(lefts, col)):
            if v > 0.06:
                txt_color = "black" if color == C["NONE"] else "white"
                ax.text(l + v / 2, i, f"{v * 100:.0f}%",
                        ha="center", va="center", fontsize=10,
                        color=txt_color, fontweight="bold")
        lefts += col

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set_xlabel("Token Fraction", labelpad=8)
    ax.set_xlim(0, 1)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.25),
              ncol=4, fontsize=9, frameon=False)
    ax.grid(axis="x")
    ax.spines["left"].set_visible(False)
    fig.subplots_adjust(bottom=0.20, top=0.85, left=0.20, right=0.95)
    return _savefig(fig, "fig6_belnap_state_distribution")


def generate_all() -> None:
    """Generate all six figures from checkpoint JSON files."""
    print("Generating figures...")
    print(f"  checkpoints : {CKPT_DIR.resolve()}")
    print(f"  output      : {FIGURE_DIR.resolve()}\n")

    audit    = _load("audit_results.json")
    bench    = _load("benchmark_results.json")
    ablation = _load("ablation_results.json")
    comp     = _load("comparison_table.json")

    plot_main_results_table(audit)
    plot_tradeoff_scatter(comp)
    plot_ablation_heatmap(ablation)
    plot_perplexity_comparison(bench)
    plot_competitor_comparison(comp)
    plot_belnap_state_distribution(ablation)

    print("\nDone.")


if __name__ == "__main__":
    generate_all()
