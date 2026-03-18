import os
import argparse
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# -----------------------------
# Command-line arguments
# -----------------------------
parser = argparse.ArgumentParser(description="Cost efficiency analysis")
parser.add_argument("--spec-len", type=int, default=5,
                    help="Fixed speculative length for all configurations (default: 5)")
parser.add_argument("--optimal-K", action="store_true",
                    help="Use optimal K* per configuration instead of fixed spec length")
parser.add_argument("--fig-width", type=float, default=7.0,
                    help="Figure width in inches (default: 7.0)")
parser.add_argument("--fig-height", type=float, default=2.8,
                    help="Figure height in inches (default: 2.8)")
args = parser.parse_args()

# Label strings for titles and annotations
K_TITLE = "$K^*$" if args.optimal_K else f"$K$={args.spec_len}"
K_TAG   = "$K^*$" if args.optimal_K else "$K$"

# -----------------------------
# Figure size configuration (width, height in inches)
# Adjust these to control aspect ratio for paper layout.
# -----------------------------
FIG_DPI          = 300
FIG_BAR          = (args.fig_width, args.fig_height)

# -----------------------------
# Output directory with timestamp
# -----------------------------
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = os.path.join("results", "cost_efficiency_analysis", TIMESTAMP)
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------
# 1) Load acceptance-rate sweep data
# -----------------------------
_acc_qwen  = pd.read_csv("results/profile_results_2026-02-20_11-42-25.csv")
_acc_llama = pd.read_csv("results/profile_results_2026-02-20_13-30-21.csv")
_acc_raw   = pd.concat([_acc_qwen, _acc_llama], ignore_index=True)

# Build lookup: (target_hf, draft_hf) → {K: α(K)}
ACC_CURVES = {}
for _, r in _acc_raw.iterrows():
    key = (r["target_model"], r["draft_model"])
    if key not in ACC_CURVES:
        ACC_CURVES[key] = {}
    ACC_CURVES[key][int(r["spec_len"])] = float(r["mean_acceptance_rate"])

# Mapping: (target_key, model_short, flavor) → (target_hf, draft_hf)
DRAFT_ACC_MAP = {
    ("llama_target_70b", "1b", "base"): (
        "meta-llama/Meta-Llama-3.1-70B-Instruct", "meta-llama/Llama-3.2-1B"),
    ("llama_target_70b", "1b", "inst"): (
        "meta-llama/Meta-Llama-3.1-70B-Instruct", "meta-llama/Llama-3.2-1B-Instruct"),
    ("llama_target_70b", "3b", "inst"): (
        "meta-llama/Meta-Llama-3.1-70B-Instruct", "meta-llama/Llama-3.2-3B-Instruct"),
    ("llama_target_70b", "8b", "inst"): (
        "meta-llama/Meta-Llama-3.1-70B-Instruct", "meta-llama/Meta-Llama-3.1-8B-Instruct"),
    ("qwen_target_32b", "0.6b", "base"): (
        "Qwen/Qwen3-32B", "Qwen/Qwen3-0.6B"),
    ("qwen_target_32b", "1.7b", "base"): (
        "Qwen/Qwen3-32B", "Qwen/Qwen3-1.7B"),
    ("qwen_target_32b", "4b", "base"): (
        "Qwen/Qwen3-32B", "Qwen/Qwen3-4B"),
    ("qwen_target_32b", "8b", "base"): (
        "Qwen/Qwen3-32B", "Qwen/Qwen3-8B"),
}

# Map hf_model → (model_short, flavor) for DRAFT_ACC_MAP lookup
HF_TO_DRAFT = {
    "llama-3.2-1b":          ("1b", "base"),
    "llama-3.2-1b-instruct": ("1b", "inst"),
    "llama-3.2-3b":          ("3b", "inst"),
    "llama-3.1-8b":          ("8b", "inst"),
    "qwen3-0.6b":            ("0.6b", "base"),
    "qwen3-1.7b":            ("1.7b", "base"),
    "qwen3-4b":              ("4b", "base"),
    "qwen3-8b":              ("8b", "base"),
}


def _find_optimal_K_cost(alpha_curve):
    """Return (optimal_K, accepted_per_verified, alpha_at_K).

    Maximises (K·α(K)+1)/K = α(K)+1/K  (accepted tokens per verified token).
    """
    best_K, best_ratio, best_alpha = None, -1.0, None
    for K in sorted(alpha_curve.keys()):
        ratio = (K * alpha_curve[K] + 1.0) / K
        if ratio > best_ratio:
            best_K, best_ratio, best_alpha = K, ratio, alpha_curve[K]
    return best_K, best_ratio, best_alpha

# -----------------------------
# 2) Verifier pricing (output tokens, $/1M tokens)
# -----------------------------
PRICE_OUT_PER_M = {
    "llama_target_70b": 0.90,   # Fireworks tier >16B (proxy for 70B-class)
    "qwen_target_32b":  0.59,   # Groq Qwen3-32B output
}

# -----------------------------
# 3) Derive cost efficiency
#
#    tokens_per_$ = (K·α(K) + 1) / K × 1e6 / price_per_M
#                 = (α(K) + 1/K) × 1e6 / price_per_M
#
#    Optimal K* maximises accepted tokens per verified token: (K·α(K)+1)/K.
#    draft_tps cancels ⇒ metric is INDEPENDENT of device and quantization.
# -----------------------------
DRAFT_HF_PAIRS = [
    ("llama_target_70b", "llama-3.2-1b"),
    ("llama_target_70b", "llama-3.2-1b-instruct"),
    ("llama_target_70b", "llama-3.2-3b"),
    ("llama_target_70b", "llama-3.1-8b"),
    ("qwen_target_32b",  "qwen3-0.6b"),
    ("qwen_target_32b",  "qwen3-1.7b"),
    ("qwen_target_32b",  "qwen3-4b"),
    ("qwen_target_32b",  "qwen3-8b"),
]

rows = []
for target, hf_model in DRAFT_HF_PAIRS:
    model_short, flavor = HF_TO_DRAFT[hf_model]
    hf_key = DRAFT_ACC_MAP[(target, model_short, flavor)]
    alpha_curve = ACC_CURVES[hf_key]
    if args.optimal_K:
        opt_K, accepted_per_verified, alpha = _find_optimal_K_cost(alpha_curve)
    else:
        opt_K = args.spec_len
        alpha = alpha_curve.get(opt_K)
        if alpha is None:
            continue
        accepted_per_verified = (opt_K * alpha + 1.0) / opt_K

    price_per_M = PRICE_OUT_PER_M[target]
    tokens_per_dollar = accepted_per_verified * 1_000_000 / price_per_M
    dollars_per_1k = 1000.0 / tokens_per_dollar
    rows.append({
        "target": target,
        "model": hf_model,
        "accept": alpha,
        "optimal_K": int(opt_K),
        "price_per_M_output": price_per_M,
        "tokens_per_$": tokens_per_dollar,
        "$_per_1k_accepted": dollars_per_1k,
    })

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT_DIR, "cost_efficiency_by_model.csv"), index=False)

# -----------------------------
# 4) Style (aligned with plot_goodput.py)
# -----------------------------
TARGET_DISPLAY = {
    "llama_target_70b": "Llama-3.1-70B",
    "qwen_target_32b":  "Qwen3-32B",
}

MODEL_DISPLAY = {
    "llama-3.2-1b":          "Llama-3.2-1B",
    "llama-3.2-1b-instruct": "Llama-3.2-1B-Instruct",
    "llama-3.2-3b":          "Llama-3.2-3B",
    "llama-3.1-8b":          "Llama-3.1-8B",
    "qwen3-0.6b":   "Qwen3-0.6B",
    "qwen3-1.7b":   "Qwen3-1.7B",
    "qwen3-4b":     "Qwen3-4B",
    "qwen3-8b":     "Qwen3-8B",
}

# Sequential palettes per target (light → dark = smaller → larger draft model)
TARGET_PALETTES = {
    "llama_target_70b": ["#a6cee3", "#6baed6", "#1f78b4", "#08306b"],
    "qwen_target_32b":  ["#b2df8a", "#33a02c", "#006d2c", "#00441b"],
}
HATCHES = ["//", "\\\\", "xx", ".."]


def _paper_rc():
    """Apply global rcParams — identical to plot_goodput.py."""
    mpl.rcParams.update({
        "font.family":        "serif",
        "font.serif":         ["Times New Roman", "DejaVu Serif", "serif"],
        "mathtext.fontset":   "stix",
        "font.size":          10,
        "axes.labelsize":     12,
        "axes.titlesize":     13,
        "axes.titleweight":   "bold",
        "legend.fontsize":    10,
        "xtick.labelsize":    10,
        "ytick.labelsize":    10,
        "xtick.direction":    "in",
        "ytick.direction":    "in",
        "xtick.major.width":  0.8,
        "ytick.major.width":  0.8,
        "axes.linewidth":     0.8,
        "pdf.fonttype":       42,
        "ps.fonttype":        42,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.05,
    })


# -----------------------------
# 5) Combined bar chart (two sub-figures, shared y-axis)
# -----------------------------
TARGET_ORDER = ["llama_target_70b", "qwen_target_32b"]

_paper_rc()
fig, axes = plt.subplots(
    1, 2, figsize=FIG_BAR, dpi=FIG_DPI,
    sharey=True,
    constrained_layout=True,
    gridspec_kw={"wspace": 0.02},
)

global_max = df["tokens_per_$"].max()

for idx, target in enumerate(TARGET_ORDER):
    ax = axes[idx]
    g = df[df["target"] == target].copy()
    g = g.sort_values("accept", ascending=True).reset_index(drop=True)

    models = [MODEL_DISPLAY.get(m, m) for m in g["model"]]
    values = g["tokens_per_$"].values
    accs   = g["accept"].values
    palette = TARGET_PALETTES[target]
    display_name = TARGET_DISPLAY[target]

    n = len(models)
    x = np.arange(n)

    bars = ax.bar(
        x, values, width=0.55,
        color=[palette[i] for i in range(n)],
        edgecolor="white", linewidth=0.6,
        hatch=[HATCHES[i] for i in range(n)],
        alpha=0.88, zorder=3,
    )

    # Value labels on top of each bar
    y_pad = global_max * 0.015
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + y_pad,
            f"{val / 1e3:.0f}K",
            ha="center", va="bottom", fontsize=8,
        )

    if idx == 0:
        ax.set_ylabel("Accepted Tokens per \\$")
    ax.set_title(f"Target: {display_name}", pad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")

    # Light horizontal grid behind bars
    ax.grid(True, axis="y", linestyle="-", linewidth=0.4, alpha=0.35, zorder=0)
    ax.set_axisbelow(True)

    # Tidy spines
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

# Shared y-axis formatting
axes[0].yaxis.set_major_formatter(
    plt.FuncFormatter(lambda v, _: f"{v / 1e3:.0f}K")
)
axes[0].set_ylim(0, global_max * 1.15)

fig.suptitle(
    f"Cost Efficiency of Draft Models  ({K_TITLE})",
    fontsize=13, fontweight="bold", y=1.07,
)
fig.savefig(os.path.join(OUT_DIR, "cost_efficiency.pdf"))
fig.savefig(os.path.join(OUT_DIR, "cost_efficiency.png"))
plt.close(fig)

print(f"Done. All outputs saved to {OUT_DIR}/")
