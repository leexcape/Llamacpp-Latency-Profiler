import os
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# -----------------------------
# Output directory with timestamp
# -----------------------------
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = os.path.join("results", "cost_efficiency_analysis", TIMESTAMP)
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------
# 1) Acceptance rates (per HF model pair — device / quant independent)
# -----------------------------
ACC = {
    # Target: Meta-Llama-3.1-70B
    ("llama_target_70b", "llama-3.2-1b"): 0.5038,
    ("llama_target_70b", "llama-3.2-1b-instruct"): 0.5102,
    ("llama_target_70b", "llama-3.2-3b"): 0.5691,
    ("llama_target_70b", "llama-3.1-8b"): 0.6488,

    # Target: Qwen3-32B
    ("qwen_target_32b", "qwen3-0.6b"): 0.3826,
    ("qwen_target_32b", "qwen3-1.7b"): 0.4494,
    ("qwen_target_32b", "qwen3-4b"):  0.4705,
    ("qwen_target_32b", "qwen3-8b"):  0.5357,
}

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
#    tokens_per_$ = (draft_tps × acc) / (price_per_token × draft_tps)
#                 = acc / price_per_token
#                 = acc × 1e6 / price_per_M
#
#    draft_tps cancels ⇒ metric is INDEPENDENT of device and quantization.
# -----------------------------
rows = []
for (target, model), acc in ACC.items():
    price_per_M = PRICE_OUT_PER_M[target]
    tokens_per_dollar = acc * 1_000_000 / price_per_M
    dollars_per_1k = 1000.0 / tokens_per_dollar
    rows.append({
        "target": target,
        "model": model,
        "accept": acc,
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
        "axes.labelsize":     11,
        "axes.titlesize":     12,
        "axes.titleweight":   "bold",
        "legend.fontsize":    9,
        "xtick.labelsize":    10,
        "ytick.labelsize":    9,
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
# 5) Per-target bar charts
# -----------------------------
for target, g in df.groupby("target"):
    _paper_rc()

    # Sort by acceptance rate (ascending) = natural model-size order
    g = g.sort_values("accept", ascending=True).reset_index(drop=True)

    models = [MODEL_DISPLAY.get(m, m) for m in g["model"]]
    values = g["tokens_per_$"].values
    accs   = g["accept"].values
    palette = TARGET_PALETTES[target]
    display_name = TARGET_DISPLAY[target]

    n = len(models)
    x = np.arange(n)

    fig, ax = plt.subplots(figsize=(5.5, 4.2), dpi=300)

    bars = ax.bar(
        x, values, width=0.55,
        color=[palette[i] for i in range(n)],
        edgecolor="white", linewidth=0.6,
        hatch=[HATCHES[i] for i in range(n)],
        alpha=0.88, zorder=3,
    )

    # Value labels on top of each bar
    y_pad = max(values) * 0.015
    for bar, val, acc in zip(bars, values, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + y_pad,
            f"{val / 1e3:.0f}K\n($\\alpha$={acc:.2f})",
            ha="center", va="bottom", fontsize=8,
        )

    ax.set_ylabel("Accepted Tokens per \\$  [higher is better]")
    ax.set_title(
        f"Cost Efficiency of Draft Models  (Target: {display_name})", pad=10
    )
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")

    # Format y-axis in thousands
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v / 1e3:.0f}K")
    )

    # Expand ylim to fit annotations
    ax.set_ylim(0, max(values) * 1.28)

    # Light horizontal grid behind bars
    ax.grid(True, axis="y", linestyle="-", linewidth=0.4, alpha=0.35, zorder=0)
    ax.set_axisbelow(True)

    # Tidy spines
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"{target}_cost_efficiency.pdf"))
    fig.savefig(os.path.join(OUT_DIR, f"{target}_cost_efficiency.png"))
    plt.close(fig)

print(f"Done. All outputs saved to {OUT_DIR}/")
