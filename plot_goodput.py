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
parser = argparse.ArgumentParser(description="Goodput analysis")
parser.add_argument("--spec-len", type=int, default=5,
                    help="Fixed speculative length for all configurations (default: 5)")
parser.add_argument("--optimal-K", action="store_true",
                    help="Use optimal K* per configuration instead of fixed spec length")
args = parser.parse_args()

# Label strings for titles and annotations
K_TITLE = "$K^*$" if args.optimal_K else f"$K$={args.spec_len}"
K_TAG   = "$K^*$" if args.optimal_K else "$K$"

# -----------------------------
# Figure size configuration (width, height in inches)
# Adjust these to control aspect ratio for paper layout.
# -----------------------------
FIG_DPI          = 300
FIG_SCATTER      = (7.5, 3.2)    # scatter / tradeoff plots
FIG_BAR          = (11.0, 3.6)   # grouped bar charts

# -----------------------------
# Output directory with timestamp
# -----------------------------
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = os.path.join("results", "goodput_analysis", TIMESTAMP)
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------
# Verification latency (seconds)
#   Goodput(K) = (K·α(K) + 1) / (K/draft_tps + T_VERIFY)
# -----------------------------
T_VERIFY = 0.5

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


# -----------------------------
# Goodput helpers
# -----------------------------
def _compute_goodput(K, alpha, draft_tps, T_verify):
    """Goodput = (K·α + 1) / (K/draft_tps + T_verify)."""
    return (K * alpha + 1.0) / (K / draft_tps + T_verify)


def _find_optimal_K(draft_tps, alpha_curve, T_verify):
    """Return (optimal_K, goodput, alpha_at_optimal_K)."""
    best_K, best_gp, best_alpha = None, -1.0, None
    for K in sorted(alpha_curve.keys()):
        gp = _compute_goodput(K, alpha_curve[K], draft_tps, T_verify)
        if gp > best_gp:
            best_K, best_gp, best_alpha = K, gp, alpha_curve[K]
    return best_K, best_gp, best_alpha


# -----------------------------
# Draft tok/s (RPi4B)
# -----------------------------
SPEED_RPI4 = {
    ("llama_target_70b", "1b", "q4_k_m", "inst"): 4.14,
    ("llama_target_70b", "1b", "q4_k_m", "base"):     4.18,
    ("llama_target_70b", "3b", "q4_k_m", "inst"): 1.68,
    ("llama_target_70b", "3b", "q8_0",   "inst"): 1.04,
    ("llama_target_70b", "8b", "q4_k_m", "inst"): 0.72,

    ("qwen_target_32b",  "0.6b",   "q4_k_m", "base"):     7.84,
    ("qwen_target_32b",  "0.6b",   "q8_0",   "base"):     4.92,
    ("qwen_target_32b",  "1.7b",   "q4_k_m", "base"):     3.02,
    ("qwen_target_32b",  "1.7b",   "q8_0",   "base"):     1.89,
    ("qwen_target_32b",  "4b",     "q4_k_m", "base"):     1.34,
    ("qwen_target_32b",  "4b",     "q8_0",   "base"):     0.83,
    ("qwen_target_32b",  "8b",     "q4_k_m", "base"):     0.72,
    ("qwen_target_32b",  "8b",     "q6_k",   "base"):     0.55,
}

# Draft tok/s (RPi5)
SPEED_RPI5 = {
    ("llama_target_70b", "1b", "q4_k_m", "inst"): 14.47,
    ("llama_target_70b", "1b", "q4_k_m", "base"):     12.86,
    ("llama_target_70b", "3b", "q4_k_m", "inst"): 4.68,
    ("llama_target_70b", "3b", "q8_0",   "inst"): 2.37,
    ("llama_target_70b", "8b", "q4_k_m", "inst"): 1.77,

    ("qwen_target_32b",  "0.6b",   "q4_k_m", "base"):     18.18,
    ("qwen_target_32b",  "0.6b",   "q8_0",   "base"):     10.38,
    ("qwen_target_32b",  "1.7b",   "q4_k_m", "base"):     7.15,
    ("qwen_target_32b",  "1.7b",   "q8_0",   "base"):     4.01,
    ("qwen_target_32b",  "4b",     "q4_k_m", "base"):     3.11,
    ("qwen_target_32b",  "4b",     "q8_0",   "base"):     1.93,
    ("qwen_target_32b",  "8b",     "q4_k_m", "base"):     1.78,
    ("qwen_target_32b",  "8b",     "q6_k",   "base"):     1.68,
}

# Draft tok/s (JETSON)
SPEED_JETSON = {
    ("llama_target_70b", "1b", "q4_k_m", "inst"): 93.14,
    ("llama_target_70b", "1b", "q4_k_m", "base"):     92.94,
    ("llama_target_70b", "3b", "q4_k_m", "inst"): 42.52,
    ("llama_target_70b", "3b", "q8_0",   "inst"): 31.36,
    ("llama_target_70b", "8b", "q4_k_m", "inst"): 25.07,

    ("qwen_target_32b",  "0.6b",   "q4_k_m", "base"):     98.63,
    ("qwen_target_32b",  "0.6b",   "q8_0",   "base"):     80.21,
    ("qwen_target_32b",  "1.7b",   "q4_k_m", "base"):     65.31,
    ("qwen_target_32b",  "1.7b",   "q8_0",   "base"):     46.64,
    ("qwen_target_32b",  "4b",     "q4_k_m", "base"):     33.70,
    ("qwen_target_32b",  "4b",     "q8_0",   "base"):     23.02,
    ("qwen_target_32b",  "8b",     "q4_k_m", "base"):     24.03,
    ("qwen_target_32b",  "8b",     "q6_k",   "base"):     18.55,
}
# -----------------------------
# 2) Build dataframe
# -----------------------------
rows = []
def add_rows(device_name, speed_dict):
    for (target, model, quant, flavor), tps in speed_dict.items():
        hf_key = DRAFT_ACC_MAP.get((target, model, flavor))
        if hf_key is None:
            continue
        alpha_curve = ACC_CURVES.get(hf_key)
        if alpha_curve is None:
            continue
        if args.optimal_K:
            opt_K, gp, alpha = _find_optimal_K(float(tps), alpha_curve, T_VERIFY)
        else:
            opt_K = args.spec_len
            alpha = alpha_curve.get(opt_K)
            if alpha is None:
                continue
            gp = _compute_goodput(opt_K, alpha, float(tps), T_VERIFY)
        rows.append({
            "target": target,
            "device": device_name,
            "model": model,
            "quant": quant,
            "flavor": flavor,
            "draft_tps": float(tps),
            "accept": float(alpha),
            "optimal_K": int(opt_K),
            "goodput": float(gp),
            "variant": f"{model}-{flavor}-{quant}",
        })

add_rows("RPi 4B", SPEED_RPI4)
add_rows("RPi 5",  SPEED_RPI5)
add_rows("Jetson AGX Orin", SPEED_JETSON)
df = pd.DataFrame(rows)

# Optional: export table for LaTeX/appendix
df.to_csv(os.path.join(OUT_DIR, "goodput_table.csv"), index=False)

# -----------------------------
# 3) Global style – research-paper quality
# -----------------------------
DEVICE_ORDER = ["RPi 4B", "RPi 5", "Jetson AGX Orin"]

# Colorblind-friendly palette (Okabe-Ito inspired)
DEVICE_COLORS = {
    "RPi 4B":          "#0072B2",   # blue
    "RPi 5":           "#D55E00",   # vermillion
    "Jetson AGX Orin": "#009E73",   # green
}
DEVICE_MARKERS = {
    "RPi 4B":          "^",
    "RPi 5":           "o",
    "Jetson AGX Orin": "s",
}
DEVICE_HATCHES = {
    "RPi 4B":          "//",
    "RPi 5":           "\\\\",
    "Jetson AGX Orin": "xx",
}

def _paper_rc():
    """Apply global rcParams for a clean, publication-ready look."""
    mpl.rcParams.update({
        "font.family":       "serif",
        "font.serif":        ["Times New Roman", "DejaVu Serif", "serif"],
        "mathtext.fontset":  "stix",
        "font.size":         15,
        "axes.labelsize":    14,
        "axes.titlesize":    18,
        "axes.titleweight":  "bold",
        "legend.fontsize":   15,
        "xtick.labelsize":   15,
        "ytick.labelsize":   15,
        "xtick.direction":   "in",
        "ytick.direction":   "in",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "axes.linewidth":    0.8,
        "pdf.fonttype":      42,
        "ps.fonttype":       42,
        "savefig.bbox":      "tight",
        "savefig.pad_inches": 0.05,
    })


# -----------------------------
# 4) Plot helpers
# -----------------------------
def plot_tradeoff(df_sub: pd.DataFrame, title: str, out_prefix: str):
    _paper_rc()

    fig, ax = plt.subplots(figsize=FIG_SCATTER, dpi=FIG_DPI)

    # Reference K for iso-goodput curves (median optimal K in this subset)
    K_ref = int(np.median(df_sub["optimal_K"]))

    # --- iso-goodput contour curves (new model) ---
    # G = (K_ref·α + 1) / (K_ref/tps + T_VERIFY)
    # ⟹ tps = K_ref·G / (K_ref·α + 1 − G·T_VERIFY)
    x = np.linspace(0.25, 0.80, 300)
    iso_values = [0.5, 1, 2, 4, 8, 16, 32, 64]
    y_max = df_sub["draft_tps"].max() * 1.6
    for G in iso_values:
        denom = K_ref * x + 1.0 - G * T_VERIFY
        y = np.where(denom > 0, K_ref * G / denom, np.nan)
        ax.plot(x, y, linestyle="--", linewidth=0.7, color="#888888", alpha=0.45)
        # place label where curve is still visible
        xr = 0.68
        dr = K_ref * xr + 1.0 - G * T_VERIFY
        if dr > 0:
            yr = K_ref * G / dr
            if 0 < yr < y_max:
                ax.text(xr + 0.005, yr, f"$G$={G}", fontsize=8, color="#666666",
                        va="bottom", ha="left")

    # --- scatter by device ---
    for dev in DEVICE_ORDER:
        g = df_sub[df_sub["device"] == dev]
        if g.empty:
            continue
        ax.scatter(
            g["accept"], g["draft_tps"],
            marker=DEVICE_MARKERS[dev],
            color=DEVICE_COLORS[dev],
            edgecolors="white",
            linewidths=0.5,
            s=60,
            alpha=0.92,
            zorder=5,
            label=dev,
        )

    # --- annotate best point per device ---
    for dev in DEVICE_ORDER:
        g = df_sub[df_sub["device"] == dev]
        if g.empty:
            continue
        best = g.loc[g["goodput"].idxmax()]
        ax.annotate(
            f"{best['model']}\n{best['quant']}  "
            f"$G$={best['goodput']:.1f}  {K_TAG}={best['optimal_K']}",
            xy=(best["accept"], best["draft_tps"]),
            xytext=(12, 10),
            textcoords="offset points",
            fontsize=8,
            color=DEVICE_COLORS[dev],
            arrowprops=dict(arrowstyle="-|>", color=DEVICE_COLORS[dev],
                            lw=0.9, shrinkA=0, shrinkB=3),
        )

    ax.set_xlabel(f"Acceptance Rate  $\\alpha({K_TAG[1:-1]})$")
    ax.set_ylabel("Draft Decoding Speed  (tok/s)")
    ax.set_title(title, pad=10)
    ax.set_yscale("log")

    # light grid
    ax.grid(True, which="major", linestyle="-", linewidth=0.4, alpha=0.35)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.3, alpha=0.20)

    # tidy spines
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax.legend(loc="upper left", frameon=True, fancybox=False,
              edgecolor="#cccccc", framealpha=0.95)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"{out_prefix}_tradeoff.pdf"))
    fig.savefig(os.path.join(OUT_DIR, f"{out_prefix}_tradeoff.png"))
    plt.close(fig)


def _draw_grouped_bar(ax, df_sub: pd.DataFrame, title: str, show_legend: bool = True):
    """Draw a grouped bar chart on the given axes."""
    # pivot to device columns
    pivot = df_sub.pivot_table(
        index="variant", columns="device", values="goodput", aggfunc="mean"
    ).fillna(0.0)

    # reorder columns to match DEVICE_ORDER
    ordered_cols = [d for d in DEVICE_ORDER if d in pivot.columns]
    pivot = pivot[ordered_cols]

    # sort by Jetson (highest dynamic range) then RPi 5
    sort_col = "Jetson AGX Orin" if "Jetson AGX Orin" in pivot.columns else ordered_cols[-1]
    pivot = pivot.sort_values(by=sort_col, ascending=False)

    n_variants = len(pivot.index)
    n_devices = len(ordered_cols)
    width = 0.75 / n_devices
    x = np.arange(n_variants)

    for i, dev in enumerate(ordered_cols):
        offset = (i - (n_devices - 1) / 2) * width
        ax.bar(
            x + offset,
            pivot[dev].values,
            width,
            label=dev,
            color=DEVICE_COLORS[dev],
            edgecolor="white",
            linewidth=0.6,
            hatch=DEVICE_HATCHES[dev],
            alpha=0.88,
            zorder=3,
        )

    ax.set_ylabel("Goodput  (tok/s)")
    ax.set_title(title, pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=38, ha="right")

    # light horizontal grid behind bars
    ax.grid(True, axis="y", linestyle="-", linewidth=0.4, alpha=0.35, zorder=0)
    ax.set_axisbelow(True)

    # tidy spines
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    if show_legend:
        ax.legend(loc="upper right", frameon=True, fancybox=False,
                  edgecolor="#cccccc", framealpha=0.45, ncol=n_devices)

# -----------------------------
# 5) Generate plots (split by target family)
# -----------------------------
TARGET_ORDER = ["llama_target_70b", "qwen_target_32b"]
BAR_TITLES = {
    "llama_target_70b": f"Target: Llama-3.1-70B  ({K_TITLE}, $T_{{\\mathrm{{verify}}}}$={T_VERIFY}s)",
    "qwen_target_32b":  f"Target: Qwen3-32B  ({K_TITLE}, $T_{{\\mathrm{{verify}}}}$={T_VERIFY}s)",
}

# Scatter tradeoff plots (one per target, unchanged)
for target, g in df.groupby("target"):
    title = {
        "llama_target_70b": f"Speed–Acceptance Tradeoff  (Target: Llama-3.1-70B, {K_TITLE}, $T_{{\\mathrm{{verify}}}}$={T_VERIFY}s)",
        "qwen_target_32b":  f"Speed–Acceptance Tradeoff  (Target: Qwen3-32B, {K_TITLE}, $T_{{\\mathrm{{verify}}}}$={T_VERIFY}s)",
    }.get(target, f"Tradeoff ({target})")
    plot_tradeoff(g, title, target)

# Combined grouped bar chart (two rows, one per target)
_paper_rc()
fig, axes = plt.subplots(
    2, 1, figsize=(FIG_BAR[0], FIG_BAR[1] * 2),
    dpi=FIG_DPI, constrained_layout=True,
)

for row, target in enumerate(TARGET_ORDER):
    g = df[df["target"] == target]
    bar_title = BAR_TITLES.get(target, f"Goodput ({target})")
    _draw_grouped_bar(axes[row], g, bar_title, show_legend=(row == 0))

fig.savefig(os.path.join(OUT_DIR, "goodput_bar.pdf"))
fig.savefig(os.path.join(OUT_DIR, "goodput_bar.png"))
plt.close(fig)

print(f"Done. All outputs saved to {OUT_DIR}/")
