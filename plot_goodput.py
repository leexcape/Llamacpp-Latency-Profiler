import math
import os
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

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
# 1) Raw inputs (from your screenshots)
# -----------------------------
ACC = {
    # Target: Meta-Llama-3.1-70B
    ("llama_target_70b", "1b"): 0.5038,
    ("llama_target_70b", "3b"): 0.5691,
    ("llama_target_70b", "8b"): 0.6488,

    # Target: Qwen3-32B
    ("qwen_target_32b", "0.6b"): 0.3826,
    ("qwen_target_32b", "1.7b"): 0.4494,
    ("qwen_target_32b", "4b"):   0.4705,
    ("qwen_target_32b", "8b"):   0.5357,
}

# Draft tok/s (RPi4B)
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
        acc = ACC.get((target, model))
        if acc is None:
            continue
        rows.append({
            "target": target,
            "device": device_name,
            "model": model,
            "quant": quant,
            "flavor": flavor,
            "draft_tps": float(tps),
            "accept": float(acc),
            "goodput_proxy": float(tps) * float(acc),
            "variant": f"{model}-{flavor}-{quant}",
        })

add_rows("RPi 4B", SPEED_RPI4)
add_rows("RPi 5",  SPEED_RPI5)
add_rows("Jetson AGX Orin", SPEED_JETSON)
df = pd.DataFrame(rows)

# Optional: export table for LaTeX/appendix
df.to_csv(os.path.join(OUT_DIR, "goodput_proxy_table.csv"), index=False)

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
        "font.size":         10,
        "axes.labelsize":    14,
        "axes.titlesize":    15,
        "axes.titleweight":  "bold",
        "legend.fontsize":   10,
        "xtick.labelsize":   12,
        "ytick.labelsize":   12,
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

    # --- iso-goodput contour curves: y = G / x ---
    x = np.linspace(0.25, 0.70, 300)
    iso_values = [0.5, 1, 2, 4, 8, 16, 32, 64]
    y_max = df_sub["draft_tps"].max() * 1.6
    for G in iso_values:
        y = G / x
        ax.plot(x, y, linestyle="--", linewidth=0.7, color="#888888", alpha=0.45)
        # place label where curve is still visible
        xr = 0.68
        yr = G / xr
        if yr < y_max:
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
        best = g.loc[g["goodput_proxy"].idxmax()]
        ax.annotate(
            f"{best['model']}\n{best['quant']}  $G$={best['goodput_proxy']:.1f}",
            xy=(best["accept"], best["draft_tps"]),
            xytext=(12, 10),
            textcoords="offset points",
            fontsize=8,
            color=DEVICE_COLORS[dev],
            arrowprops=dict(arrowstyle="-|>", color=DEVICE_COLORS[dev],
                            lw=0.9, shrinkA=0, shrinkB=3),
        )

    ax.set_xlabel("Acceptance Rate  (accepted / proposed)")
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


def plot_grouped_bar(df_sub: pd.DataFrame, title: str, out_prefix: str):
    _paper_rc()

    # pivot to device columns
    pivot = df_sub.pivot_table(
        index="variant", columns="device", values="goodput_proxy", aggfunc="mean"
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

    fig, ax = plt.subplots(figsize=FIG_BAR, dpi=FIG_DPI)

    for i, dev in enumerate(ordered_cols):
        offset = (i - (n_devices - 1) / 2) * width
        bars = ax.bar(
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

    ax.set_ylabel("Goodput Proxy  (tok/s)")
    ax.set_title(title, pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=38, ha="right")

    # light horizontal grid behind bars
    ax.grid(True, axis="y", linestyle="-", linewidth=0.4, alpha=0.35, zorder=0)
    ax.set_axisbelow(True)

    # tidy spines
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax.legend(loc="upper right", frameon=True, fancybox=False,
              edgecolor="#cccccc", framealpha=0.95, ncol=n_devices)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"{out_prefix}_bar.pdf"))
    fig.savefig(os.path.join(OUT_DIR, f"{out_prefix}_bar.png"))
    plt.close(fig)

# -----------------------------
# 5) Generate plots (split by target family)
# -----------------------------
for target, g in df.groupby("target"):
    title = {
        "llama_target_70b": "Speed–Acceptance Tradeoff  (Target: Llama-3.1-70B)",
        "qwen_target_32b":  "Speed–Acceptance Tradeoff  (Target: Qwen3-32B)",
    }.get(target, f"Tradeoff ({target})")
    out = target
    plot_tradeoff(g, title, out)
    bar_title = {
        "llama_target_70b": "Goodput Proxy Comparison  (Target: Llama-3.1-70B)",
        "qwen_target_32b":  "Goodput Proxy Comparison  (Target: Qwen3-32B)",
    }.get(target, f"Goodput proxy ({target})")
    plot_grouped_bar(g, bar_title, out)

print(f"Done. All outputs saved to {OUT_DIR}/")
