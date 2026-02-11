import math
import os
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
    ("llama_target_70b", "llama-3.2-1b"): 0.5038,
    ("llama_target_70b", "llama-3.2-3b"): 0.5691,
    ("llama_target_70b", "llama-3.1-8b"): 0.6488,

    # Target: Qwen3-32B
    ("qwen_target_32b", "qwen3-0.6b"): 0.3826,
    ("qwen_target_32b", "qwen3-1.7b"): 0.4494,
    ("qwen_target_32b", "qwen3-4b"): 0.4705,
    ("qwen_target_32b", "qwen3-8b"): 0.5357,
}

# Draft tok/s (RPi4B)
SPEED_RPI4 = {
    ("llama_target_70b", "llama-3.2-1b", "q4_k_m", "instruct"): 4.14,
    ("llama_target_70b", "llama-3.2-1b", "q4_k_m", "base"):     4.18,
    ("llama_target_70b", "llama-3.2-3b", "q4_k_m", "instruct"): 1.68,
    ("llama_target_70b", "llama-3.2-3b", "q8_0",   "instruct"): 1.04,
    ("llama_target_70b", "llama-3.1-8b", "q4_k_m", "instruct"): 0.72,

    ("qwen_target_32b",  "qwen3-0.6b",   "q4_k_m", "base"):     7.84,
    ("qwen_target_32b",  "qwen3-0.6b",   "q8_0",   "base"):     4.92,
    ("qwen_target_32b",  "qwen3-1.7b",   "q4_k_m", "base"):     3.02,
    ("qwen_target_32b",  "qwen3-1.7b",   "q8_0",   "base"):     1.89,
    ("qwen_target_32b",  "qwen3-4b",     "q4_k_m", "base"):     1.34,
    ("qwen_target_32b",  "qwen3-4b",     "q8_0",   "base"):     0.83,
    ("qwen_target_32b",  "qwen3-8b",     "q4_k_m", "base"):     0.72,
    ("qwen_target_32b",  "qwen3-8b",     "q6_k",   "base"):     0.55,
}

# Draft tok/s (RPi5)
SPEED_RPI5 = {
    ("llama_target_70b", "llama-3.2-1b", "q4_k_m", "instruct"): 14.47,
    ("llama_target_70b", "llama-3.2-1b", "q4_k_m", "base"):     12.86,
    ("llama_target_70b", "llama-3.2-3b", "q4_k_m", "instruct"): 4.68,
    ("llama_target_70b", "llama-3.2-3b", "q8_0",   "instruct"): 2.37,
    ("llama_target_70b", "llama-3.1-8b", "q4_k_m", "instruct"): 1.77,

    ("qwen_target_32b",  "qwen3-0.6b",   "q4_k_m", "base"):     18.18,
    ("qwen_target_32b",  "qwen3-0.6b",   "q8_0",   "base"):     10.38,
    ("qwen_target_32b",  "qwen3-1.7b",   "q4_k_m", "base"):     7.15,
    ("qwen_target_32b",  "qwen3-1.7b",   "q8_0",   "base"):     4.01,
    ("qwen_target_32b",  "qwen3-4b",     "q4_k_m", "base"):     3.11,
    ("qwen_target_32b",  "qwen3-4b",     "q8_0",   "base"):     1.93,
    ("qwen_target_32b",  "qwen3-8b",     "q4_k_m", "base"):     1.78,
    ("qwen_target_32b",  "qwen3-8b",     "q6_k",   "base"):     1.68,
}

# Draft tok/s (JETSON)
SPEED_JETSON = {
    ("llama_target_70b", "llama-3.2-1b", "q4_k_m", "instruct"): 93.14,
    ("llama_target_70b", "llama-3.2-1b", "q4_k_m", "base"):     92.94,
    ("llama_target_70b", "llama-3.2-3b", "q4_k_m", "instruct"): 42.52,
    ("llama_target_70b", "llama-3.2-3b", "q8_0",   "instruct"): 31.36,
    ("llama_target_70b", "llama-3.1-8b", "q4_k_m", "instruct"): 25.07,

    ("qwen_target_32b",  "qwen3-0.6b",   "q4_k_m", "base"):     98.63,
    ("qwen_target_32b",  "qwen3-0.6b",   "q8_0",   "base"):     80.21,
    ("qwen_target_32b",  "qwen3-1.7b",   "q4_k_m", "base"):     65.31,
    ("qwen_target_32b",  "qwen3-1.7b",   "q8_0",   "base"):     46.64,
    ("qwen_target_32b",  "qwen3-4b",     "q4_k_m", "base"):     33.70,
    ("qwen_target_32b",  "qwen3-4b",     "q8_0",   "base"):     23.02,
    ("qwen_target_32b",  "qwen3-8b",     "q4_k_m", "base"):     24.03,
    ("qwen_target_32b",  "qwen3-8b",     "q6_k",   "base"):     18.55,
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

add_rows("RPi4B", SPEED_RPI4)
add_rows("RPi5",  SPEED_RPI5)
df = pd.DataFrame(rows)

# Optional: export table for LaTeX/appendix
df.to_csv(os.path.join(OUT_DIR, "goodput_proxy_table.csv"), index=False)

# -----------------------------
# 3) Plot helpers
# -----------------------------
def plot_tradeoff(df_sub: pd.DataFrame, title: str, out_prefix: str):
    # Matplotlib defaults are fine; keep it paper-ish via sizing and fonts
    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(6.6, 4.2), dpi=300)

    # iso-goodput curves: y = G / x
    x = np.linspace(0.25, 0.70, 250)
    for G in [0.5, 1, 2, 4, 8]:
        y = G / x
        ax.plot(x, y, linestyle="--", linewidth=1, alpha=0.5)
        # label near right side (avoid clutter)
        xr = 0.68
        yr = G / xr
        ax.text(xr, yr, f"G={G}", fontsize=8, alpha=0.8)

    # scatter by device with different markers
    markers = {"RPi4B": "^", "RPi5": "o"}
    for dev, g in df_sub.groupby("device", sort=False):
        ax.scatter(
            g["accept"], g["draft_tps"],
            marker=markers.get(dev, "o"),
            s=50,
            alpha=0.9,
            label=dev,
        )

    # annotate best point per device (max goodput proxy)
    for dev, g in df_sub.groupby("device"):
        best = g.loc[g["goodput_proxy"].idxmax()]
        ax.annotate(
            f"best: {best['model']} {best['quant']}\nG={best['goodput_proxy']:.2f}",
            xy=(best["accept"], best["draft_tps"]),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=9,
            arrowprops=dict(arrowstyle="->", linewidth=1),
        )

    ax.set_title(title)
    ax.set_xlabel("Acceptance rate (accepted/proposed)")
    ax.set_ylabel("Draft decoding speed (tok/s)")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.legend(loc="best", frameon=True)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"{out_prefix}_tradeoff.pdf"))
    fig.savefig(os.path.join(OUT_DIR, f"{out_prefix}_tradeoff.png"))
    plt.close(fig)

def plot_grouped_bar(df_sub: pd.DataFrame, title: str, out_prefix: str):
    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    # pivot to device columns
    pivot = df_sub.pivot_table(
        index="variant", columns="device", values="goodput_proxy", aggfunc="mean"
    ).fillna(0.0)

    # sort by RPi5 (or max across devices)
    if "RPi5" in pivot.columns:
        pivot = pivot.sort_values(by="RPi5", ascending=False)
    else:
        pivot = pivot.sort_values(by=pivot.columns.tolist(), ascending=False)

    x = np.arange(len(pivot.index))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10.5, 4.2), dpi=300)

    devices = list(pivot.columns)
    for i, dev in enumerate(devices):
        ax.bar(x + (i - (len(devices)-1)/2)*width, pivot[dev].values, width, label=dev)

    ax.set_title(title)
    ax.set_ylabel("Verified goodput proxy (tok/s) = draft_tps × acceptance")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=35, ha="right")
    ax.grid(True, axis="y", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.legend(loc="best", frameon=True)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"{out_prefix}_bar.pdf"))
    fig.savefig(os.path.join(OUT_DIR, f"{out_prefix}_bar.png"))
    plt.close(fig)

# -----------------------------
# 4) Generate plots (split by target family)
# -----------------------------
for target, g in df.groupby("target"):
    title = {
        "llama_target_70b": "Tradeoff on RPi4B/5 (target=Meta-Llama-3.1-70B)",
        "qwen_target_32b":  "Tradeoff on RPi4B/5 (target=Qwen3-32B)",
    }.get(target, f"Tradeoff ({target})")
    out = target
    plot_tradeoff(g, title, out)
    plot_grouped_bar(g, title.replace("Tradeoff", "Goodput proxy"), out)

print(f"Done. All outputs saved to {OUT_DIR}/")
