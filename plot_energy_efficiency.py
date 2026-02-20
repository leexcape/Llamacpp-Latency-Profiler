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
OUT_DIR = os.path.join("results", "energy_efficiency_analysis", TIMESTAMP)
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------
# 1) Load raw profiling CSVs
# -----------------------------
CSV_FILES = [
    "results/pi5_llamacpp_profile_19models.csv",
    "results/jetson_agx_orin_llamacpp_profile_19models.csv",
]
raw = pd.concat([pd.read_csv(f) for f in CSV_FILES], ignore_index=True)

# -----------------------------
# 2) Acceptance rates (target, hf_model) → α
# -----------------------------
ACC = {
    # Target: Meta-Llama-3.1-70B
    ("llama_target_70b", "llama-3.2-1b"):          0.5038,
    ("llama_target_70b", "llama-3.2-1b-instruct"):  0.5102,
    ("llama_target_70b", "llama-3.2-3b"):           0.5691,
    ("llama_target_70b", "llama-3.1-8b"):           0.6488,

    # Target: Qwen3-32B
    ("qwen_target_32b", "qwen3-0.6b"): 0.3826,
    ("qwen_target_32b", "qwen3-1.7b"): 0.4494,
    ("qwen_target_32b", "qwen3-4b"):   0.4705,
    ("qwen_target_32b", "qwen3-8b"):   0.5357,
}

# -----------------------------
# 3) Map GGUF filename → (target, hf_model, quant)
#    Only draft models with known acceptance rates are included.
# -----------------------------
GGUF_MAP = {
    # Llama family (target = Llama-3.1-70B)
    "llama-3.2-1b-q4_k_m.gguf":                  ("llama_target_70b", "llama-3.2-1b",          "Q4_K_M"),
    "llama-3.2-1b-instruct-q4_k_m.gguf":         ("llama_target_70b", "llama-3.2-1b-instruct", "Q4_K_M"),
    "llama-3.2-3b-instruct-q4_k_m.gguf":         ("llama_target_70b", "llama-3.2-3b",          "Q4_K_M"),
    "Llama-3.2-3B-Instruct-Q8_0.gguf":           ("llama_target_70b", "llama-3.2-3b",          "Q8_0"),
    "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf":    ("llama_target_70b", "llama-3.1-8b",          "Q4_K_M"),

    # Qwen3 family (target = Qwen3-32B)
    "Qwen3-0.6B-Q4_K_M.gguf":                    ("qwen_target_32b",  "qwen3-0.6b",            "Q4_K_M"),
    "Qwen3-0.6B-Q8_0.gguf":                      ("qwen_target_32b",  "qwen3-0.6b",            "Q8_0"),
    "Qwen3-1.7B-Q4_K_M.gguf":                    ("qwen_target_32b",  "qwen3-1.7b",            "Q4_K_M"),
    "Qwen3-1.7B-Q8_0.gguf":                      ("qwen_target_32b",  "qwen3-1.7b",            "Q8_0"),
    "Qwen3-4B-Q4_K_M.gguf":                      ("qwen_target_32b",  "qwen3-4b",              "Q4_K_M"),
    "Qwen3-4B-Q8_0.gguf":                        ("qwen_target_32b",  "qwen3-4b",              "Q8_0"),
    "Qwen3-8B-Q4_K_M.gguf":                      ("qwen_target_32b",  "qwen3-8b",              "Q4_K_M"),
    "Qwen3-8B-Q6_K.gguf":                        ("qwen_target_32b",  "qwen3-8b",              "Q6_K"),
}

DEVICE_MAP = {
    "raspberry_pi_5":  "RPi 5",
    "jetson_agx_orin": "Jetson AGX Orin",
}

# -----------------------------
# 4) Build analysis dataframe
#
#    J/verified_token = power_avg_w / (tok_s_avg × α)
# -----------------------------
rows = []
for _, r in raw.iterrows():
    mapping = GGUF_MAP.get(r["model_name"])
    if mapping is None:
        continue
    target, hf_model, quant = mapping
    acc = ACC.get((target, hf_model))
    if acc is None:
        continue

    device = DEVICE_MAP.get(r["device"])
    if device is None:
        continue

    tok_s = float(r["tok_s_avg"])
    power_w = float(r["power_avg_w"])
    goodput = tok_s * acc
    j_per_verified = power_w / goodput

    rows.append({
        "target":              target,
        "device":              device,
        "hf_model":            hf_model,
        "quant":               quant,
        "tok_s":               tok_s,
        "power_w":             power_w,
        "accept":              acc,
        "goodput":             goodput,
        "j_per_verified_tok":  j_per_verified,
    })

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT_DIR, "energy_efficiency_results.csv"), index=False)

# -----------------------------
# 5) Style (aligned with plot_goodput.py)
# -----------------------------
DEVICE_ORDER = ["RPi 5", "Jetson AGX Orin"]

DEVICE_COLORS = {
    "RPi 5":           "#D55E00",   # vermillion
    "Jetson AGX Orin": "#009E73",   # green
}
DEVICE_MARKERS = {
    "RPi 5":           "o",
    "Jetson AGX Orin": "s",
}
DEVICE_HATCHES = {
    "RPi 5":           "\\\\",
    "Jetson AGX Orin": "xx",
}

MODEL_DISPLAY = {
    "llama-3.2-1b":          "Llama-3.2-1B",
    "llama-3.2-1b-instruct": "Llama-3.2-1B-Inst.",
    "llama-3.2-3b":          "Llama-3.2-3B",
    "llama-3.1-8b":          "Llama-3.1-8B",
    "qwen3-0.6b":            "Qwen3-0.6B",
    "qwen3-1.7b":            "Qwen3-1.7B",
    "qwen3-4b":              "Qwen3-4B",
    "qwen3-8b":              "Qwen3-8B",
}

TARGET_DISPLAY = {
    "llama_target_70b": "Llama-3.1-70B",
    "qwen_target_32b":  "Qwen3-32B",
}


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
# 6) Per-target grouped bar chart: J / verified token
# -----------------------------
for target, g in df.groupby("target"):
    _paper_rc()
    display_name = TARGET_DISPLAY[target]

    # Build variant display label
    g = g.copy()
    g["variant_display"] = g.apply(
        lambda r: f"{MODEL_DISPLAY.get(r['hf_model'], r['hf_model'])}\n({r['quant']})",
        axis=1,
    )

    pivot = g.pivot_table(
        index="variant_display", columns="device",
        values="j_per_verified_tok", aggfunc="mean",
    ).fillna(0.0)

    # Reorder columns to match DEVICE_ORDER
    ordered_cols = [d for d in DEVICE_ORDER if d in pivot.columns]
    pivot = pivot[ordered_cols]

    # Sort by Jetson value ascending (most efficient first)
    sort_col = "Jetson AGX Orin" if "Jetson AGX Orin" in pivot.columns else ordered_cols[0]
    pivot = pivot.sort_values(by=sort_col, ascending=True)

    n_variants = len(pivot.index)
    n_devices = len(ordered_cols)
    width = 0.75 / n_devices
    x = np.arange(n_variants)

    fig, ax = plt.subplots(figsize=FIG_BAR, dpi=FIG_DPI)

    for i, dev in enumerate(ordered_cols):
        offset = (i - (n_devices - 1) / 2) * width
        ax.bar(
            x + offset, pivot[dev].values, width,
            label=dev,
            color=DEVICE_COLORS[dev],
            edgecolor="white", linewidth=0.6,
            hatch=DEVICE_HATCHES[dev],
            alpha=0.88, zorder=3,
        )

    ax.set_ylabel("Energy / Verified Token  (J/tok)")
    ax.set_title(
        f"Energy Efficiency Comparison  (Target: {display_name})", pad=10
    )
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=38, ha="right")

    # Light horizontal grid behind bars
    ax.grid(True, axis="y", linestyle="-", linewidth=0.4, alpha=0.35, zorder=0)
    ax.set_axisbelow(True)

    # Tidy spines
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax.legend(loc="upper left", frameon=True, fancybox=False,
              edgecolor="#cccccc", framealpha=0.95, ncol=n_devices)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"{target}_energy_bar.pdf"))
    fig.savefig(os.path.join(OUT_DIR, f"{target}_energy_bar.png"))
    plt.close(fig)


# -----------------------------
# 7) Per-target scatter: Goodput vs J/verified_token
#    Iso-power curves: power = J/tok × goodput ⇒ J/tok = P / goodput
# -----------------------------
for target, g in df.groupby("target"):
    _paper_rc()
    display_name = TARGET_DISPLAY[target]

    fig, ax = plt.subplots(figsize=FIG_SCATTER, dpi=FIG_DPI)

    # Cap y-axis to the data range with padding
    y_upper = g["j_per_verified_tok"].max() * 1.25
    ax.set_ylim(0, y_upper)

    # Iso-power contour curves (clipped to visible region)
    x_range = np.linspace(
        g["goodput"].min() * 0.5, g["goodput"].max() * 1.5, 300
    )
    for P in [5, 10, 20, 40, 60]:
        y = P / x_range
        y_clipped = np.where(y <= y_upper, y, np.nan)
        ax.plot(x_range, y_clipped, linestyle="--", linewidth=0.7,
                color="#888888", alpha=0.40)
        # Label near the right edge
        xr = x_range[-1] * 0.95
        yr = P / xr
        if yr < y_upper and yr > 0:
            ax.text(xr, yr, f"{P} W", fontsize=8, color="#666666",
                    va="bottom", ha="right")

    # Scatter by device
    for dev in DEVICE_ORDER:
        gg = g[g["device"] == dev]
        if gg.empty:
            continue
        ax.scatter(
            gg["goodput"], gg["j_per_verified_tok"],
            marker=DEVICE_MARKERS[dev],
            color=DEVICE_COLORS[dev],
            edgecolors="white", linewidths=0.5,
            s=60, alpha=0.92, zorder=5,
            label=dev,
        )

    # Find frontier (best) point per device for distinct styling
    best_indices = set()
    for dev in DEVICE_ORDER:
        gg = g[g["device"] == dev]
        if gg.empty:
            continue
        best_indices.add(gg["j_per_verified_tok"].idxmin())

    # Annotate ALL points; frontier points get bold + arrow, others get muted text
    for idx, r in g.iterrows():
        dev = r["device"]
        model_label = MODEL_DISPLAY.get(r["hf_model"], r["hf_model"])
        is_frontier = idx in best_indices

        if is_frontier:
            label = (f"{model_label}\n({r['quant']})  "
                     f"{r['j_per_verified_tok']:.2f} J/tok")
            ax.annotate(
                label,
                xy=(r["goodput"], r["j_per_verified_tok"]),
                xytext=(14, -16), textcoords="offset points",
                fontsize=9, fontweight="bold",
                color=DEVICE_COLORS[dev],
                arrowprops=dict(arrowstyle="-|>", color=DEVICE_COLORS[dev],
                                lw=1.0, shrinkA=0, shrinkB=3),
            )
        else:
            label = f"{model_label} ({r['quant']})"
            ax.annotate(
                label,
                xy=(r["goodput"], r["j_per_verified_tok"]),
                xytext=(8, 6), textcoords="offset points",
                fontsize=7, alpha=0.6,
                color=DEVICE_COLORS[dev],
            )

    ax.set_xlabel("Verified Goodput  (tok/s)  =  draft_tps $\\times$ $\\alpha$")
    ax.set_ylabel("Energy / Verified Token  (J/tok)")
    ax.set_title(
        f"Speed–Energy Tradeoff  (Target: {display_name})", pad=10
    )

    # Light grid
    ax.grid(True, which="major", linestyle="-", linewidth=0.4, alpha=0.35)

    # Tidy spines
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax.legend(loc="upper right", frameon=True, fancybox=False,
              edgecolor="#cccccc", framealpha=0.95)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"{target}_energy_scatter.pdf"))
    fig.savefig(os.path.join(OUT_DIR, f"{target}_energy_scatter.png"))
    plt.close(fig)

print(f"Done. All outputs saved to {OUT_DIR}/")
