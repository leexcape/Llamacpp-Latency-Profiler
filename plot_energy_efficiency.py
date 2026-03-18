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
parser = argparse.ArgumentParser(description="Energy efficiency analysis")
parser.add_argument("--spec-len", type=int, default=5,
                    help="Fixed speculative length for all configurations (default: 5)")
parser.add_argument("--optimal-K", action="store_true",
                    help="Use optimal K* per configuration instead of fixed spec length")
parser.add_argument("--scatter-width", type=float, default=7.5,
                    help="Scatter figure width in inches (default: 7.5)")
parser.add_argument("--scatter-height", type=float, default=2.8,
                    help="Scatter figure height per subplot in inches (default: 2.8)")
args = parser.parse_args()

# Label strings for titles and annotations
K_TITLE = "$K^*$" if args.optimal_K else f"$K$={args.spec_len}"
K_TAG   = "$K^*$" if args.optimal_K else "$K$"

# -----------------------------
# Figure size configuration (width, height in inches)
# Adjust these to control aspect ratio for paper layout.
# -----------------------------
FIG_DPI          = 300
FIG_SCATTER      = (args.scatter_width, args.scatter_height)
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
# Verification latency (seconds)
#   Goodput(K) = (K·α(K) + 1) / (K/draft_tps + T_VERIFY)
# -----------------------------
T_VERIFY = 0.5

# -----------------------------
# 2) Load acceptance-rate sweep data
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
#    Goodput(K) = (K·α(K) + 1) / (K/draft_tps + T_VERIFY)
#    J/verified_token = power_avg_w / Goodput
# -----------------------------
rows = []
for _, r in raw.iterrows():
    mapping = GGUF_MAP.get(r["model_name"])
    if mapping is None:
        continue
    target, hf_model, quant = mapping

    draft_info = HF_TO_DRAFT.get(hf_model)
    if draft_info is None:
        continue
    model_short, flavor = draft_info
    hf_key = DRAFT_ACC_MAP.get((target, model_short, flavor))
    if hf_key is None:
        continue
    alpha_curve = ACC_CURVES.get(hf_key)
    if alpha_curve is None:
        continue

    device = DEVICE_MAP.get(r["device"])
    if device is None:
        continue

    tok_s = float(r["tok_s_avg"])
    power_w = float(r["power_avg_w"])

    if args.optimal_K:
        opt_K, goodput, alpha = _find_optimal_K(tok_s, alpha_curve, T_VERIFY)
    else:
        opt_K = args.spec_len
        alpha = alpha_curve.get(opt_K)
        if alpha is None:
            continue
        goodput = _compute_goodput(opt_K, alpha, tok_s, T_VERIFY)

    # Only count energy during local drafting (K/draft_tps);
    # T_verify happens on the remote server, not the local device.
    j_per_verified = power_w * (opt_K / tok_s) / (opt_K * alpha + 1.0)

    rows.append({
        "target":              target,
        "device":              device,
        "hf_model":            hf_model,
        "quant":               quant,
        "tok_s":               tok_s,
        "power_w":             power_w,
        "accept":              alpha,
        "optimal_K":           int(opt_K),
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
        "font.size":          15,
        "axes.labelsize":     15,
        "axes.titlesize":     18,
        "axes.titleweight":   "bold",
        "legend.fontsize":    10,
        "xtick.labelsize":    15,
        "ytick.labelsize":    15,
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
# 6) Combined grouped bar chart: J / verified token (two rows)
# -----------------------------
TARGET_ORDER = ["llama_target_70b", "qwen_target_32b"]


def _draw_energy_bar(ax, g, target, show_legend=True):
    """Draw an energy-efficiency grouped bar chart on the given axes."""
    display_name = TARGET_DISPLAY[target]

    g = g.copy()
    g["variant_display"] = g.apply(
        lambda r: f"{MODEL_DISPLAY.get(r['hf_model'], r['hf_model'])}\n({r['quant']})",
        axis=1,
    )

    pivot = g.pivot_table(
        index="variant_display", columns="device",
        values="j_per_verified_tok", aggfunc="mean",
    ).fillna(0.0)

    ordered_cols = [d for d in DEVICE_ORDER if d in pivot.columns]
    pivot = pivot[ordered_cols]

    sort_col = "Jetson AGX Orin" if "Jetson AGX Orin" in pivot.columns else ordered_cols[0]
    pivot = pivot.sort_values(by=sort_col, ascending=True)

    n_variants = len(pivot.index)
    n_devices = len(ordered_cols)
    width = 0.75 / n_devices
    x = np.arange(n_variants)

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

    ax.set_ylabel("J / Verified Token")
    ax.set_title(
        f"Energy Efficiency Comparison  (Target: {display_name}, {K_TITLE}", pad=10
    )
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=38, ha="right")

    ax.grid(True, axis="y", linestyle="-", linewidth=0.4, alpha=0.35, zorder=0)
    ax.set_axisbelow(True)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    if show_legend:
        ax.legend(loc="upper left", frameon=True, fancybox=False,
                  edgecolor="#cccccc", framealpha=0.95, ncol=n_devices)


_paper_rc()
fig, axes = plt.subplots(
    2, 1, figsize=(FIG_BAR[0], FIG_BAR[1] * 2),
    dpi=FIG_DPI, constrained_layout=True,
)

for row, target in enumerate(TARGET_ORDER):
    g = df[df["target"] == target]
    _draw_energy_bar(axes[row], g, target, show_legend=(row == 0))

fig.savefig(os.path.join(OUT_DIR, "energy_bar.pdf"))
fig.savefig(os.path.join(OUT_DIR, "energy_bar.png"))
plt.close(fig)


# -----------------------------
# 7) Combined scatter: Goodput vs J/verified_token (two rows)
#    Iso-power curves: power = J/tok × goodput ⇒ J/tok = P / goodput
# -----------------------------
def _draw_energy_scatter(ax, g, target, show_legend=True):
    """Draw a speed-energy scatter plot on the given axes."""
    display_name = TARGET_DISPLAY[target]

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

    # Annotate only the best (lowest J/tok) point per device
    for idx in best_indices:
        r = g.loc[idx]
        dev = r["device"]
        model_label = MODEL_DISPLAY.get(r["hf_model"], r["hf_model"])
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

    ax.set_xlabel("Verified Goodput  (tok/s)  =  $(K\\alpha(K)+1)\\,/\\,(K/\\mathrm{tps}+T_{\\mathrm{verify}})$")
    ax.set_ylabel("J / Verified Token")
    ax.set_title(
        f"Speed–Energy Tradeoff  (Target: {display_name}, {K_TITLE}", pad=10
    )

    # Light grid
    ax.grid(True, which="major", linestyle="-", linewidth=0.4, alpha=0.35)

    # Tidy spines
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    if show_legend:
        ax.legend(loc="upper center", frameon=True, fancybox=False,
                  edgecolor="#cccccc", framealpha=0.95)


_paper_rc()
fig, axes = plt.subplots(
    2, 1, figsize=(FIG_SCATTER[0], FIG_SCATTER[1] * 2),
    dpi=FIG_DPI, constrained_layout=True,
)

for row, target in enumerate(TARGET_ORDER):
    g = df[df["target"] == target]
    _draw_energy_scatter(axes[row], g, target, show_legend=(row == 0))

fig.savefig(os.path.join(OUT_DIR, "energy_scatter.pdf"))
fig.savefig(os.path.join(OUT_DIR, "energy_scatter.png"))
plt.close(fig)

print(f"Done. All outputs saved to {OUT_DIR}/")
