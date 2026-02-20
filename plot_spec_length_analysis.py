#!/usr/bin/env python3
"""
Speculative-length analysis: acceptance rate, tokens per round,
and goodput trade-off as a function of speculative length K.

Goodput model
=============
Each speculative-decoding round:
  1. Draft K tokens on the edge device   → time = K / draft_tps
  2. Verify all K tokens in one pass      → time = T_verify
  3. Accept K × α(K) draft tokens + 1 bonus token from the verifier

  Goodput(K) = (K × α(K) + 1) / (K / draft_tps + T_verify)

The optimal K* balances:
  • Amortising the fixed verification cost  (favours large K)
  • Minimising wasted draft computation     (favours small K)

The trade-off is governed by R = draft_tps × T_verify (tokens the
drafter could produce during one verification call).
"""

import os
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D

# ─────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────
FIG_DPI = 300
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = os.path.join("results", "spec_length_analysis", TIMESTAMP)
os.makedirs(OUT_DIR, exist_ok=True)

T_VERIFY_DEFAULT = 0.2          # seconds, used in goodput-vs-K plots
T_VERIFY_SWEEP = np.arange(0.05, 2.05, 0.05)  # for optimal-K analysis

# ─────────────────────────────────────────────────────────
# Load acceptance-rate sweep data
# ─────────────────────────────────────────────────────────
acc_qwen = pd.read_csv("results/profile_results_2026-02-20_11-42-25.csv")
acc_llama = pd.read_csv("results/profile_results_2026-02-20_13-30-21.csv")
acc_df = pd.concat([acc_qwen, acc_llama], ignore_index=True)

# ─────────────────────────────────────────────────────────
# Name mappings
# ─────────────────────────────────────────────────────────
DRAFT_HF_TO_DISPLAY = {
    "Qwen/Qwen3-0.6B":                          "Qwen3-0.6B",
    "Qwen/Qwen3-1.7B":                          "Qwen3-1.7B",
    "Qwen/Qwen3-4B":                            "Qwen3-4B",
    "Qwen/Qwen3-8B":                            "Qwen3-8B",
    "meta-llama/Llama-3.2-1B":                   "Llama-3.2-1B",
    "meta-llama/Llama-3.2-1B-Instruct":          "Llama-3.2-1B-Inst.",
    "meta-llama/Llama-3.2-3B-Instruct":          "Llama-3.2-3B-Inst.",
    "meta-llama/Meta-Llama-3.1-8B-Instruct":     "Llama-3.1-8B-Inst.",
}
TARGET_HF_TO_KEY = {
    "Qwen/Qwen3-32B":                           "qwen_target_32b",
    "meta-llama/Meta-Llama-3.1-70B-Instruct":   "llama_target_70b",
}
TARGET_DISPLAY = {
    "llama_target_70b": "Llama-3.1-70B",
    "qwen_target_32b":  "Qwen3-32B",
}

acc_df["draft_display"]  = acc_df["draft_model"].map(DRAFT_HF_TO_DISPLAY)
acc_df["target_key"]     = acc_df["target_model"].map(TARGET_HF_TO_KEY)

# Derived columns
acc_df["tokens_per_round"] = (
    acc_df["spec_len"] * acc_df["mean_acceptance_rate"] + 1
)
acc_df["wasted_per_round"] = (
    acc_df["spec_len"] * (1 - acc_df["mean_acceptance_rate"])
)

# ─────────────────────────────────────────────────────────
# Draft-model speeds (Q4_K_M, default quantisation)
# ─────────────────────────────────────────────────────────
DRAFT_TPS = {
    # --- Llama family (target = Llama-3.1-70B) ---
    ("Llama-3.2-1B",       "RPi 4B"): 4.18,
    ("Llama-3.2-1B",       "RPi 5"):  12.86,
    ("Llama-3.2-1B",       "Jetson AGX Orin"): 92.94,
    ("Llama-3.2-1B-Inst.", "RPi 4B"): 4.14,
    ("Llama-3.2-1B-Inst.", "RPi 5"):  14.47,
    ("Llama-3.2-1B-Inst.", "Jetson AGX Orin"): 93.14,
    ("Llama-3.2-3B-Inst.", "RPi 4B"): 1.68,
    ("Llama-3.2-3B-Inst.", "RPi 5"):  4.68,
    ("Llama-3.2-3B-Inst.", "Jetson AGX Orin"): 42.52,
    ("Llama-3.1-8B-Inst.", "RPi 4B"): 0.72,
    ("Llama-3.1-8B-Inst.", "RPi 5"):  1.77,
    ("Llama-3.1-8B-Inst.", "Jetson AGX Orin"): 25.07,
    # --- Qwen family (target = Qwen3-32B) ---
    ("Qwen3-0.6B", "RPi 4B"): 7.84,
    ("Qwen3-0.6B", "RPi 5"):  18.18,
    ("Qwen3-0.6B", "Jetson AGX Orin"): 98.63,
    ("Qwen3-1.7B", "RPi 4B"): 3.02,
    ("Qwen3-1.7B", "RPi 5"):  7.15,
    ("Qwen3-1.7B", "Jetson AGX Orin"): 65.31,
    ("Qwen3-4B",   "RPi 4B"): 1.34,
    ("Qwen3-4B",   "RPi 5"):  3.11,
    ("Qwen3-4B",   "Jetson AGX Orin"): 33.70,
    ("Qwen3-8B",   "RPi 4B"): 0.72,
    ("Qwen3-8B",   "RPi 5"):  1.78,
    ("Qwen3-8B",   "Jetson AGX Orin"): 24.03,
}

# ─────────────────────────────────────────────────────────
# Visual constants
# ─────────────────────────────────────────────────────────
TARGET_ORDER = [
    ("llama_target_70b", "Llama-3.1-70B"),
    ("qwen_target_32b",  "Qwen3-32B"),
]
DRAFT_ORDER = {
    "llama_target_70b": [
        "Llama-3.2-1B", "Llama-3.2-1B-Inst.",
        "Llama-3.2-3B-Inst.", "Llama-3.1-8B-Inst.",
    ],
    "qwen_target_32b": [
        "Qwen3-0.6B", "Qwen3-1.7B", "Qwen3-4B", "Qwen3-8B",
    ],
}
DRAFT_PALETTES = {
    "llama_target_70b": {
        "Llama-3.2-1B":       "#a6cee3",
        "Llama-3.2-1B-Inst.": "#6baed6",
        "Llama-3.2-3B-Inst.": "#1f78b4",
        "Llama-3.1-8B-Inst.": "#08306b",
    },
    "qwen_target_32b": {
        "Qwen3-0.6B": "#b2df8a",
        "Qwen3-1.7B": "#33a02c",
        "Qwen3-4B":   "#006d2c",
        "Qwen3-8B":   "#00441b",
    },
}
DRAFT_MARKERS = ["o", "s", "^", "D"]

DEVICE_ORDER  = ["RPi 4B", "RPi 5", "Jetson AGX Orin"]
DEVICE_COLORS = {
    "RPi 4B": "#0072B2", "RPi 5": "#D55E00", "Jetson AGX Orin": "#009E73",
}
DEVICE_LSTYLES = {"RPi 4B": ":", "RPi 5": "--", "Jetson AGX Orin": "-"}


def _paper_rc():
    mpl.rcParams.update({
        "font.family":        "serif",
        "font.serif":         ["Times New Roman", "DejaVu Serif", "serif"],
        "mathtext.fontset":   "stix",
        "font.size":          10,
        "axes.labelsize":     12,
        "axes.titlesize":     13,
        "axes.titleweight":   "bold",
        "legend.fontsize":    9,
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


def _tidy(ax):
    ax.grid(True, linestyle="-", linewidth=0.4, alpha=0.35)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


# ─────────────────────────────────────────────────────────
# Goodput helpers
# ─────────────────────────────────────────────────────────
def compute_goodput(K, alpha_K, draft_tps, T_verify):
    """Accepted tok/s = (K·α + 1) / (K/draft_tps + T_verify)."""
    return (K * alpha_K + 1.0) / (K / draft_tps + T_verify)


def _get_alpha_series(target_key, draft_display):
    """Return (spec_lens, alphas) arrays sorted by K."""
    sub = acc_df[
        (acc_df["target_key"] == target_key)
        & (acc_df["draft_display"] == draft_display)
    ].sort_values("spec_len")
    return sub["spec_len"].values, sub["mean_acceptance_rate"].values


# ═════════════════════════════════════════════════════════
# Figure 1 — Acceptance-rate decay
# ═════════════════════════════════════════════════════════
def plot_acceptance_rate():
    _paper_rc()
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.5), dpi=FIG_DPI)

    for col, (tk, td) in enumerate(TARGET_ORDER):
        ax = axes[col]
        palette = DRAFT_PALETTES[tk]
        sub = acc_df[acc_df["target_key"] == tk]

        for i, draft in enumerate(DRAFT_ORDER[tk]):
            g = sub[sub["draft_display"] == draft].sort_values("spec_len")
            ax.plot(
                g["spec_len"], g["mean_acceptance_rate"],
                marker=DRAFT_MARKERS[i], color=palette[draft],
                linewidth=1.5, markersize=5, label=draft, zorder=5,
            )
            ax.fill_between(
                g["spec_len"],
                g["mean_acceptance_rate"] - g["std_acceptance_rate"],
                g["mean_acceptance_rate"] + g["std_acceptance_rate"],
                color=palette[draft], alpha=0.15,
            )

        ax.set_xlabel("Speculative Length  $K$")
        ax.set_ylabel("Acceptance Rate  $\\alpha(K)$")
        ax.set_title(f"Target: {td}", pad=8)
        ax.set_xticks(range(2, 11))
        ax.set_ylim(0, 1.0)
        ax.legend(
            loc="upper right", frameon=True, fancybox=False,
            edgecolor="#cccccc", framealpha=0.95,
        )
        _tidy(ax)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "acceptance_rate_vs_speclen.pdf"))
    fig.savefig(os.path.join(OUT_DIR, "acceptance_rate_vs_speclen.png"))
    plt.close(fig)


# ═════════════════════════════════════════════════════════
# Figure 2 — Accepted tokens per round (sub-linear growth)
# ═════════════════════════════════════════════════════════
def plot_tokens_per_round():
    _paper_rc()
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.5), dpi=FIG_DPI)

    for col, (tk, td) in enumerate(TARGET_ORDER):
        ax = axes[col]
        palette = DRAFT_PALETTES[tk]
        sub = acc_df[acc_df["target_key"] == tk]

        K_range = np.arange(2, 11)
        # Ideal reference (α = 1)
        ax.plot(
            K_range, K_range + 1, linestyle="--", color="#888888",
            linewidth=1.0, alpha=0.6, label="Ideal ($\\alpha$=1)", zorder=2,
        )
        # Shade the gap for the smallest draft model (worst case)
        worst_draft = DRAFT_ORDER[tk][0]
        gw = sub[sub["draft_display"] == worst_draft].sort_values("spec_len")
        ax.fill_between(
            gw["spec_len"],
            gw["spec_len"] * gw["mean_acceptance_rate"].values + 1,
            gw["spec_len"].values + 1,
            color="#e0e0e0", alpha=0.5, label="Wasted (smallest draft)",
            zorder=1,
        )

        for i, draft in enumerate(DRAFT_ORDER[tk]):
            g = sub[sub["draft_display"] == draft].sort_values("spec_len")
            tokens = g["spec_len"] * g["mean_acceptance_rate"] + 1
            ax.plot(
                g["spec_len"], tokens,
                marker=DRAFT_MARKERS[i], color=palette[draft],
                linewidth=1.5, markersize=5, label=draft, zorder=5,
            )

        ax.set_xlabel("Speculative Length  $K$")
        ax.set_ylabel("Accepted Tokens / Round")
        ax.set_title(f"Target: {td}", pad=8)
        ax.set_xticks(range(2, 11))
        ax.legend(
            loc="upper left", frameon=True, fancybox=False,
            edgecolor="#cccccc", framealpha=0.95, fontsize=8,
        )
        _tidy(ax)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "tokens_per_round_vs_speclen.pdf"))
    fig.savefig(os.path.join(OUT_DIR, "tokens_per_round_vs_speclen.png"))
    plt.close(fig)


# ═════════════════════════════════════════════════════════
# Figure 3 — Goodput vs K  (target × device grid)
# ═════════════════════════════════════════════════════════
def plot_goodput_vs_speclen(T_verify=T_VERIFY_DEFAULT):
    _paper_rc()
    fig, axes = plt.subplots(
        2, 3, figsize=(14.0, 7.0), dpi=FIG_DPI,
        sharex=True,
    )

    for row, (tk, td) in enumerate(TARGET_ORDER):
        palette = DRAFT_PALETTES[tk]
        sub = acc_df[acc_df["target_key"] == tk]

        for col, device in enumerate(DEVICE_ORDER):
            ax = axes[row][col]

            for i, draft in enumerate(DRAFT_ORDER[tk]):
                g = sub[sub["draft_display"] == draft].sort_values("spec_len")
                tps = DRAFT_TPS.get((draft, device))
                if tps is None:
                    continue

                gp = [
                    compute_goodput(K, a, tps, T_verify)
                    for K, a in zip(g["spec_len"], g["mean_acceptance_rate"])
                ]

                ax.plot(
                    g["spec_len"], gp,
                    marker=DRAFT_MARKERS[i], color=palette[draft],
                    linewidth=1.5, markersize=5, label=draft, zorder=5,
                )

                # Mark optimal K with a star
                best_idx = int(np.argmax(gp))
                ax.plot(
                    g["spec_len"].iloc[best_idx], gp[best_idx],
                    marker="*", markersize=14, color=palette[draft],
                    markeredgecolor="black", markeredgewidth=0.6, zorder=10,
                )

            ax.set_xticks(range(2, 11))
            if row == 1:
                ax.set_xlabel("Speculative Length  $K$")
            if col == 0:
                ax.set_ylabel("Goodput  (tok/s)")
            ax.set_title(f"{td} — {device}", pad=6, fontsize=11)
            _tidy(ax)

            if row == 0 and col == 2:
                ax.legend(
                    loc="best", frameon=True, fancybox=False,
                    edgecolor="#cccccc", framealpha=0.95, fontsize=8,
                )
            if row == 1 and col == 2:
                ax.legend(
                    loc="best", frameon=True, fancybox=False,
                    edgecolor="#cccccc", framealpha=0.95, fontsize=8,
                )

    fig.suptitle(
        f"Goodput vs Speculative Length  "
        f"($T_{{\\mathrm{{verify}}}}$ = {T_verify}s, "
        f"$\\bigstar$ = optimal $K$)",
        fontsize=14, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    tag = f"Tv{T_verify:.1f}s".replace(".", "p")
    fig.savefig(os.path.join(OUT_DIR, f"goodput_vs_speclen_{tag}.pdf"))
    fig.savefig(os.path.join(OUT_DIR, f"goodput_vs_speclen_{tag}.png"))
    plt.close(fig)


# ═════════════════════════════════════════════════════════
# Figure 4 — Optimal K* vs T_verify  (target × device grid)
# ═════════════════════════════════════════════════════════
def plot_optimal_K():
    _paper_rc()
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 7.0), dpi=FIG_DPI, sharex=True)

    for row, (tk, td) in enumerate(TARGET_ORDER):
        palette = DRAFT_PALETTES[tk]

        for col, device in enumerate(DEVICE_ORDER):
            ax = axes[row][col]

            for i, draft in enumerate(DRAFT_ORDER[tk]):
                tps = DRAFT_TPS.get((draft, device))
                if tps is None:
                    continue
                Ks, alphas = _get_alpha_series(tk, draft)

                opt_Ks = []
                for tv in T_VERIFY_SWEEP:
                    gps = [compute_goodput(K, a, tps, tv) for K, a in zip(Ks, alphas)]
                    opt_Ks.append(Ks[int(np.argmax(gps))])

                ax.plot(
                    T_VERIFY_SWEEP, opt_Ks,
                    marker=DRAFT_MARKERS[i], color=palette[draft],
                    linewidth=1.3, markersize=3, label=draft, zorder=5,
                )

            ax.set_yticks(range(2, 11))
            ax.set_ylim(1.5, 10.5)
            if row == 1:
                ax.set_xlabel("Verification Latency  $T_{\\mathrm{verify}}$  (s)")
            if col == 0:
                ax.set_ylabel("Optimal $K^*$")
            ax.set_title(f"{td} — {device}", pad=6, fontsize=11)
            _tidy(ax)

            if row == 0 and col == 0:
                ax.legend(
                    loc="lower right", frameon=True, fancybox=False,
                    edgecolor="#cccccc", framealpha=0.95, fontsize=8,
                )
            if row == 1 and col == 0:
                ax.legend(
                    loc="lower right", frameon=True, fancybox=False,
                    edgecolor="#cccccc", framealpha=0.95, fontsize=8,
                )

    fig.suptitle(
        "Optimal Speculative Length $K^*$ vs Verification Latency",
        fontsize=14, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "optimal_speclen_vs_Tverify.pdf"))
    fig.savefig(os.path.join(OUT_DIR, "optimal_speclen_vs_Tverify.png"))
    plt.close(fig)


# ═════════════════════════════════════════════════════════
# Export summary table
# ═════════════════════════════════════════════════════════
def export_summary():
    """CSV with optimal K* for representative T_verify values."""
    tv_samples = [0.1, 0.2, 0.5, 1.0, 2.0]
    rows = []
    for tk, td in TARGET_ORDER:
        for draft in DRAFT_ORDER[tk]:
            Ks, alphas = _get_alpha_series(tk, draft)
            for device in DEVICE_ORDER:
                tps = DRAFT_TPS.get((draft, device))
                if tps is None:
                    continue
                for tv in tv_samples:
                    gps = [compute_goodput(K, a, tps, tv)
                           for K, a in zip(Ks, alphas)]
                    best_idx = int(np.argmax(gps))
                    rows.append({
                        "target": td,
                        "draft_model": draft,
                        "device": device,
                        "draft_tps": tps,
                        "T_verify_s": tv,
                        "R": tps * tv,
                        "optimal_K": Ks[best_idx],
                        "goodput_at_optimal_K": gps[best_idx],
                    })
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(OUT_DIR, "optimal_speclen_summary.csv"), index=False)


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    plot_acceptance_rate()
    print("  [1/5] acceptance_rate_vs_speclen")

    plot_tokens_per_round()
    print("  [2/5] tokens_per_round_vs_speclen")

    plot_goodput_vs_speclen(T_verify=T_VERIFY_DEFAULT)
    print(f"  [3/5] goodput_vs_speclen (T_verify={T_VERIFY_DEFAULT}s)")

    plot_optimal_K()
    print("  [4/5] optimal_speclen_vs_Tverify")

    export_summary()
    print("  [5/5] optimal_speclen_summary.csv")

    print(f"\nDone. All outputs saved to {OUT_DIR}/")
