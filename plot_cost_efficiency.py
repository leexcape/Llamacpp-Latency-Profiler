import os
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Output directory with timestamp
# -----------------------------
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = os.path.join("results", "cost_efficiency_analysis", TIMESTAMP)
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------
# 1) Inputs (from your screenshots)
# -----------------------------
ACC = {
    ("llama_target_70b", "llama-3.2-1b"): 0.5038,
    ("llama_target_70b", "llama-3.2-3b"): 0.5691,
    ("llama_target_70b", "llama-3.1-8b"): 0.6488,

    ("qwen_target_32b", "qwen3-0.6b"): 0.3826,
    ("qwen_target_32b", "qwen3-1.7b"): 0.4494,
    ("qwen_target_32b", "qwen3-4b"): 0.4705,
    ("qwen_target_32b", "qwen3-8b"): 0.5357,
}

SPEED_RPI4 = {
    ("llama_target_70b", "llama-3.2-1b", "q4_k_m", "instruct"): 4.14,
    ("llama_target_70b", "llama-3.2-1b", "q4_k_m", "base"): 4.18,
    ("llama_target_70b", "llama-3.2-3b", "q4_k_m", "instruct"): 1.68,
    ("llama_target_70b", "llama-3.2-3b", "q8_0", "instruct"): 1.04,
    ("llama_target_70b", "llama-3.1-8b", "q4_k_m", "instruct"): 0.72,

    ("qwen_target_32b", "qwen3-0.6b", "q4_k_m", "base"): 7.84,
    ("qwen_target_32b", "qwen3-0.6b", "q8_0", "base"): 4.92,
    ("qwen_target_32b", "qwen3-1.7b", "q4_k_m", "base"): 3.02,
    ("qwen_target_32b", "qwen3-1.7b", "q8_0", "base"): 1.89,
    ("qwen_target_32b", "qwen3-4b", "q4_k_m", "base"): 1.34,
    ("qwen_target_32b", "qwen3-4b", "q8_0", "base"): 0.83,
    ("qwen_target_32b", "qwen3-8b", "q4_k_m", "base"): 0.72,
    ("qwen_target_32b", "qwen3-8b", "q6_k", "base"): 0.55,
}

SPEED_RPI5 = {
    ("llama_target_70b", "llama-3.2-1b", "q4_k_m", "instruct"): 14.47,
    ("llama_target_70b", "llama-3.2-1b", "q4_k_m", "base"): 12.86,
    ("llama_target_70b", "llama-3.2-3b", "q4_k_m", "instruct"): 4.68,
    ("llama_target_70b", "llama-3.2-3b", "q8_0", "instruct"): 2.37,
    ("llama_target_70b", "llama-3.1-8b", "q4_k_m", "instruct"): 1.77,

    ("qwen_target_32b", "qwen3-0.6b", "q4_k_m", "base"): 18.18,
    ("qwen_target_32b", "qwen3-0.6b", "q8_0", "base"): 10.38,
    ("qwen_target_32b", "qwen3-1.7b", "q4_k_m", "base"): 7.15,
    ("qwen_target_32b", "qwen3-1.7b", "q8_0", "base"): 4.01,
    ("qwen_target_32b", "qwen3-4b", "q4_k_m", "base"): 3.11,
    ("qwen_target_32b", "qwen3-4b", "q8_0", "base"): 1.93,
    ("qwen_target_32b", "qwen3-8b", "q4_k_m", "base"): 1.78,
    ("qwen_target_32b", "qwen3-8b", "q6_k", "base"): 1.68,
}

# -----------------------------
# 2) Pricing (token-priced verifier)
# -----------------------------
PRICE_OUT_PER_M = {
    "qwen_target_32b": 0.59,   # Groq Qwen3-32B output $/1M
    "llama_target_70b": 0.90,  # Fireworks tier >16B $/1M (proxy for 70B-class)
}

SPEC_LEN = 32

# Optional fixed per-verification-call overhead ($/call).
# Set >0 to make "verification more frequent => more expensive" show up explicitly.
C0_PER_VERIFY_CALL_DOLLAR = 0.0

# -----------------------------
# 3) Build dataframe
# -----------------------------
rows = []

def ingest(device, speed_dict):
    for (target, model, quant, flavor), draft_tps in speed_dict.items():
        acc = ACC.get((target, model))
        if acc is None:
            continue

        p = PRICE_OUT_PER_M[target] / 1_000_000.0  # $/token (output-priced)

        goodput = draft_tps * acc
        verify_rate_calls = draft_tps / SPEC_LEN

        token_cost_per_s = p * draft_tps
        call_overhead_per_s = C0_PER_VERIFY_CALL_DOLLAR * verify_rate_calls
        verify_cost_per_s = token_cost_per_s + call_overhead_per_s

        tokens_per_dollar = goodput / verify_cost_per_s
        dollars_per_1k = 1000.0 / tokens_per_dollar

        rows.append({
            "target": target,
            "device": device,
            "model": model,
            "quant": quant,
            "flavor": flavor,
            "variant": f"{model}-{flavor}-{quant}",
            "draft_tps": float(draft_tps),
            "accept": float(acc),
            "goodput_proxy_tps": float(goodput),
            "verify_rate_calls_per_s": float(verify_rate_calls),
            "verify_cost_$per_s": float(verify_cost_per_s),
            "tokens_per_$": float(tokens_per_dollar),
            "$_per_1k_accepted_tokens": float(dollars_per_1k),
        })

ingest("RPi4B", SPEED_RPI4)
ingest("RPi5", SPEED_RPI5)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT_DIR, "cost_aware_results.csv"), index=False)

# -----------------------------
# 4) Plot style
# -----------------------------
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

titles = {
    "qwen_target_32b": "Cost-aware tradeoff (target=Qwen3-32B; token-priced verifier)",
    "llama_target_70b": "Cost-aware tradeoff (target=Llama-70B-class; token-priced verifier)",
}
markers = {"RPi4B": "^", "RPi5": "o"}

# -----------------------------
# 5) Plot A: Pareto scatter (goodput vs $/1k accepted tokens)
# -----------------------------
for target, g in df.groupby("target"):
    fig, ax = plt.subplots(figsize=(7.0, 4.4), dpi=300)

    for dev, gg in g.groupby("device", sort=False):
        ax.scatter(
            gg["goodput_proxy_tps"],
            gg["$_per_1k_accepted_tokens"],
            marker=markers.get(dev, "o"),
            s=60,
            alpha=0.9,
            label=dev,
        )
        for _, r in gg.iterrows():
            ax.text(
                r["goodput_proxy_tps"],
                r["$_per_1k_accepted_tokens"],
                f"{r['model']} {r['quant']}",
                fontsize=7,
                alpha=0.85,
                ha="left",
                va="bottom",
            )

    ax.set_title(titles.get(target, target))
    ax.set_xlabel("Verified goodput proxy (tok/s) = draft_tps × acceptance")
    ax.set_ylabel("Verifier cost per 1k accepted tokens ($)  [lower is better]")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.legend(loc="best", frameon=True)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"{target}_pareto_goodput_vs_cost.pdf"))
    fig.savefig(os.path.join(OUT_DIR, f"{target}_pareto_goodput_vs_cost.png"))
    plt.close(fig)

# -----------------------------
# 6) Plot B: Cost efficiency bars (tokens per $)
# -----------------------------
def plot_efficiency_bars(g: pd.DataFrame, target: str):
    # (i) aggregated by model size (clean, recommended for main paper)
    agg = (
        g.groupby(["model"], as_index=False)
         .agg(tokens_per_dollar=("tokens_per_$", "mean"),
              dollars_per_1k=("$_per_1k_accepted_tokens", "mean"),
              accept=("accept", "mean"))
         .sort_values("tokens_per_dollar", ascending=False)
    )

    fig, ax = plt.subplots(figsize=(6.8, 3.8), dpi=300)
    x = np.arange(len(agg["model"]))
    ax.bar(x, agg["tokens_per_dollar"].values)
    ax.set_xticks(x)
    ax.set_xticklabels(agg["model"].tolist(), rotation=20, ha="right")
    ax.set_title(f"Cost efficiency by model (target={target})")
    ax.set_ylabel("Accepted tokens per $  [higher is better]")
    ax.grid(True, axis="y", linestyle=":", linewidth=0.8, alpha=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"{target}_cost_efficiency_bar_model.pdf"))
    fig.savefig(os.path.join(OUT_DIR, f"{target}_cost_efficiency_bar_model.png"))
    plt.close(fig)

    # (ii) per variant, grouped by device (denser; good for appendix)
    pivot = g.pivot_table(index="variant", columns="device", values="tokens_per_$", aggfunc="mean").fillna(0.0)
    pivot["__max"] = pivot.max(axis=1)
    pivot = pivot.sort_values("__max", ascending=False).drop(columns="__max")

    fig, ax = plt.subplots(figsize=(10.8, 4.2), dpi=300)
    x = np.arange(len(pivot.index))
    width = 0.38
    devices = list(pivot.columns)
    for i, dev in enumerate(devices):
        ax.bar(x + (i - (len(devices)-1)/2)*width, pivot[dev].values, width, label=dev)

    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index.tolist(), rotation=35, ha="right")
    ax.set_title(f"Cost efficiency by variant (target={target})")
    ax.set_ylabel("Accepted tokens per $  [higher is better]")
    ax.grid(True, axis="y", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"{target}_cost_efficiency_bar_variant.pdf"))
    fig.savefig(os.path.join(OUT_DIR, f"{target}_cost_efficiency_bar_variant.png"))
    plt.close(fig)

for target, g in df.groupby("target"):
    plot_efficiency_bars(g, target)

print(f"Done. All outputs saved to {OUT_DIR}/")
