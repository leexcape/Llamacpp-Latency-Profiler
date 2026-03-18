#!/usr/bin/env python3
"""Enumerate ALL (draft_model, quantization, K) configurations for each
(target LLM family, edge device) and find the optimum under three metrics:

  1. **Goodput** (tok/s)   = (K·α(K) + 1) / (K / draft_tps + T_verify)
  2. **Cost efficiency**   = (K·α(K) + 1) / (K · p)    [accepted tokens per $]
     equivalently: minimise  cost_per_accepted = K·p / (K·α(K)+1)
  3. **Energy efficiency**  = P·(K / draft_tps) / (K·α(K)+1)   [J per accepted tok]
     (only local drafting energy; T_verify is remote)

Usage:
    conda run -n transformers_env python find_optimal_config.py
"""

import pandas as pd
import numpy as np
from itertools import product

# ── Parameters ──────────────────────────────────────────────────────────────
T_VERIFY = 0.5          # verification latency (seconds)
K_RANGE  = range(2, 11) # K = 2, 3, …, 10

# Verifier pricing (output tokens, $/1M tokens)
PRICE_OUT_PER_M = {
    "llama_target_70b": 0.90,   # Fireworks tier for 70B-class
    "qwen_target_32b":  0.59,   # Groq Qwen3-32B output
}

# ── Acceptance-rate data ────────────────────────────────────────────────────
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

# ── Draft model mapping ────────────────────────────────────────────────────
# (target_key, model_short, flavor) → (target_hf, draft_hf)
DRAFT_ACC_MAP = {
    ("llama_target_70b", "1b", "base"): ("meta-llama/Meta-Llama-3.1-70B-Instruct", "meta-llama/Llama-3.2-1B"),
    ("llama_target_70b", "1b", "inst"): ("meta-llama/Meta-Llama-3.1-70B-Instruct", "meta-llama/Llama-3.2-1B-Instruct"),
    ("llama_target_70b", "3b", "inst"): ("meta-llama/Meta-Llama-3.1-70B-Instruct", "meta-llama/Llama-3.2-3B-Instruct"),
    ("llama_target_70b", "8b", "inst"): ("meta-llama/Meta-Llama-3.1-70B-Instruct", "meta-llama/Meta-Llama-3.1-8B-Instruct"),
    ("qwen_target_32b", "0.6b", "base"): ("Qwen/Qwen3-32B", "Qwen/Qwen3-0.6B"),
    ("qwen_target_32b", "1.7b", "base"): ("Qwen/Qwen3-32B", "Qwen/Qwen3-1.7B"),
    ("qwen_target_32b", "4b",   "base"): ("Qwen/Qwen3-32B", "Qwen/Qwen3-4B"),
    ("qwen_target_32b", "8b",   "base"): ("Qwen/Qwen3-32B", "Qwen/Qwen3-8B"),
}

# ── Device speed data  (tok/s) ─────────────────────────────────────────────
# key = (target, model_size, quant, flavor)
SPEED = {
    "RPi 4B": {
        ("llama_target_70b", "1b", "q4_k_m", "inst"): 4.14,
        ("llama_target_70b", "1b", "q4_k_m", "base"): 4.18,
        ("llama_target_70b", "3b", "q4_k_m", "inst"): 1.68,
        ("llama_target_70b", "3b", "q8_0",   "inst"): 1.04,
        ("llama_target_70b", "8b", "q4_k_m", "inst"): 0.72,
        ("qwen_target_32b", "0.6b", "q4_k_m", "base"): 7.84,
        ("qwen_target_32b", "0.6b", "q8_0",   "base"): 4.92,
        ("qwen_target_32b", "1.7b", "q4_k_m", "base"): 3.02,
        ("qwen_target_32b", "1.7b", "q8_0",   "base"): 1.89,
        ("qwen_target_32b", "4b",   "q4_k_m", "base"): 1.34,
        ("qwen_target_32b", "4b",   "q8_0",   "base"): 0.83,
        ("qwen_target_32b", "8b",   "q4_k_m", "base"): 0.72,
        ("qwen_target_32b", "8b",   "q6_k",   "base"): 0.55,
    },
    "RPi 5": {
        ("llama_target_70b", "1b", "q4_k_m", "inst"): 14.47,
        ("llama_target_70b", "1b", "q4_k_m", "base"): 12.86,
        ("llama_target_70b", "3b", "q4_k_m", "inst"): 4.68,
        ("llama_target_70b", "3b", "q8_0",   "inst"): 2.37,
        ("llama_target_70b", "8b", "q4_k_m", "inst"): 1.77,
        ("qwen_target_32b", "0.6b", "q4_k_m", "base"): 18.18,
        ("qwen_target_32b", "0.6b", "q8_0",   "base"): 10.38,
        ("qwen_target_32b", "1.7b", "q4_k_m", "base"): 7.15,
        ("qwen_target_32b", "1.7b", "q8_0",   "base"): 4.01,
        ("qwen_target_32b", "4b",   "q4_k_m", "base"): 3.11,
        ("qwen_target_32b", "4b",   "q8_0",   "base"): 1.93,
        ("qwen_target_32b", "8b",   "q4_k_m", "base"): 1.78,
        ("qwen_target_32b", "8b",   "q6_k",   "base"): 1.68,
    },
    "Jetson": {
        ("llama_target_70b", "1b", "q4_k_m", "inst"): 93.14,
        ("llama_target_70b", "1b", "q4_k_m", "base"): 92.94,
        ("llama_target_70b", "3b", "q4_k_m", "inst"): 42.52,
        ("llama_target_70b", "3b", "q8_0",   "inst"): 31.36,
        ("llama_target_70b", "8b", "q4_k_m", "inst"): 25.07,
        ("qwen_target_32b", "0.6b", "q4_k_m", "base"): 98.63,
        ("qwen_target_32b", "0.6b", "q8_0",   "base"): 80.21,
        ("qwen_target_32b", "1.7b", "q4_k_m", "base"): 65.31,
        ("qwen_target_32b", "1.7b", "q8_0",   "base"): 46.64,
        ("qwen_target_32b", "4b",   "q4_k_m", "base"): 33.70,
        ("qwen_target_32b", "4b",   "q8_0",   "base"): 23.02,
        ("qwen_target_32b", "8b",   "q4_k_m", "base"): 24.03,
        ("qwen_target_32b", "8b",   "q6_k",   "base"): 18.55,
    },
}

# ── Power data (W) for energy calculation ──────────────────────────────────
# key = (device, target, model_size, quant, flavor) → power_avg_w
_pi5_csv = pd.read_csv("results/pi5_llamacpp_profile_19models.csv")
_jet_csv = pd.read_csv("results/jetson_agx_orin_llamacpp_profile_19models.csv")

GGUF_MAP = {
    "llama-3.2-1b-q4_k_m.gguf":               ("llama_target_70b", "1b", "q4_k_m", "base"),
    "llama-3.2-1b-instruct-q4_k_m.gguf":      ("llama_target_70b", "1b", "q4_k_m", "inst"),
    "llama-3.2-3b-instruct-q4_k_m.gguf":      ("llama_target_70b", "3b", "q4_k_m", "inst"),
    "Llama-3.2-3B-Instruct-Q8_0.gguf":        ("llama_target_70b", "3b", "q8_0",   "inst"),
    "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf": ("llama_target_70b", "8b", "q4_k_m", "inst"),
    "Qwen3-0.6B-Q4_K_M.gguf":                 ("qwen_target_32b", "0.6b", "q4_k_m", "base"),
    "Qwen3-0.6B-Q8_0.gguf":                   ("qwen_target_32b", "0.6b", "q8_0",   "base"),
    "Qwen3-1.7B-Q4_K_M.gguf":                 ("qwen_target_32b", "1.7b", "q4_k_m", "base"),
    "Qwen3-1.7B-Q8_0.gguf":                   ("qwen_target_32b", "1.7b", "q8_0",   "base"),
    "Qwen3-4B-Q4_K_M.gguf":                   ("qwen_target_32b", "4b",   "q4_k_m", "base"),
    "Qwen3-4B-Q8_0.gguf":                     ("qwen_target_32b", "4b",   "q8_0",   "base"),
    "Qwen3-8B-Q4_K_M.gguf":                   ("qwen_target_32b", "8b",   "q4_k_m", "base"),
    "Qwen3-8B-Q6_K.gguf":                     ("qwen_target_32b", "8b",   "q6_k",   "base"),
}
DEVICE_CSV_MAP = {"raspberry_pi_5": "RPi 5", "jetson_agx_orin": "Jetson"}

POWER = {}  # (device, target, model_size, quant, flavor) → power_avg_w
for csv_df in [_pi5_csv, _jet_csv]:
    for _, r in csv_df.iterrows():
        mapping = GGUF_MAP.get(r["model_name"])
        if mapping is None:
            continue
        dev = DEVICE_CSV_MAP.get(r["device"])
        if dev is None:
            continue
        target, model_size, quant, flavor = mapping
        POWER[(dev, target, model_size, quant, flavor)] = float(r["power_avg_w"])

# ── Display names ───────────────────────────────────────────────────────────
def _model_display(model_size, quant, flavor):
    """Human-readable model variant name."""
    fam = {"inst": "-Instruct", "base": ""}[flavor]
    return f"{model_size.upper()}{fam} ({quant.upper()})"

TARGET_DISPLAY = {
    "llama_target_70b": "Llama-3.1-70B",
    "qwen_target_32b":  "Qwen3-32B",
}

# ── Enumerate all configurations ───────────────────────────────────────────
TARGETS = ["llama_target_70b", "qwen_target_32b"]
DEVICES = ["RPi 4B", "RPi 5", "Jetson"]

all_rows = []

for target in TARGETS:
    price_per_M = PRICE_OUT_PER_M[target]
    for device in DEVICES:
        speed_dict = SPEED[device]
        for (tgt, model_size, quant, flavor), tps in speed_dict.items():
            if tgt != target:
                continue
            # look up acceptance rate curve
            hf_key = DRAFT_ACC_MAP.get((target, model_size, flavor))
            if hf_key is None:
                continue
            alpha_curve = ACC_CURVES.get(hf_key)
            if alpha_curve is None:
                continue

            power_key = (device, target, model_size, quant, flavor)
            power_w = POWER.get(power_key)  # may be None for RPi 4B

            for K in K_RANGE:
                alpha = alpha_curve.get(K)
                if alpha is None:
                    continue

                accepted_per_round = K * alpha + 1.0
                draft_time = K / tps
                round_time = draft_time + T_VERIFY

                goodput = accepted_per_round / round_time

                # Cost: tokens sent per round = K (verified by server)
                # Cost per round = K * price_per_M / 1e6
                # Accepted per round = K*α(K)+1
                # Cost per accepted token = K * price_per_M / (1e6 * (K*α(K)+1))
                # tokens_per_dollar = (K*α(K)+1) / (K * price_per_M / 1e6)
                #                   = (K*α(K)+1) * 1e6 / (K * price_per_M)
                tokens_per_dollar = accepted_per_round * 1e6 / (K * price_per_M)
                cost_per_1k = 1000.0 / tokens_per_dollar

                # Energy: only local drafting
                if power_w is not None:
                    j_per_accepted = power_w * draft_time / accepted_per_round
                else:
                    j_per_accepted = None

                variant = _model_display(model_size, quant, flavor)

                all_rows.append({
                    "target": target,
                    "device": device,
                    "model_size": model_size,
                    "quant": quant,
                    "flavor": flavor,
                    "variant": variant,
                    "K": K,
                    "alpha": alpha,
                    "draft_tps": tps,
                    "power_w": power_w,
                    "accepted_per_round": accepted_per_round,
                    "goodput": goodput,
                    "tokens_per_dollar": tokens_per_dollar,
                    "cost_per_1k": cost_per_1k,
                    "j_per_accepted": j_per_accepted,
                })

df = pd.DataFrame(all_rows)

# ── Find optima ─────────────────────────────────────────────────────────────
print("=" * 100)
print("OPTIMAL CONFIGURATIONS BY (Target, Device, Metric)")
print("=" * 100)

metrics = [
    ("goodput",           "max", "Goodput (tok/s)",        "goodput"),
    ("tokens_per_dollar", "max", "Cost Eff. (tok/$)",      "tokens_per_dollar"),
    ("j_per_accepted",    "min", "Energy Eff. (J/tok)",    "j_per_accepted"),
]

def _fmt_energy(v):
    return "N/A" if v is None else f"{v:.4f}"

results = []

for target in TARGETS:
    for device in DEVICES:
        subset = df[(df["target"] == target) & (df["device"] == device)]
        if subset.empty:
            continue
        print(f"\n{'─' * 80}")
        print(f"  Target: {TARGET_DISPLAY[target]:20s}   Device: {device}")
        print(f"{'─' * 80}")

        for col, direction, label, _ in metrics:
            if col == "j_per_accepted" and device == "RPi 4B":
                print(f"  {label:25s}:  N/A (no power data for RPi 4B)")
                results.append({
                    "target": target, "device": device, "metric": label,
                    "variant": "N/A", "K": None, "alpha": None,
                    "value": None, "goodput": None,
                    "tokens_per_dollar": None, "j_per_accepted": None,
                })
                continue

            valid = subset.dropna(subset=[col])
            if valid.empty:
                continue

            if direction == "max":
                best_idx = valid[col].idxmax()
            else:
                best_idx = valid[col].idxmin()

            best = valid.loc[best_idx]

            print(f"  {label:25s}:  {best['variant']:30s}  K={int(best['K']):2d}  "
                  f"α={best['alpha']:.3f}  "
                  f"G={best['goodput']:.2f} tok/s  "
                  f"Cost={best['tokens_per_dollar']/1e3:.0f}K tok/$  "
                  f"Energy={_fmt_energy(best['j_per_accepted'])} J/tok")

            results.append({
                "target": target, "device": device, "metric": label,
                "variant": best["variant"],
                "K": int(best["K"]),
                "alpha": best["alpha"],
                "value": best[col],
                "goodput": best["goodput"],
                "tokens_per_dollar": best["tokens_per_dollar"],
                "j_per_accepted": best["j_per_accepted"],
            })

# ── WHY K=2 is optimal for cost AND energy ──────────────────────────────────
# Both metrics share the same K-dependent factor:
#   Cost:   η_cost = (α(K) + 1/K) / p          → maximise (α(K) + 1/K)
#   Energy: E      = (P/v_d) / (α(K) + 1/K)    → maximise (α(K) + 1/K)
# Since α(K) decreases with K AND 1/K decreases with K,
# their sum (α(K) + 1/K) is maximised at K=2 for every draft model.
print("\n\n" + "=" * 110)
print("WHY K=2 IS OPTIMAL FOR BOTH COST AND ENERGY")
print("Both depend on the same factor: α(K) + 1/K   (higher = better)")
print("=" * 110)

for target in TARGETS:
    price = PRICE_OUT_PER_M[target]
    print(f"\n  Target: {TARGET_DISPLAY[target]} (price = ${price}/M output tokens)")
    for (tgt, model_size, flavor), hf_key in DRAFT_ACC_MAP.items():
        if tgt != target:
            continue
        alpha_curve = ACC_CURVES.get(hf_key)
        if alpha_curve is None:
            continue
        print(f"\n    Draft: {model_size} ({flavor})")
        print(f"    {'K':>4s}  {'α(K)':>8s}  {'1/K':>8s}  {'α(K)+1/K':>10s}  {'tok/$':>12s}")
        for K in sorted(alpha_curve.keys()):
            a = alpha_curve[K]
            bonus = 1.0 / K
            factor = a + bonus
            tok_per_dollar = factor * 1e6 / price
            print(f"    {K:4d}  {a:8.4f}  {bonus:8.4f}  {factor:10.4f}  {tok_per_dollar:12.0f}")

# ── Table 2 style: one row per (target, device, objective), ALL 3 metrics ──
print("\n\n" + "=" * 120)
print("TABLE 2 STYLE: Optimal configs with ALL metrics shown (for comparison)")
print("=" * 120)

# Collect best values per (target, device) to bold them later
best_vals = {}  # (target, device) → {"goodput": max_G, "tokens_per_dollar": max_C, "j_per_accepted": min_E}
for target in TARGETS:
    for device in DEVICES:
        rows_here = [r for r in results if r["target"] == target and r["device"] == device and r["K"] is not None]
        if not rows_here:
            continue
        g_vals = [r["goodput"] for r in rows_here if r["goodput"] is not None]
        c_vals = [r["tokens_per_dollar"] for r in rows_here if r["tokens_per_dollar"] is not None]
        e_vals = [r["j_per_accepted"] for r in rows_here if r["j_per_accepted"] is not None]
        best_vals[(target, device)] = {
            "goodput": max(g_vals) if g_vals else None,
            "tokens_per_dollar": max(c_vals) if c_vals else None,
            "j_per_accepted": min(e_vals) if e_vals else None,
        }

METRIC_LABELS = {
    "Goodput (tok/s)":     "Max Goodput",
    "Cost Eff. (tok/$)":   "Min Cost/tok",
    "Energy Eff. (J/tok)": "Min Energy",
}

for target in TARGETS:
    print(f"\n  Target: {TARGET_DISPLAY[target]}")
    print(f"  {'Device':10s}  {'Objective':15s}  {'Configuration':30s}  {'K':>3s}  "
          f"{'G (tok/s)':>10s}  {'η_cost (K tok/$)':>17s}  {'E (J/tok)':>10s}")
    print("  " + "─" * 105)
    for device in DEVICES:
        first_in_device = True
        for metric_name in ["Goodput (tok/s)", "Cost Eff. (tok/$)", "Energy Eff. (J/tok)"]:
            row = [r for r in results
                   if r["target"] == target and r["device"] == device and r["metric"] == metric_name]
            if not row:
                continue
            r = row[0]
            if r["K"] is None:
                # N/A row (e.g. energy for RPi 4B)
                dev_str = device if first_in_device else ""
                print(f"  {dev_str:10s}  {METRIC_LABELS[metric_name]:15s}  {'---':30s}  {'---':>3s}  "
                      f"{'---':>10s}  {'---':>17s}  {'---':>10s}")
                first_in_device = False
                continue

            bv = best_vals.get((target, device), {})

            # Format each metric, marking with * if it's the best in this (target, device)
            g_val = r["goodput"]
            c_val = r["tokens_per_dollar"]
            e_val = r["j_per_accepted"]

            g_str = f"{g_val:.2f}"
            if bv.get("goodput") is not None and abs(g_val - bv["goodput"]) < 0.001:
                g_str = f"**{g_val:.2f}**"

            c_str = f"{c_val/1e3:.0f}K"
            if bv.get("tokens_per_dollar") is not None and abs(c_val - bv["tokens_per_dollar"]) < 0.1:
                c_str = f"**{c_val/1e3:.0f}K**"

            if e_val is not None:
                e_str = f"{e_val:.2f}"
                if bv.get("j_per_accepted") is not None and abs(e_val - bv["j_per_accepted"]) < 0.0001:
                    e_str = f"**{e_val:.2f}**"
            else:
                e_str = "---"

            dev_str = device if first_in_device else ""
            print(f"  {dev_str:10s}  {METRIC_LABELS[metric_name]:15s}  {r['variant']:30s}  {r['K']:3d}  "
                  f"{g_str:>10s}  {c_str:>17s}  {e_str:>10s}")
            first_in_device = False

print("\nDone.")
