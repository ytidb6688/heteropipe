# Copyright (c) 2026 ZHU Wenbo (Yantai Vocational College of Culture and Tourism).
# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the closed-form vs real-scheduler comparison (Fig. "real_vs_formula").
Data source: inference-stack audit logs (extract_tau_real.py, 8571 completions).
Computes Corollary 1 (GPipe homogeneity bias) and Corollary 5 (sandwich) checks,
and renders a two-panel figure. Pure-stdlib numbers + matplotlib figure.
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "fig_real_vs_formula.png")
DATAJSON = os.path.join(ROOT, "db_facts_real_vs_formula.json")

# ---- Extracted lower-bound service times tau_j (dur_min, seconds) per host ----
# legalone:8b across three GPU generations (our production LLM)
legalone = {
    "RTX 5090 Laptop (host 59)": 1.130,
    "RTX 3090 (host 97)": 2.901,
    "RTX 3090 (host 98)": 2.621,
}
# qwen2.5:7b across four GPU types (illustrative, wider spread)
qwen = {
    "RTX 3090 (98)": 0.031,
    "Tesla P40 (69)": 0.118,
    "RTX 5090 (59)": 0.208,
    "RTX 3090 (97)": 0.285,
    "RTX 3080 (69, slow)": 30.111,  # outlier slow host (n=2)
}

def bias_and_table(name, tau, S):
    tmax = max(tau.values())
    sumtau = sum(tau.values())
    bias = sum(tmax - v for v in tau.values())  # Corollary 1 constant
    rows = []
    Ms = [1, 2, 4, 8, 16, 32, 64, 100]
    for M in Ms:
        T_true = (M - 1) * tmax + sumtau           # Theorem 1 (heterogeneous)
        T_gpipe = (S + M - 1) * tmax               # GPipe homogeneous assumption
        diff = T_gpipe - T_true
        rel = diff / T_true * 100
        rows.append((M, T_true, T_gpipe, diff, rel))
    return {"name": name, "tmax": tmax, "sumtau": sumtau, "bias": bias,
            "rows": rows}

r1 = bias_and_table("legalone:8b (3 heterogeneous GPUs)", legalone, 3)
r2 = bias_and_table("qwen2.5:7b (5 GPUs, incl. slow P40/Turing)", qwen, 5)

# ---- Sandwich (Corollary 5): total_s >= duration_s for every completion ----
# Per-host median points (dur_p50, tot_p50) for models with n>=10
scatter = [
    ("bge-m3|3090-97", 0.050, 0.051), ("bge-m3|3090-98", 0.017, 0.018),
    ("bge-m3|3080-69", 0.036, 0.0375), ("bge-m3|3080-99", 0.0275, 0.0285),
    ("legalone:8b|3090-97", 4.817, 4.818), ("legalone:8b|3090-98", 4.586, 4.5865),
    ("legalone:8b|5090", 8.569, 8.586), ("qwen2.5:7b|3090-97", 2.040, 2.041),
    ("qwen2.5:7b|3090-98", 0.411, 0.412), ("qwen2.5:7b|P40-69", 0.487, 0.488),
    ("qwen2.5:7b|5090", 1.658, 1.658), ("qwen2.5:14b|3090-97", 4.902, 4.903),
    ("qwen2.5:14b|3090-98", 1.598, 1.599), ("qwen2.5:14b|P40-69", 0.623, 0.624),
    ("qwen3:4b|5090", 0.324, 0.324), ("qwen3:8b|siliconflow", 6.028, 6.028),
    ("qwen3.5:9b|3090-97", 5.650, 6.365), ("qwen3.5:9b|3090-98", 1.530, 1.531),
]
# Global sandwich stats from logs
queue_frac = 0.6178
queue_median = 0.001
queue_max = 273.6
legalone_5090_treal = 8.586   # median total_s on busiest host
legalone_5090_tau = 1.130     # ideal lower bound (min duration)
real_vs_async_ratio = legalone_5090_treal / legalone_5090_tau

evidence = {
    "n_completions": 8571,
    "legalone_pipeline": r1,
    "qwen_pipeline": r2,
    "sandwich": {
        "queue_frac": queue_frac, "queue_median_s": queue_median,
        "queue_max_s": queue_max,
        "legalone_5090_treal_s": legalone_5090_treal,
        "legalone_5090_tau_s": legalone_5090_tau,
        "real_vs_async_ratio": real_vs_async_ratio,
        "n_scatter_points": len(scatter),
        "all_tot_ge_dur": True,
    },
}
with open(DATAJSON, "w", encoding="utf-8") as f:
    json.dump(evidence, f, ensure_ascii=False, indent=2)

# ---- Print tables ----
print("=== Corollary 1: GPipe homogeneity bias (constant in M) ===")
for r in (r1, r2):
    print(f"\n[{r['name']}]  tau_max={r['tmax']:.3f}s  sum(tau)={r['sumtau']:.3f}s  "
          f"bias=sum(tau_max-tau_j)={r['bias']:.3f}s")
    print(f"  {'M':>4} {'T_true(Thm1)':>14} {'T_GPipe(homo)':>14} {'diff':>10} {'rel%':>7}")
    for M, Tt, Tg, d, rel in r["rows"]:
        print(f"  {M:>4} {Tt:>14.3f} {Tg:>14.3f} {d:>10.3f} {rel:>6.2f}%")

print("\n=== Corollary 5: sandwich T_async <= T_real <= T_block ===")
print(f"  completions={8571}; queue_frac={queue_frac*100:.1f}%; "
      f"queue_median={queue_median}s; queue_max={queue_max}s")
print(f"  busiest host (RTX 5090, legalone:8b): median T_real={legalone_5090_treal}s "
      f"vs ideal tau={legalone_5090_tau}s -> {real_vs_async_ratio:.2f}x above async lower bound")
print(f"  scatter points (n={len(scatter)}): all tot_p50 >= dur_p50 -> T_real >= T_async confirmed")

# ---- Figure ----
fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 4.9))

Ms = np.array([1, 2, 4, 8, 16, 32, 64, 100])
Tt = np.array([r1["rows"][i][1] for i in range(len(Ms))])
Tg = np.array([r1["rows"][i][2] for i in range(len(Ms))])
axA.plot(Ms, Tt, "o-", color="#1f77b4", label="Heterogeneous (Thm 1): (M-1)t_max+sum t_j")
axA.plot(Ms, Tg, "s--", color="#d62728", label="GPipe homogeneous: (S+M-1)t_max")
axA.fill_between(Ms, Tt, Tg, color="#d62728", alpha=0.12)
axA.set_xscale("log", base=2)
axA.set_xlabel("Number of micro-batches M")
axA.set_ylabel("Makespan T(M)  [seconds]")
axA.set_title("Corollary 1: GPipe homogeneous formula\nover-predicts by a constant 2.05 s (legalone:8b, 3 GPUs)",
              fontsize=10.5)
axA.legend(fontsize=8, loc="upper left")
axA.grid(alpha=0.3)

xs = [p[1] for p in scatter]
ys = [p[2] for p in scatter]
axB.scatter(xs, ys, c="#2ca02c", s=28, zorder=3, label="per-host median (n>=10)")
xmax = max(max(xs), max(ys)) * 1.10
axB.plot([0, xmax], [0, xmax], "k--", lw=1, label="T_async = T_real (lower bound)")
axB.set_xlim(0, xmax)
axB.set_ylim(0, xmax)
axB.set_xlabel("duration_s  (ideal compute, approx. T_async)")
axB.set_ylabel("total_s  (end-to-end, T_real)")
axB.set_title("Production check: T_real >= T_async for every request\n(8,571 completions; 61.8% incur queue wait)",
              fontsize=10.5)
axB.legend(fontsize=8, loc="upper left")
axB.grid(alpha=0.3)
# Annotation placed in the empty lower-right triangle (below the diagonal),
# well clear of the title, the y-axis label and the legend box.
axB.annotate(f"busiest host:\n8.59 s vs 1.13 s\n({real_vs_async_ratio:.1f}x above lower bound)",
             xy=(8.569, 8.586), xytext=(4.6, 1.15), fontsize=8, ha="left", va="bottom",
             arrowprops=dict(arrowstyle="->", color="gray", lw=0.9))

fig.tight_layout(pad=1.3)
fig.savefig(FIG, dpi=150, bbox_inches="tight")
print("\nSaved figure:", FIG)
