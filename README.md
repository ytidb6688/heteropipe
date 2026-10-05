# heteropipe

Closed-form **makespan**, **GPipe homogeneity bias**, **phase transition**, and
**async–blocking sandwich** for heterogeneous micro-batch pipelines that serve
large language models (LLMs) on heterogeneous GPU clusters.

This is the reference implementation of the preprint

> ZHU Wenbo, *Exact Makespan Bounds and the GPipe Homogeneity Bias for
> Heterogeneous LLM Inference Pipelines*, 2026.

## What it computes

For a chain of $S$ pipeline stages with per-micro-batch service times $\tau=[\tau_0,\dots,\tau_{S-1}]$
and per-link latencies $d=[d_0,\dots,d_{S-2}]$:

| Result | Formula |
|---|---|
| Theorem 1 — communication-free | $T=(M-1)\tau_{\max}+\sum_j\tau_j$ |
| Theorem 2 — asynchronous | $T=\sum_j\tau_j+\sum_j d_j+(M-1)\tau_{\max}$ |
| Theorem 3 — blocking | $T=(M-1)\sigma_{\max}+\sum_s\sigma_s$, where $\sigma_s=\tau_s+d_s$ |
| classic GPipe (homogeneous) | $T_c=(S+M-1)\tau_{\max}$ |
| Corollary 1 — homogeneity bias | $T_c-T_{\text{free}}=\sum_j(\tau_{\max}-\tau_j)$, constant in $M$ |
| Corollary 3 — full bias | $T_c-T_{\text{async}}=\sum_j(\tau_{\max}-\tau_j)-\sum_j d_j$ |
| Corollary 4 — phase transition | $\exists s: d_s \gt \tau_{\max}-\tau_s$ |
| Corollary 5 — overlap gain | $T_{\text{block}}-T_{\text{async}}=(M-1)(\sigma_{\max}-\tau_{\max})$ |
| Proposition 1 — asymptotic throughput | async $\lim_{M\to\infty}M/T(M)=1/\tau_{\max}$ — set by the bottleneck stage alone, independent of every $d_j$;<br>blocking $\lim_{M\to\infty}M/T(M)=1/\sigma_{\max}$; the two coincide iff $d_s\le\tau_{\max}-\tau_s$ for all $s$ |
| Crossover batch count | $M^{\ast}=1+(\sum_j\tau_j+\sum_j d_j)/\tau_{\max}$ |

## Install

```bash
pip install .            # from this directory, or
pip install heteropipe   # once published to PyPI
```

No runtime dependencies — pure standard library, Python ≥ 3.8.

## Quick start

```python
from heteropipe import Pipeline

# legalone:8b on RTX 5090 / 3090-97 / 3090-98 (Sec. 7.2 of the paper)
p = Pipeline(tau=[1.13, 2.90, 2.62])

p.bias()                # 2.05   — Corollary 1, the constant GPipe bias
p.async_(M=100)         # Theorem 2 makespan at M=100
p.phase()               # False  — no communication given
p.crossover()           # fill-to-steady-state crossover

p.analyze(M=100)        # dict with every quantity at once
```

With communication:

```python
p = Pipeline(tau=[0.5, 1, 1, 5], d=[0.2, 0.1, 0.3])
p.async_(M=16)          # Theorem 2
p.block(M=16)           # Theorem 3
p.overlap_gap(M=16)     # Corollary 5 — upper bound on overlap gain
p.phase()               # Corollary 4 — communication phase transition?
p.slope(rho=0.8)        # effective steady-state slope under 80% overlap
p.throughput(blocking=True)
```

Measure the overlap ratio $\rho$ from an observed makespan $T_{\text{real}}$:

```python
rho = p.overlap_ratio(T_real=42.0, M=16)   # ρ = (T_block − T_real)/(T_block − T_async)
```

## Command line

```bash
heteropipe --tau 1.13,2.90,2.62 --M 100
heteropipe --tau 0.5,1,1,5 --d 0.2,0.1,0.3 --M 32 --rho 0.8 --json
```

## Online calculator

Two self-contained, dependency-free browser calculators implement the same
formulas:

* `calculator.html` — English version (primary)
* `calculator_zh.html` — Chinese version

Open either directly in a browser, or serve with `python -m http.server`.

## Dataset

`data/real_vs_formula.csv` is the de-identified, per-(model, node) production
trace (8,571 completed requests, 2026-09-03 → 2026-10-04) that backs §7.2 and
Figure 5 of the preprint. Host identifiers are mapped to anonymized `node-N`
labels; no IP addresses, timestamps, request IDs, or payloads are published.
See `data/DATA.md` for the data dictionary and provenance, and
`data/DATA_LICENSE.md` for terms. Regenerate with:

```bash
python scripts/extract_tau_real.py > db_facts_real_vs_formula.json
python scripts/make_dataset_csv.py
```

## Cite

```bibtex
@misc{zhu2026heteropipe,
  title   = {Exact Makespan Bounds and the GPipe Homogeneity Bias for
             Heterogeneous LLM Inference Pipelines},
  author  = {Zhu, Wenbo},
  year    = {2026},
  note    = {Preprint},
  url     = {https://github.com/ytidb6688/heteropipe}
}
```

## License

Code and documentation: **Apache 2.0** — see `LICENSE`.
Dataset (`data/`): **CC-BY-4.0** — see `data/DATA_LICENSE.md`.
