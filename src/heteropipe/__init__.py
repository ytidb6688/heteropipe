"""heteropipe — closed-form makespan for heterogeneous LLM micro-batch pipelines.

This package implements the exact closed forms of

    ZHU Wenbo, "Exact Makespan Bounds and the GPipe Homogeneity Bias for
    Heterogeneous LLM Inference Pipelines" (preprint, 2026).

It computes, for a chain of ``S`` pipeline stages with per-micro-batch service
times ``tau`` and per-link communication latencies ``d``:

* Theorem 1  — communication-free makespan
* Theorem 2  — asynchronous-communication makespan
* Theorem 3  — blocking-communication makespan
* Corollary 1 — the constant GPipe homogeneity bias  ``sum(tau_max - tau_j)``
* Corollary 3 — the full bias decomposition (homogeneity overestimate minus
  communication underestimate)
* Corollary 4 — the communication phase transition ``d_s > tau_max - tau_s``
* Corollary 5 — the async-blocking sandwich / overlap-gain bound
* Proposition 1 — asymptotic throughput (compute-balanced)
* Proposition 2 — bottleneck-position irrelevance
* the fill-to-steady-state crossover batch count ``M*``
* the overlap ratio ``rho`` (measurable from logs)

Example
-------
>>> from heteropipe import Pipeline
>>> p = Pipeline(tau=[1.13, 2.90, 2.62])   # legalone:8b on 5090 / 3090 / 3090
>>> p.bias()                              # Corollary 1, constant in M
2.05
>>> p.analyze(M=100)                      # every quantity at M=100
"""

from .core import (
    __version__,
    validate_pipeline,
    blocking_sigma,
    makespan_free,
    makespan_async,
    makespan_blocking,
    gpipe_makespan,
    homogeneity_bias,
    full_bias_decomposition,
    phase_transition,
    overlap_gap,
    effective_slope,
    overlap_ratio,
    asymptotic_throughput,
    crossover_batches,
)
from .pipeline import Pipeline

__all__ = [
    "__version__",
    "validate_pipeline",
    "blocking_sigma",
    "makespan_free",
    "makespan_async",
    "makespan_blocking",
    "gpipe_makespan",
    "homogeneity_bias",
    "full_bias_decomposition",
    "phase_transition",
    "overlap_gap",
    "effective_slope",
    "overlap_ratio",
    "asymptotic_throughput",
    "crossover_batches",
    "Pipeline",
]
