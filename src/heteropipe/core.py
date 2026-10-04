# Copyright (c) 2026 ZHU Wenbo (Yantai Vocational College of Culture and Tourism).
# SPDX-License-Identifier: MIT

"""Core closed-form formulas.

Notation (identical to the reference preprint):

* ``tau[j]`` — per-micro-batch service time of stage ``j``, ``j = 0 .. S-1``.
* ``d[j]``   — communication latency of the link ``j -> j+1``,
  ``j = 0 .. S-2`` (there are ``S-1`` links).
* ``M``      — number of micro-batches.

The formulas:

* Theorem 1 (communication-free):
    ``T = (M-1)*tau_max + sum(tau)``
* Theorem 2 (asynchronous):
    ``T = sum(tau) + sum(d) + (M-1)*tau_max``
* Theorem 3 (blocking), with ``sigma_s = tau_s + d_s`` (``s < S-1``) and
  ``sigma_{S-1} = tau_{S-1}``:
    ``T = (M-1)*sigma_max + sum(sigma)``
* classic homogeneous GPipe formula:
    ``T_c = (S + M - 1) * tau_max``
* Corollary 1 (homogeneity bias):
    ``T_c - T_free = sum(tau_max - tau_j)``  (constant in ``M``)
* Corollary 2 (communication is additive): the steady-state slope is
  ``tau_max`` regardless of ``d``.
* Corollary 3 (full bias decomposition):
    ``T_c - T_async = sum(tau_max - tau_j) - sum(d_j)``
* Corollary 4 (phase transition): blocking slope exceeds ``tau_max`` iff
  some ``d_s > tau_max - tau_s``.
* Corollary 5 (sandwich / overlap gain):
    ``T_block - T_async = (M-1)*(sigma_max - tau_max)``
* Proposition 1 (asymptotic throughput): async ``1/tau_max``,
  blocking ``1/sigma_max``.
* Proposition 2 (bottleneck-position irrelevance): free/async makespan
  depends on the bottleneck only through ``tau_max``.
* Crossover: ``M* = 1 + (sum(tau) + sum(d)) / tau_max``.
* Overlap ratio: ``rho = (T_block - T_real) / (T_block - T_async)``.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

__version__ = "0.1.0"

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
]


def validate_pipeline(
    tau: Sequence[float], d: Optional[Sequence[float]] = None
) -> Tuple[List[float], List[float]]:
    """Validate and normalize ``tau`` and ``d``.

    Returns ``(tau_list, d_list)`` where ``d_list`` always has length
    ``S-1`` (zero-filled when ``d`` is omitted).
    """
    tau = [float(t) for t in tau]
    if not tau:
        raise ValueError("tau must be non-empty (at least one stage)")
    if any(t < 0 for t in tau):
        raise ValueError("tau values must be non-negative")

    S = len(tau)
    if d is None:
        d = [0.0] * (S - 1)
    else:
        d = [float(x) for x in d]
        if len(d) != S - 1:
            raise ValueError(
                f"d must have S-1 = {S - 1} links (one per stage boundary), "
                f"got {len(d)}"
            )
        if any(x < 0 for x in d):
            raise ValueError("d values must be non-negative")

    return tau, d


def blocking_sigma(tau: Sequence[float], d: Optional[Sequence[float]] = None) -> List[float]:
    """Effective service times under blocking communication (Theorem 3).

    ``sigma_s = tau_s + d_s`` for ``s < S-1``; ``sigma_{S-1} = tau_{S-1}``
    (the last stage sends nothing).
    """
    tau, d = validate_pipeline(tau, d)
    return [tau[s] + d[s] for s in range(len(tau) - 1)] + [tau[-1]]


def makespan_free(tau: Sequence[float], M: int) -> float:
    """Theorem 1 — communication-free makespan.

    ``T(M) = (M-1)*tau_max + sum(tau)``.
    """
    tau, _ = validate_pipeline(tau)
    _check_M(M)
    tau_max = max(tau)
    return (M - 1) * tau_max + sum(tau)


def makespan_async(
    tau: Sequence[float], M: int, d: Optional[Sequence[float]] = None
) -> float:
    """Theorem 2 — asynchronous-communication makespan.

    ``T(M) = sum(tau) + sum(d) + (M-1)*tau_max``.
    """
    tau, d = validate_pipeline(tau, d)
    _check_M(M)
    return sum(tau) + sum(d) + (M - 1) * max(tau)


def makespan_blocking(
    tau: Sequence[float], M: int, d: Optional[Sequence[float]] = None
) -> float:
    """Theorem 3 — blocking-communication makespan.

    ``T(M) = (M-1)*sigma_max + sum(sigma)``.
    """
    tau, d = validate_pipeline(tau, d)
    _check_M(M)
    sigma = blocking_sigma(tau, d)
    return (M - 1) * max(sigma) + sum(sigma)


def gpipe_makespan(tau: Sequence[float], M: int) -> float:
    """The classic homogeneous GPipe formula ``T_c = (S + M - 1)*tau_max``.

    This assigns every stage the bottleneck time ``tau_max``.
    """
    tau, _ = validate_pipeline(tau)
    _check_M(M)
    return (len(tau) + M - 1) * max(tau)


def homogeneity_bias(tau: Sequence[float]) -> float:
    """Corollary 1 — the constant GPipe homogeneity bias.

    ``T_c - T_free = sum(tau_max - tau_j)``, independent of ``M``.
    Zero iff all stages are perfectly balanced.
    """
    tau, _ = validate_pipeline(tau)
    tau_max = max(tau)
    return sum(tau_max - t for t in tau)


def full_bias_decomposition(
    tau: Sequence[float], d: Optional[Sequence[float]] = None
) -> float:
    """Corollary 3 — full bias decomposition of the classic formula.

    ``T_c - T_async = sum(tau_max - tau_j) - sum(d_j)``.

    The first term is the homogeneity *overestimate*; the second is the
    communication *underestimate*.  When the result is negative, the classic
    formula underestimates makespan (a sign reversal absent from the
    communication-free case).
    """
    tau, d = validate_pipeline(tau, d)
    return homogeneity_bias(tau) - sum(d)


def phase_transition(tau: Sequence[float], d: Optional[Sequence[float]] = None) -> bool:
    """Corollary 4 — does communication enter the steady-state slope?

    Returns ``True`` iff some ``d_s > tau_max - tau_s`` (i.e. blocking
    ``sigma_max > tau_max``), the condition under which a link latency grows
    from an additive constant into a throughput bottleneck.
    """
    tau, d = validate_pipeline(tau, d)
    tau_max = max(tau)
    for s in range(len(d)):
        if d[s] > tau_max - tau[s]:
            return True
    return False


def overlap_gap(
    tau: Sequence[float], M: int, d: Optional[Sequence[float]] = None
) -> float:
    """Corollary 5 — the async-blocking gap (upper bound on overlap gain).

    ``T_block - T_async = (M-1)*(sigma_max - tau_max) >= 0``.
    """
    tau, d = validate_pipeline(tau, d)
    _check_M(M)
    sigma_max = max(blocking_sigma(tau, d))
    return (M - 1) * (sigma_max - max(tau))


def effective_slope(
    tau: Sequence[float],
    rho: float = 1.0,
    d: Optional[Sequence[float]] = None,
) -> float:
    """Steady-state slope under overlap ratio ``rho`` in [0, 1].

    ``tau_max + (1-rho)*(sigma_max - tau_max)``.  ``rho=1`` recovers the
    async slope ``tau_max``; ``rho=0`` the blocking slope ``sigma_max``.
    """
    tau, d = validate_pipeline(tau, d)
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must be in [0, 1]")
    sigma_max = max(blocking_sigma(tau, d))
    return max(tau) + (1.0 - rho) * (sigma_max - max(tau))


def overlap_ratio(
    tau: Sequence[float],
    T_real: float,
    M: int,
    d: Optional[Sequence[float]] = None,
) -> float:
    """Measured overlap ratio ``rho = (T_block - T_real)/(T_block - T_async)``.

    ``T_real`` is the observed makespan.  Returns ``1.0`` in the degenerate
    case where no overlap gain exists (``sigma_max == tau_max``).
    """
    tau, d = validate_pipeline(tau, d)
    _check_M(M)
    T_block = makespan_blocking(tau, M, d)
    T_async = makespan_async(tau, M, d)
    denom = T_block - T_async
    if denom == 0.0:
        return 1.0
    return (T_block - T_real) / denom


def asymptotic_throughput(
    tau: Sequence[float],
    d: Optional[Sequence[float]] = None,
    blocking: bool = False,
) -> float:
    """Proposition 1 — asymptotic throughput as ``M -> infinity``.

    Async: ``1/tau_max`` (compute-balanced, independent of ``d``).
    Blocking: ``1/sigma_max``.
    """
    tau, d = validate_pipeline(tau, d)
    if blocking:
        return 1.0 / max(blocking_sigma(tau, d))
    return 1.0 / max(tau)


def crossover_batches(tau: Sequence[float], d: Optional[Sequence[float]] = None) -> float:
    """Fill-to-steady-state crossover micro-batch count ``M*``.

    ``M* = 1 + (sum(tau) + sum(d)) / tau_max``.  Below ``M*`` the fill cost
    dominates; above it the steady-state throughput is approached.
    """
    tau, d = validate_pipeline(tau, d)
    tau_max = max(tau)
    return 1.0 + (sum(tau) + sum(d)) / tau_max


def _check_M(M: int) -> None:
    if not isinstance(M, int) or M < 1:
        raise ValueError("M must be a positive integer")
