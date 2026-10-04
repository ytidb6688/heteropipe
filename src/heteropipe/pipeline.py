"""High-level ``Pipeline`` object that aggregates every closed form.

Use :class:`Pipeline` when you want all quantities at once; use the pure
functions in :mod:`heteropipe.core` when you want individual values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

from . import core


@dataclass(frozen=True)
class Pipeline:
    """A heterogeneous micro-batch pipeline (an ``S``-stage chain).

    Parameters
    ----------
    tau:
        Per-micro-batch service times of the ``S`` stages, in stage order.
    d:
        Per-link communication latencies (``S-1`` values).  Optional; defaults
        to zero (the communication-free case).

    Attributes
    ----------
    S        : number of stages
    tau_max  : bottleneck service time
    sigma    : effective blocking service times (``tau_s + d_s``)
    sigma_max: bottleneck effective service time under blocking communication
    """

    tau: Tuple[float, ...]
    d: Tuple[float, ...] = ()

    def __post_init__(self) -> None:
        tau, d = core.validate_pipeline(self.tau, self.d if self.d else None)
        object.__setattr__(self, "tau", tuple(tau))
        object.__setattr__(self, "d", tuple(d))

    # ---- structural quantities ----------------------------------------
    @property
    def S(self) -> int:
        return len(self.tau)

    @property
    def tau_max(self) -> float:
        return max(self.tau)

    @property
    def sigma(self) -> Tuple[float, ...]:
        return tuple(core.blocking_sigma(self.tau, self.d))

    @property
    def sigma_max(self) -> float:
        return max(self.sigma)

    # ---- makespan closed forms ----------------------------------------
    def free(self, M: int) -> float:
        """Theorem 1 — communication-free makespan."""
        return core.makespan_free(self.tau, M)

    def async_(self, M: int) -> float:
        """Theorem 2 — asynchronous-communication makespan."""
        return core.makespan_async(self.tau, M, self.d)

    def block(self, M: int) -> float:
        """Theorem 3 — blocking-communication makespan."""
        return core.makespan_blocking(self.tau, M, self.d)

    def gpipe(self, M: int) -> float:
        """Classic homogeneous GPipe formula ``T_c = (S+M-1)*tau_max``."""
        return core.gpipe_makespan(self.tau, M)

    # ---- corollaries / propositions ------------------------------------
    def bias(self) -> float:
        """Corollary 1 — the constant GPipe homogeneity bias (independent of ``M``)."""
        return core.homogeneity_bias(self.tau)

    def full_bias(self) -> float:
        """Corollary 3 — full bias decomposition (homogeneity minus communication)."""
        return core.full_bias_decomposition(self.tau, self.d)

    def phase(self) -> bool:
        """Corollary 4 — ``True`` iff communication enters the steady-state slope."""
        return core.phase_transition(self.tau, self.d)

    def overlap_gap(self, M: int) -> float:
        """Corollary 5 — upper bound on the overlap gain at ``M``."""
        return core.overlap_gap(self.tau, M, self.d)

    def slope(self, rho: float = 1.0) -> float:
        """Effective steady-state slope under overlap ratio ``rho``."""
        return core.effective_slope(self.tau, rho, self.d)

    def throughput(self, blocking: bool = False) -> float:
        """Proposition 1 — asymptotic throughput (async ``1/tau_max``)."""
        return core.asymptotic_throughput(self.tau, self.d, blocking)

    def crossover(self) -> float:
        """Fill-to-steady-state crossover micro-batch count ``M*``."""
        return core.crossover_batches(self.tau, self.d)

    def overlap_ratio(self, T_real: float, M: int) -> float:
        """Measured overlap ratio ``rho`` from an observed makespan ``T_real``."""
        return core.overlap_ratio(self.tau, T_real, M, self.d)

    # ---- aggregate ------------------------------------------------------
    def analyze(self, M: int, rho: Optional[float] = None) -> Dict[str, Any]:
        """Return a dict of every quantity at micro-batch count ``M``.

        ``rho`` optionally overrides the overlap ratio used for
        ``effective_slope`` (defaults to the fully-async ideal ``rho=1``).
        """
        if rho is None:
            rho = 1.0
        T_free = self.free(M)
        T_async = self.async_(M)
        T_block = self.block(M)
        T_gpipe = self.gpipe(M)
        return {
            "S": self.S,
            "M": M,
            "tau": list(self.tau),
            "d": list(self.d),
            "tau_max": self.tau_max,
            "sigma_max": self.sigma_max,
            "T_free": T_free,
            "T_async": T_async,
            "T_block": T_block,
            "T_gpipe": T_gpipe,
            "homogeneity_bias": self.bias(),
            "full_bias": self.full_bias(),
            "phase_transition": self.phase(),
            "overlap_gap": self.overlap_gap(M),
            "effective_slope": self.slope(rho),
            "asymptotic_throughput_async": self.throughput(blocking=False),
            "asymptotic_throughput_blocking": self.throughput(blocking=True),
            "crossover_batches": self.crossover(),
        }

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"Pipeline(S={self.S}, tau={list(self.tau)}, d={list(self.d)}, "
            f"tau_max={self.tau_max}, sigma_max={self.sigma_max})"
        )
