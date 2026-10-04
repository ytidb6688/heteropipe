# Copyright (c) 2026 ZHU Wenbo (Yantai Vocational College of Culture and Tourism).
# SPDX-License-Identifier: Apache-2.0

"""Command-line interface for ``heteropipe``.

Examples
--------
.. code-block:: bash

    # legalone:8b on RTX 5090 / 3090-97 / 3090-98, M=100
    heteropipe --tau 1.13,2.90,2.62 --M 100

    # with per-link latencies (S-1 values) and JSON output
    heteropipe --tau 0.5,1,1,5 --d 0.2,0.1,0.3 --M 32 --rho 0.8 --json
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from .pipeline import Pipeline


def _parse_floats(s: str, name: str) -> List[float]:
    if s.strip() == "":
        return []
    try:
        return [float(x) for x in s.split(",")]
    except ValueError:
        raise SystemExit(f"invalid {name}: expected comma-separated numbers, got {s!r}")


def _fmt(v: float) -> str:
    if abs(v) >= 1000:
        return f"{v:,.3f}"
    return f"{v:.3f}"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="heteropipe",
        description="Closed-form makespan / bias / phase-transition / sandwich "
        "for heterogeneous LLM micro-batch pipelines.",
    )
    p.add_argument("--tau", required=True,
                   help="comma-separated per-stage service times (S values)")
    p.add_argument("--d", default="",
                   help="comma-separated per-link latencies (S-1 values); omit for none")
    p.add_argument("--M", type=int, default=8, help="number of micro-batches (default 8)")
    p.add_argument("--rho", type=float, default=None,
                   help="overlap ratio in [0,1] for the effective slope (default 1.0)")
    p.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    tau = _parse_floats(args.tau, "tau")
    d = _parse_floats(args.d, "d") or None

    try:
        pipe = Pipeline(tau=tau, d=d)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    rho = args.rho if args.rho is not None else 1.0
    try:
        res = pipe.analyze(M=args.M, rho=rho)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(res, indent=2))
        return 0

    print(f"heteropipe  S={res['S']} stages, M={res['M']} micro-batches")
    print(f"  tau      = {res['tau']}")
    print(f"  d        = {res['d']}")
    print(f"  tau_max  = {_fmt(res['tau_max'])}")
    print(f"  sigma_max= {_fmt(res['sigma_max'])}")
    print("-" * 62)
    print(f"  T_free   (Theorem 1) = {_fmt(res['T_free'])}")
    print(f"  T_async  (Theorem 2) = {_fmt(res['T_async'])}")
    print(f"  T_block  (Theorem 3) = {_fmt(res['T_block'])}")
    print(f"  T_gpipe  (classic)   = {_fmt(res['T_gpipe'])}")
    print("-" * 62)
    print(f"  homogeneity bias (Cor 1)  = {_fmt(res['homogeneity_bias'])}")
    print(f"  full bias        (Cor 3)  = {_fmt(res['full_bias'])}")
    print(f"  phase transition (Cor 4)  = {res['phase_transition']}")
    print(f"  overlap gap      (Cor 5)  = {_fmt(res['overlap_gap'])}")
    print(f"  effective slope  (rho)    = {_fmt(res['effective_slope'])}")
    print(f"  throughput async          = {_fmt(res['asymptotic_throughput_async'])}")
    print(f"  throughput blocking       = {_fmt(res['asymptotic_throughput_blocking'])}")
    print(f"  crossover batches M*      = {_fmt(res['crossover_batches'])}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
