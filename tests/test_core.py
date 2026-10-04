"""Tests for the closed-form formulas.

These tests pin the formulas to the paper's own worked examples:

* illustrative 4-stage partition ``tau=[0.5, 1, 1, 5]`` (Sec. 3.1):
  ``T(1)=7.5``, ``T_c(1)=20``, bias ``12.5``, ``T(8)=42.5``, ``M*=2.5``.
* production ``legalone:8b`` pipeline ``tau=[1.13, 2.90, 2.62]`` (Sec. 7.2):
  bias ``2.05``.
"""

from __future__ import annotations

from heteropipe import Pipeline, core


def assert_close(a, b, tol=1e-9):
    assert abs(a - b) < tol, f"{a} != {b}"


def test_illustrative_M1():
    tau = [0.5, 1, 1, 5]
    assert_close(core.makespan_free(tau, 1), 7.5)
    assert_close(core.gpipe_makespan(tau, 1), 20.0)
    assert_close(core.homogeneity_bias(tau), 12.5)


def test_illustrative_M8():
    tau = [0.5, 1, 1, 5]
    assert_close(core.makespan_free(tau, 8), 42.5)
    assert_close(core.gpipe_makespan(tau, 8), 55.0)
    # absolute gap is constant, equal to the bias
    assert_close(55.0 - 42.5, core.homogeneity_bias(tau))


def test_crossover():
    tau = [0.5, 1, 1, 5]
    assert_close(core.crossover_batches(tau), 2.5)


def test_legalone_bias():
    tau = [1.13, 2.90, 2.62]
    assert_close(core.homogeneity_bias(tau), 2.05)


def test_async_is_free_plus_comm():
    tau = [0.5, 1, 1, 5]
    d = [0.2, 0.1, 0.3]
    M = 16
    assert_close(core.makespan_async(tau, M, d),
                 core.makespan_free(tau, M) + sum(d))


def test_blocking_reduces_to_free_when_d_zero():
    tau = [0.5, 1, 1, 5]
    M = 16
    assert_close(core.makespan_blocking(tau, M), core.makespan_free(tau, M))


# A communication case whose bottleneck is NOT the last stage, so that
# sigma_max = tau_0 + d_0 = 5.2 > tau_max = 5 and the sandwich is non-degenerate.
COMM_TAU = [5.0, 1.0, 1.0, 0.5]
COMM_D = [0.2, 0.1, 0.3]


def test_sandwich_ordering():
    M = 16
    T_async = core.makespan_async(COMM_TAU, M, COMM_D)
    T_block = core.makespan_blocking(COMM_TAU, M, COMM_D)
    assert T_async < T_block  # strict, since sigma_max > tau_max
    assert_close(core.overlap_gap(COMM_TAU, M, COMM_D), T_block - T_async)


def test_phase_transition_threshold():
    tau = [1.0, 2.0, 1.0]
    # slack of stage 0 is tau_max - tau_0 = 1.0; d_0 = 1.1 crosses it
    assert core.phase_transition(tau, [1.1, 0.0]) is True
    assert core.phase_transition(tau, [0.9, 0.0]) is False


def test_overlap_ratio_endpoints():
    M = 16
    T_async = core.makespan_async(COMM_TAU, M, COMM_D)
    T_block = core.makespan_blocking(COMM_TAU, M, COMM_D)
    assert_close(core.overlap_ratio(COMM_TAU, T_async, M, COMM_D), 1.0)
    assert_close(core.overlap_ratio(COMM_TAU, T_block, M, COMM_D), 0.0)


def test_effective_slope_endpoints():
    assert_close(core.effective_slope(COMM_TAU, rho=1.0, d=COMM_D), max(COMM_TAU))
    assert_close(core.effective_slope(COMM_TAU, rho=0.0, d=COMM_D),
                 max(core.blocking_sigma(COMM_TAU, COMM_D)))


def test_proposition1_throughput():
    assert_close(core.asymptotic_throughput(COMM_TAU, COMM_D, blocking=False), 1.0 / 5.0)
    assert_close(core.asymptotic_throughput(COMM_TAU, COMM_D, blocking=True),
                 1.0 / max(core.blocking_sigma(COMM_TAU, COMM_D)))


def test_pipeline_object_and_analyze():
    p = Pipeline(tau=[1.13, 2.90, 2.62])
    assert p.S == 3
    assert_close(p.bias(), 2.05)
    res = p.analyze(M=100)
    assert res["M"] == 100
    assert_close(res["T_free"], (100 - 1) * 2.90 + sum([1.13, 2.90, 2.62]))
    assert_close(res["homogeneity_bias"], 2.05)


def test_validation_errors():
    def raises(fn, *args, **kw):
        try:
            fn(*args, **kw)
        except ValueError:
            return
        raise AssertionError(f"expected ValueError from {fn.__name__}")

    raises(core.validate_pipeline, [])
    raises(core.validate_pipeline, [1, 2, 3], d=[0.1])  # wrong length (need 2)
    raises(core.makespan_free, [1, 2], M=0)


if __name__ == "__main__":
    # lightweight runner when pytest is unavailable
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
