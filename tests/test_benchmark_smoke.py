import torch
import pytest

pytestmark = [pytest.mark.benchmark, pytest.mark.slow]


def test_benchmark_runs_small_cpu():
    pytest.importorskip("warp")
    from benchmarks.benchmark_relaxation import run_benchmark
    rows = run_benchmark(E=8, N=128, half_width=0.03, device="cpu",
                         solvers=("xpbd", "energy", "tp_sobolev"),
                         energy_steps=50, tp_iters=20, relax_iters=40, seed=20)
    assert set(r["solver"] for r in rows) >= {"xpbd", "energy", "tp_sobolev"}
    for r in rows:
        assert 0.0 <= r["valid_frac"] <= 1.0
        assert r["seconds"] >= 0.0


def test_pipeline_benchmark_runs_small_cpu():
    pytest.importorskip("warp")
    from benchmarks.benchmark_pipeline import run_pipeline_benchmark
    rows = run_pipeline_benchmark(E=8, N=128, half_width=0.03, device="cpu",
                                  relax_iters=30, max_regen_iters=4, reps=1)
    assert any(r["mode"] == "eager" for r in rows)
    for r in rows:
        assert 0.0 <= r["valid_frac"] <= 1.0
        assert r["seconds"] >= 0.0
