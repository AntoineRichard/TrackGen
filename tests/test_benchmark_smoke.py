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


def test_collision_benchmark_runs_small_cpu():
    pytest.importorskip("warp")
    from benchmarks.benchmark_collision import run_collision_benchmark
    m = run_collision_benchmark(E=4, B=2, device="cpu", sdf_resolution=16,
                                iters=2, warmup=1, seed=42)
    assert m["boxes"] == 8
    assert m["seg_eager_s"] >= 0.0 and m["sdf_eager_s"] >= 0.0
    assert m["seg_graph_s"] is None and m["sdf_graph_s"] is None  # cpu: no graph
    assert 0.0 <= m["flag_agreement"] <= 1.0


def test_pipeline_benchmark_runs_small_cpu():
    pytest.importorskip("warp")
    from benchmarks.benchmark_pipeline import run_pipeline_benchmark
    rows = run_pipeline_benchmark(E=8, N=128, half_width=0.03, device="cpu",
                                  relax_iters=30, max_regen_iters=4, reps=1)
    assert any(r["mode"] == "eager" for r in rows)
    for r in rows:
        assert 0.0 <= r["valid_frac"] <= 1.0
        assert r["seconds"] >= 0.0
