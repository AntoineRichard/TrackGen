"""Benchmark accounting, real regeneration, and bounded-worker failure behavior."""
from __future__ import annotations

import importlib
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest


def _runner():
    try:
        return importlib.import_module("benchmarks.benchmark_generators")
    except ModuleNotFoundError:
        pytest.fail("the dedicated generator benchmark runner is missing")


def test_raw_summary_reports_all_attempts_and_time_units_without_road_yield():
    bench = _runner()
    samples = [dict(epoch=i, wall_ms=w, device_ms=d, attempted_count=4)
               for i, w, d in [(1, 8., 1.), (2, 12., 2.), (3, 16., 3.)]]
    cpu = bench.summarize(samples, mode="cpu_eager")
    gpu = bench.summarize(samples, mode="cuda_graph")
    assert cpu["median_ms"] == 12.
    assert gpu["median_ms"] == 2.
    assert cpu["min_ms"] == 8. and cpu["max_ms"] == 16.
    assert cpu["attempted_count"] == 12
    assert "yield" not in cpu and "valid_count" not in cpu
    assert cpu["attempted_tracks_per_second"] == pytest.approx(1000 / 3)
    assert gpu["attempted_tracks_per_second"] == pytest.approx(2000.)
    assert "valid_tracks_per_second" not in gpu
    assert "p95_ms" not in cpu


def test_supervisor_import_does_not_load_warp_or_numpy():
    _runner()
    code = ("import sys; import benchmarks.benchmark_generators; "
            "assert 'warp' not in sys.modules; assert 'numpy' not in sys.modules")
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


def test_real_small_cpu_regeneration_checks_and_json_provenance():
    result = _runner().run_case("polar", envs=4, device="cpu", repeats=2,
                                warmup_seconds=0, source_revision="a" * 40,
                                label="CPU test")
    assert result["status"] == "ok"
    assert result["mode"] == "cpu_eager"
    assert datetime.fromisoformat(result["completed_at_utc"]) >= datetime.fromisoformat(result["started_at_utc"])
    ram = result["hardware"]["ram_bytes"]
    assert ram is None or ram > 0
    assert result["epochs"] == [1, 2]
    assert [s["epoch"] for s in result["samples"]] == [1, 2]
    assert [s["attempted_count"] for s in result["samples"]] == [4, 4]
    assert all(s["wall_ms"] > 0 and s["device_ms"] is None for s in result["samples"])
    assert result["checks"]["epoch_and_effective_seed"] is True
    assert result["checks"]["raw_storage_overwritten"] is True
    assert result["checks"]["changed_raw_rows"] > 0
    assert result["checks"]["warmup_calls"] >= 3
    assert result["checks"]["warmup_replay_seconds"] >= 1.
    assert result["config"]["num_points"] == 256
    assert result["config"]["scale"] == 1.
    assert result["config"]["generator_params"]["size"] == 1.
    assert "spacing" not in result["config"] and "relax_iters" not in result["config"]
    assert result["schema_version"] == result["protocol"]["version"] == 2
    assert result["protocol"]["pipeline"] == "generator_only"
    assert result["checks"]["proof_calls"] == 3
    assert "yield" not in result["summary"]
    assert all("yield" not in sample and "valid_count" not in sample
               for sample in result["samples"])
    assert result["seeds"]["count"] == 4 and result["seeds"]["dtype"] == "int32"
    assert result["summary"]["attempted_count"] == 8
    assert result["provenance"]["source_revision"] == "a" * 40
    for field in ("runner_sha256", "library_sha256", "protocol_sha256"):
        assert len(result["provenance"][field]) == 64
    encoded = json.dumps(result, allow_nan=False)
    assert os.path.expanduser("~") not in encoded
    assert "hostname" not in result["hardware"]


def test_timeout_preserves_phase_and_kills_descendants(tmp_path):
    bench = _runner()
    phase = tmp_path / "phase.json"
    worker_result = tmp_path / "worker.json"
    marker = tmp_path / "surviving-grandchild"
    grandchild = ("import pathlib,signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); "
                  f"time.sleep(1.2); pathlib.Path({str(marker)!r}).write_text('alive')")
    child = ("import json,pathlib,subprocess,sys,time; "
             f"subprocess.Popen([sys.executable,'-c',{grandchild!r}]); "
             f"pathlib.Path({str(phase)!r}).write_text(json.dumps({{'phase':'constructor'}})); "
             "time.sleep(30)")
    result = bench.run_worker([sys.executable, "-c", child], timeout_seconds=.4,
                              progress_path=phase, result_path=worker_result,
                              log_path=tmp_path / "worker.log")
    assert result["status"] == "timeout"
    assert result["phase"] == "constructor"
    assert "summary" not in result and "samples" not in result
    time.sleep(1.3)
    assert not marker.exists(), "timeout left a child process running"


@pytest.mark.cuda
def test_actual_cuda_graph_records_then_recomputes_advancing_epochs():
    import warp as wp
    if not wp.is_cuda_available():
        if os.environ.get("TRACKGEN_REQUIRE_CUDA") == "1":
            pytest.fail("required CUDA device unavailable")
        pytest.skip("CUDA device unavailable")
    result = _runner().run_case("polar", envs=4, device="cuda:0", repeats=2,
                                warmup_seconds=0, source_revision="a" * 40,
                                label="CUDA test")
    assert result["status"] == "ok" and result["mode"] == "cuda_graph"
    assert result["checks"]["capture_did_not_execute"] is True
    assert result["checks"]["epoch_and_effective_seed"] is True
    assert result["checks"]["raw_storage_overwritten"] is True
    assert result["checks"]["changed_raw_rows"] > 0
    assert result["epochs"] == [1, 2]
    assert all(s["device_ms"] > 0 and s["wall_ms"] > 0 for s in result["samples"])


def test_supervised_cli_retains_requested_batch_and_public_provenance(tmp_path):
    _runner()
    completed = subprocess.run(
        [sys.executable, "-m", "benchmarks.benchmark_generators", "--device", "cpu",
         "--envs", "4", "--generators", "polar", "--repeats", "1",
         "--warmup-seconds", "0", "--timeout-seconds", "30", "--output-dir", str(tmp_path),
         "--label", "CLI CPU", "--source-revision", "b" * 40],
        capture_output=True, text=True, timeout=40)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads((tmp_path / "polar.json").read_text())
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["records"] == [result]
    assert result["config"]["num_envs"] == 4
    assert result["summary"]["attempted_count"] == 4
    assert result["provenance"]["source_revision"] == "b" * 40
    assert str(tmp_path) not in json.dumps(result)


@pytest.mark.parametrize("option", ["--warmup-seconds", "--timeout-seconds"])
@pytest.mark.parametrize("value", ["nan", "inf"])
def test_cli_rejects_nonfinite_budgets_without_starting_workers(tmp_path, monkeypatch, option, value):
    bench = _runner()

    def unexpected_spawn(*args, **kwargs):
        pytest.fail("invalid budget reached the worker launcher")

    monkeypatch.setattr(bench, "run_worker", unexpected_spawn)
    with pytest.raises(SystemExit) as stopped:
        bench.main(["--device", "cpu", "--envs", "1", "--repeats", "1", "--generators", "polar",
                    "--output-dir", str(tmp_path), "--label", "CPU", "--source-revision", "a" * 40,
                    option, value])
    assert stopped.value.code == 2
    assert not list(tmp_path.glob("*.log"))


def test_worker_failure_retains_protocol_and_source_revision(tmp_path):
    _runner()
    completed = subprocess.run(
        [sys.executable, "-m", "benchmarks.benchmark_generators", "--device", "cuda:999999",
         "--envs", "4", "--generators", "polar", "--repeats", "1",
         "--timeout-seconds", "15", "--output-dir", str(tmp_path), "--label", "Unavailable GPU",
         "--source-revision", "c" * 40], capture_output=True, text=True, timeout=20)
    assert completed.returncode == 1
    result = json.loads((tmp_path / "polar.json").read_text())
    assert result["status"] == "error" and result["phase"] == "initialization"
    assert result["request"]["device"] == "cuda:999999"
    assert result["request"]["envs"] == 4
    assert result["provenance"]["source_revision"] == "c" * 40
    assert len(result["provenance"]["protocol_sha256"]) == 64
    assert "summary" not in result and "samples" not in result


@pytest.mark.parametrize("family", ["bezier", "checkpoint", "hull", "polar", "voronoi", "repulsive"])
def test_raw_body_matches_public_raw_tap_and_skips_every_common_stage(family, monkeypatch):
    import numpy as np
    from track_gen import TrackGenConfig, TrackGenerator
    from track_gen.exec import LaunchContext

    cfg = TrackGenConfig(generator=family, num_envs=1, device="cpu", num_points=256,
                         N_max=512, half_width=.03, spacing=.018, scale=1., size=1.)
    gen = TrackGenerator(cfg, seeds=7)
    body = _runner()._RawGeneratorBody(gen)
    program = gen.program
    before_raw = program.tap("raw.xy").numpy().copy()
    untouched = {name: getattr(program.result, name).numpy().tobytes()
                 for name in ("center", "inner", "outer", "valid", "count", "fail_mask", "generation")}
    before_cl = program.tap("cl.xy").numpy().tobytes()

    def forbidden_common_stage(ctx):
        pytest.fail("a shared road stage ran during the generator-only body")

    with monkeypatch.context() as patch:
        patch.setattr(program, "_enqueues", [entry if index == body.index else forbidden_common_stage
                      for index, entry in enumerate(program._enqueues)])
        body.enqueue(LaunchContext.current("cpu"))
    raw, accepted = body.raw.numpy().copy(), body.accepted.numpy().copy()
    assert raw.shape == (256, 2) and body.raw_points == 256
    assert not np.array_equal(raw, before_raw, equal_nan=True)
    assert gen.seeds.epoch.numpy().tolist() == [1]
    assert all(getattr(program.result, name).numpy().tobytes() == value
               for name, value in untouched.items())
    assert program.tap("cl.xy").numpy().tobytes() == before_cl

    gen.seeds.reset_epoch()
    gen.generate(advance=True)
    np.testing.assert_array_equal(raw, program.tap("raw.xy").numpy())
    np.testing.assert_array_equal(accepted, program.tap("gen.valid").numpy())
    assert program.result.generation.numpy().tolist() == [1]


def test_raw_guard_rejects_changing_geometry_with_all_generator_flags_rejected(monkeypatch):
    bench = _runner()
    original = bench._RawGeneratorBody.enqueue

    def reject_all(self, ctx):
        original(self, ctx)
        ctx.fill(self.accepted, 0)

    monkeypatch.setattr(bench._RawGeneratorBody, "enqueue", reject_all)
    with pytest.raises(bench.BenchmarkGuardError, match="changing finite accepted raw geometry"):
        bench.run_case("polar", envs=4, device="cpu", repeats=1, warmup_seconds=0,
                       source_revision="a" * 40, label="Rejected raw test")
