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


def test_summary_preserves_failed_candidates_and_time_units():
    bench = _runner()
    samples = [dict(epoch=i, wall_ms=w, device_ms=d, valid_count=v, attempted_count=4)
               for i, w, d, v in [(1, 8., 1., 2), (2, 12., 2., 0), (3, 16., 3., 4)]]
    cpu = bench.summarize(samples, mode="cpu_eager")
    gpu = bench.summarize(samples, mode="cuda_graph")
    assert cpu["median_ms"] == 12.
    assert gpu["median_ms"] == 2.
    assert cpu["min_ms"] == 8. and cpu["max_ms"] == 16.
    assert cpu["valid_count"] == 6 and cpu["attempted_count"] == 12
    assert cpu["yield"] == .5
    assert cpu["attempted_tracks_per_second"] == pytest.approx(1000 / 3)
    assert gpu["valid_tracks_per_second"] == pytest.approx(1000.)
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
    assert result["checks"]["epoch_and_generation"] is True
    assert result["checks"]["poison_overwritten"] is True
    assert result["checks"]["changed_valid_rows"] > 0
    assert result["checks"]["warmup_calls"] >= 3
    assert result["checks"]["warmup_replay_seconds"] >= 1.
    assert result["config"]["N_max"] == 512
    assert result["config"]["half_width"] == .03
    assert result["config"]["spacing"] == .018
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
    assert result["checks"]["epoch_and_generation"] is True
    assert result["checks"]["poison_overwritten"] is True
    assert result["checks"]["changed_valid_rows"] > 0
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
