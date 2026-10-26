"""Finished-road timing; one isolated worker per family, with downloadable JSON.

Example (CPU uses a deliberately smaller batch than CUDA)::

    python -m benchmarks.benchmark_generators --device cpu --envs 64 \\
        --output-dir /tmp/trackgen-cpu --label 'Mac CPU'
    python -m benchmarks.benchmark_generators --device cuda:0 --envs 4096 \\
        --output-dir /tmp/trackgen-gpu --label 'NVIDIA GB10'

The supervisor imports only the standard library. Workers alone import Warp and
create device contexts. Output contains no hostnames, user names or absolute paths;
worker logs and progress files are local diagnostics, not publishable artifacts.
"""
from __future__ import annotations

import argparse
import dataclasses
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import signal
import statistics
import subprocess
import sys
import time

GENERATORS = ("bezier", "checkpoint", "hull", "polar", "voronoi", "repulsive")
_ROAD = dict(num_points=256, N_max=512, half_width=.03, spacing=.018,
             scale=1., size=1., relax_enable=True, relax_solver="xpbd", relax_iters=50,
             z_profile="flat", on_invalid="poison")
_PROTOCOL = dict(version=1, pipeline="public finished road", road=_ROAD,
                 seeds="explicit int32 arange(E)", advance=True, mask=None,
                 min_warmup_calls=3, min_warmup_seconds=1., timed_epochs="1..repeats",
                 cpu_timer="completed-call perf_counter_ns",
                 cuda_timer="Warp events around ScopedCapture graph replay",
                 wall_timer="completed-call perf_counter_ns",
                 excluded=["initialization", "constructor", "capture", "first_replay", "warmup"],
                 yield_denominator="all attempted candidates, no refill")


class BenchmarkGuardError(RuntimeError):
    """The measured operation could not be demonstrated to regenerate a full batch."""


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _protocol(envs, repeats, warmup_seconds, device):
    return dict(_PROTOCOL, envs=envs, repeats=repeats,
                warmup_seconds=max(1., warmup_seconds),
                device_mode="cpu_eager" if device == "cpu" else "cuda_graph")


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _phase(path, name, **details):
    if path is not None:
        _write_json(path, {"phase": name, **details})


def _read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


def _revision(value=None):
    if value is None:
        try:
            value = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                            cwd=Path(__file__).resolve().parents[1],
                                            stderr=subprocess.DEVNULL, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            raise ValueError("archive runs require --source-revision") from None
    if not re.fullmatch(r"[0-9a-fA-F]{40,64}", value):
        raise ValueError("source revision must be a full hexadecimal commit identifier")
    return value.lower()


def _provenance(source_revision, protocol):
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for source in sorted((root / "track_gen").rglob("*.py")):
        digest.update(source.relative_to(root).as_posix().encode() + b"\0")
        digest.update(source.read_bytes())
        digest.update(b"\0")
    return dict(source_revision=_revision(source_revision),
                runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                library_sha256=digest.hexdigest(),
                protocol_sha256=hashlib.sha256(_json_bytes(protocol)).hexdigest())


def summarize(samples, *, mode):
    """Median/range and candidate accounting; invalid rows stay in the denominator."""
    if not samples:
        raise ValueError("at least one timing sample is required")
    field = "device_ms" if mode == "cuda_graph" else "wall_ms"
    values = [sample[field] for sample in samples]
    if any(v is None or not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError("timing samples must be finite positive milliseconds")
    batch = samples[0]["attempted_count"]
    if batch < 1 or any(s["attempted_count"] != batch or
                        not 0 <= s["valid_count"] <= batch for s in samples):
        raise ValueError("samples must describe the same complete candidate batch")
    median = statistics.median(values)
    attempted = sum(s["attempted_count"] for s in samples)
    valid = sum(s["valid_count"] for s in samples)
    throughput = batch * 1000. / median
    return dict(metric=field, median_ms=median, min_ms=min(values), max_ms=max(values),
                sample_count=len(samples), batch_size=batch,
                valid_count=valid, attempted_count=attempted,
                attempted_tracks_per_second=throughput,
                valid_tracks_per_second=throughput * valid / attempted,
                microseconds_per_attempted_track=median * 1000. / batch) | {"yield": valid / attempted}


def _cpu_name():
    if platform.system() == "Darwin":
        try:
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"],
                                            stderr=subprocess.DEVNULL, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    if platform.system() == "Linux":
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.partition(":")[2].strip()
        except OSError:
            pass
    return platform.processor() or platform.machine()


def _ram_bytes():
    if platform.system() == "Darwin":
        try:
            return int(subprocess.check_output(["sysctl", "-n", "hw.memsize"],
                                               stderr=subprocess.DEVNULL, text=True))
        except (OSError, ValueError, subprocess.CalledProcessError):
            return None
    try:
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError):
        return None


def _metadata(wp, np, device):
    hardware = dict(cpu=_cpu_name(), architecture=platform.machine(),
                    logical_cpus=os.cpu_count(), ram_bytes=_ram_bytes(), gpu=None, gpu_memory_bytes=None,
                    compute_capability=None)
    if device.is_cuda:
        hardware.update(gpu=device.name, gpu_memory_bytes=int(device.total_memory),
                        compute_capability=int(device.arch))
    software = dict(os=platform.system(), os_release=platform.release(),
                    python=platform.python_version(), numpy=np.__version__, warp=wp.__version__,
                    cuda_driver=list(wp.get_cuda_driver_version() or ()) if device.is_cuda else None,
                    cuda_toolkit=list(wp.get_cuda_toolkit_version() or ()) if device.is_cuda else None,
                    kernel_cache="as found; process construction excluded")
    return hardware, software


def _check_epoch(gen, track, expected, np):
    if not (np.all(gen.seeds.epoch.numpy() == expected)
            and np.all(track.generation.numpy() == expected)):
        raise BenchmarkGuardError("epoch or generation did not advance once for every candidate")


def _geometry(track, envs, capacity, np):
    valid = track.valid.numpy().astype(bool)
    count = track.count.numpy()
    center = track.center.numpy().reshape(envs, capacity, 3)
    geometry = {}
    for e in np.flatnonzero(valid):
        row = center[e, :int(count[e])]
        if len(row) < 3 or not np.isfinite(row).all():
            raise BenchmarkGuardError("valid candidate has nonfinite or incomplete geometry")
        geometry[int(e)] = hashlib.sha256(row.tobytes()).hexdigest()
    return geometry


def _sample(track, epoch, envs, wall_ms, device_ms, np):
    valid = track.valid.numpy().astype(bool)
    count = track.count.numpy()[valid]
    fail = track.fail_mask.numpy()
    if not np.array_equal(valid, fail == 0):
        raise BenchmarkGuardError("validity and failure flags disagree")
    n_valid = int(valid.sum())
    return dict(epoch=epoch, wall_ms=wall_ms, device_ms=device_ms,
                valid_count=n_valid, attempted_count=envs,
                fail_counts={name: int(((fail & (1 << bit)) != 0).sum())
                             for name, bit in track.fail_bits.items()},
                truncated_count=int((track.truncated.numpy() != 0).sum()),
                count_valid=dict(min=int(count.min()), max=int(count.max()),
                                 mean=float(count.mean())) if count.size else None) | {"yield": n_valid / envs}


def run_case(generator, *, envs=4096, device="cpu", repeats=5, warmup_seconds=1.,
             source_revision=None, label="Benchmark", progress_path=None):
    """Run one worker case. Raises on unavailable CUDA or failed recomputation guards.

    Successful JSON includes ``samples`` (wall_ms/device_ms, epoch, valid_count,
    attempted_count, yield, fail_counts), ``summary`` (metric, median/min/max_ms,
    microseconds_per_attempted_track, throughput and aggregate yield), ``config``,
    ``seeds``, ``epochs``, ``phases_seconds``, ``checks``, ``provenance``, ``hardware``
    and ``software``. No percentile is inferred from the default five samples.
    """
    if (generator not in GENERATORS or envs < 1 or repeats < 1
            or not math.isfinite(warmup_seconds) or warmup_seconds < 0):
        raise ValueError("known generator, positive envs/repeats and nonnegative warmup required")
    if device != "cpu" and not re.fullmatch(r"cuda(?::[0-9]+)?", device):
        raise ValueError("device must be cpu or an explicit CUDA device")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 +()_-]{0,79}", label):
        raise ValueError("label must be a public hardware label without addresses or paths")
    protocol = _protocol(envs, repeats, warmup_seconds, device)
    provenance = _provenance(source_revision, protocol)
    started_at = _utc_now()
    phases = {}

    def phase(name, **details):
        _phase(progress_path, name, phases_seconds=phases, **details)

    phase("initialization")
    started = time.perf_counter_ns()
    import numpy as np
    import warp as wp
    from track_gen import TrackGenConfig, TrackGenerator, __version__
    wp.init()
    dev = wp.get_device(device)  # explicit requested device; never fall back to CPU
    hardware, software = _metadata(wp, np, dev)
    software["trackgen"] = __version__
    phases["initialization"] = (time.perf_counter_ns() - started) * 1e-9
    seeds = np.arange(envs, dtype=np.int32)
    seed_array = wp.array(seeds, dtype=wp.int32, device=dev)
    cfg = TrackGenConfig(generator=generator, num_envs=envs, device=str(dev), **_ROAD)
    phase("constructor")
    started = time.perf_counter_ns()
    gen = TrackGenerator(cfg, seeds=seed_array)
    phases["constructor"] = (time.perf_counter_ns() - started) * 1e-9
    track = gen.program.result
    stream = wp.get_stream(dev) if dev.is_cuda else None
    checks = dict(capture_did_not_execute=None, epoch_and_generation=False,
                  poison_overwritten=False, changed_valid_rows=0, warmup_calls=0,
                  warmup_replay_seconds=0.)
    _check_epoch(gen, track, 0, np)
    if dev.is_cuda:
        phase("capture")
        before = {name: getattr(track, name).numpy().tobytes()
                  for name in ("center", "valid", "count", "generation")}
        started = time.perf_counter_ns()
        with wp.ScopedCapture(stream=stream) as cap:
            gen.generate(stream=stream, advance=True, sync=False)
        phases["capture"] = (time.perf_counter_ns() - started) * 1e-9
        _check_epoch(gen, track, 0, np)
        if any(getattr(track, name).numpy().tobytes() != value
               for name, value in before.items()):
            raise BenchmarkGuardError("recording executed or modified result storage")
        checks["capture_did_not_execute"] = True

        def replay():
            wp.capture_launch(cap.graph, stream=stream)
            stream.synchronize()
    else:
        phases["capture"] = None

        def replay():
            gen.generate(advance=True)

    previous = {}
    changed = set()
    calls = 0
    warm_replay_seconds = 0.
    warm_started = time.perf_counter_ns()
    while calls < 3 or warm_replay_seconds < max(1., warmup_seconds):
        phase("first_replay" if calls == 0 else "warmup", completed_calls=calls)
        # In-place output poisoning proves the graph actually rewrites results.
        track.center.fill_(wp.vec3f(123456.))
        track.valid.fill_(-123456)
        track.count.fill_(-123456)
        track.generation.fill_(-123456)
        if stream is not None:
            stream.synchronize()
        started = time.perf_counter_ns()
        replay()
        elapsed = (time.perf_counter_ns() - started) * 1e-9
        calls += 1
        warm_replay_seconds += elapsed
        if calls == 1:
            phases["first_replay"] = elapsed
        _check_epoch(gen, track, calls, np)
        if (np.any(track.center.numpy() == 123456.)
                or np.any(track.valid.numpy() == -123456)
                or np.any(track.count.numpy() == -123456)):
            raise BenchmarkGuardError("replay left poisoned output storage untouched")
        current = _geometry(track, envs, cfg.N_max, np)
        changed.update(e for e in previous.keys() & current.keys() if previous[e] != current[e])
        previous.update(current)
    phases["warmup"] = (time.perf_counter_ns() - warm_started) * 1e-9 - phases["first_replay"]
    if not changed:
        raise BenchmarkGuardError("no changing finite valid geometry observed across warm epochs")
    checks.update(epoch_and_generation=True, poison_overwritten=True,
                  changed_valid_rows=len(changed), warmup_calls=calls,
                  warmup_replay_seconds=warm_replay_seconds)
    phase("epoch_reset")
    gen.seeds.reset_epoch(stream=stream)
    if stream is not None:
        stream.synchronize()
    if not np.all(gen.seeds.epoch.numpy() == 0):
        raise BenchmarkGuardError("epoch reset failed")
    # All event allocations happen before any measured interval.
    events = [(wp.Event(dev, enable_timing=True), wp.Event(dev, enable_timing=True))
              for _ in range(repeats)] if dev.is_cuda else []
    samples = []
    for epoch in range(1, repeats + 1):
        phase("timing", epoch=epoch, total=repeats)
        started = time.perf_counter_ns()
        if dev.is_cuda:
            a, b = events[epoch - 1]
            stream.record_event(a)
            wp.capture_launch(cap.graph, stream=stream)
            stream.record_event(b)
            stream.synchronize()
        else:
            replay()
        wall_ms = (time.perf_counter_ns() - started) * 1e-6
        device_ms = float(wp.get_event_elapsed_time(a, b)) if dev.is_cuda else None
        # No readback, hashing or host-side quality work is inside the interval.
        phase("diagnostics", epoch=epoch, total=repeats)
        _check_epoch(gen, track, epoch, np)
        samples.append(_sample(track, epoch, envs, wall_ms, device_ms, np))
    mode = "cuda_graph" if dev.is_cuda else "cpu_eager"
    result = dict(schema_version=1, status="ok", generator=generator, label=label,
                  device=str(dev), mode=mode, config=dataclasses.asdict(gen.config),
                  started_at_utc=started_at, completed_at_utc=_utc_now(),
                  seeds=dict(start=0, count=envs, dtype="int32",
                             sha256=hashlib.sha256(seeds.astype("<i4").tobytes()).hexdigest()),
                  epochs=list(range(1, repeats + 1)), protocol=protocol,
                  provenance=provenance, hardware=hardware, software=software,
                  phases_seconds=phases, checks=checks, samples=samples,
                  summary=summarize(samples, mode=mode))
    phase("complete")
    return result


def _stop_group(process):
    """Stop the entire isolated worker group, including surviving descendants."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=1.)
    except subprocess.TimeoutExpired:
        pass
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def run_worker(command, *, timeout_seconds, progress_path, result_path, log_path):
    """Bound one subprocess; timeout rows retain phase and never claim a latency."""
    if not math.isfinite(timeout_seconds) or timeout_seconds < 0:
        raise ValueError("timeout must be nonnegative (zero is unlimited)")
    for path in (progress_path, result_path):
        Path(path).unlink(missing_ok=True)
    started = time.monotonic()
    last_phase = None
    with Path(log_path).open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, start_new_session=True)
        try:
            while True:
                phase = _read_json(progress_path).get("phase", "startup")
                if phase != last_phase:
                    print(f"phase: {phase}", flush=True)
                    last_phase = phase
                elapsed = time.monotonic() - started
                if timeout_seconds and elapsed >= timeout_seconds:
                    _stop_group(process)
                    Path(result_path).unlink(missing_ok=True)
                    return dict(status="timeout",
                                phase=_read_json(progress_path).get("phase", phase),
                                phases_seconds=_read_json(progress_path).get("phases_seconds", {}),
                                elapsed_seconds=elapsed, timeout_seconds=timeout_seconds)
                try:
                    code = process.wait(timeout=min(.25, max(.001, timeout_seconds - elapsed))
                                        if timeout_seconds else .25)
                    break
                except subprocess.TimeoutExpired:
                    pass
        except BaseException:
            _stop_group(process)
            raise
    result = _read_json(result_path)
    if code == 0 and result.get("status") == "ok":
        return result
    if result.get("status") == "error":
        return result
    return dict(status="error", phase=_read_json(progress_path).get("phase", "startup"),
                error_type="WorkerExit", returncode=code,
                elapsed_seconds=time.monotonic() - started)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True)
    parser.add_argument("--envs", type=int, default=4096)
    parser.add_argument("--generators", nargs="+", choices=GENERATORS, default=list(GENERATORS))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup-seconds", type=float, default=1.)
    parser.add_argument("--timeout-seconds", type=float, default=0., help="per family; 0 is unlimited")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--source-revision")
    parser.add_argument("--worker", choices=GENERATORS, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        revision = _revision(args.source_revision)
    except ValueError as exc:
        parser.error(str(exc))
    if (args.envs < 1 or args.repeats < 1
            or not math.isfinite(args.warmup_seconds) or args.warmup_seconds < 0
            or not math.isfinite(args.timeout_seconds) or args.timeout_seconds < 0):
        parser.error("envs/repeats must be positive; warmup/timeout must be nonnegative")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.worker:
        progress = args.output_dir / f"{args.worker}.progress.json"
        destination = args.output_dir / f"{args.worker}.worker.json"
        try:
            result = run_case(args.worker, envs=args.envs, device=args.device,
                              repeats=args.repeats, warmup_seconds=args.warmup_seconds,
                              source_revision=revision, label=args.label, progress_path=progress)
        except Exception as exc:
            result = dict(status="error", generator=args.worker,
                          phase=_read_json(progress).get("phase", "startup"),
                          error_type=type(exc).__name__)
            _write_json(destination, result)
            raise
        _write_json(destination, result)
        return 0
    records = []
    for name in args.generators:
        protocol = _protocol(args.envs, args.repeats, args.warmup_seconds, args.device)
        envelope = dict(request=dict(device=args.device, envs=args.envs, repeats=args.repeats,
                                     road=_ROAD, timeout_seconds=args.timeout_seconds),
                        protocol=protocol, provenance=_provenance(revision, protocol),
                        started_at_utc=_utc_now())
        print(f"{name}: {args.envs} candidates on {args.device}", flush=True)
        command = [sys.executable, "-m", "benchmarks.benchmark_generators", "--worker", name,
                   "--device", args.device, "--envs", str(args.envs), "--repeats", str(args.repeats),
                   "--warmup-seconds", str(args.warmup_seconds), "--output-dir", str(args.output_dir),
                   "--label", args.label, "--source-revision", revision]
        record = run_worker(command, timeout_seconds=args.timeout_seconds,
                            progress_path=args.output_dir / f"{name}.progress.json",
                            result_path=args.output_dir / f"{name}.worker.json",
                            log_path=args.output_dir / f"{name}.log")
        if record["status"] != "ok":
            record.update(envelope, completed_at_utc=_utc_now())
        record.update(generator=name, label=args.label)
        _write_json(args.output_dir / f"{name}.json", record)
        records.append(record)
        _write_json(args.output_dir / "manifest.json", dict(schema_version=1, records=records))
        print(f"{name}: {record['status']}", flush=True)
    return int(any(record["status"] != "ok" for record in records))


if __name__ == "__main__":
    raise SystemExit(main())
