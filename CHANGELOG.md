# Changelog

## 0.2.0.dev1 (unreleased, `v2` branch)

### Added

- **`track_gen.exec.LaunchContext`** (roadmap PR2): one device and one stream per call.
  Every launch, copy and fill the library issues now goes through it with an explicit
  `stream=`, so a call can be recorded into the graph the caller is capturing.
  - The stream is resolved per call. In priority order: an explicit `stream=` (a
    `wp.Stream`, or `"torch"`); an enclosing `LaunchContext.scope()`; the producer's
    constructor `stream=` (`TrackGenerator` / `GateGenerator`, roadmap PR3); torch's
    current stream when torch is capturing on the device (inside `torch.cuda.graph(g)`
    library calls are recorded into `g`); Warp's current stream.
  - Per-call methods take `stream=None`: `query` / `update` / `reset` / `sample` /
    `bake` / `refresh`, `Course.step` / `reset`, and the `PerEnvSeededRNG` samplers.
    Pipeline and generator helpers take `ctx=None`.
- `track_gen.ops.fill(ctx, arr, value)`: typed fills on the context's stream. It replaces
  `array.zero_()` on hot paths; that memset targets Warp's current stream.
- `GeneratorSpec.accepts_ctx` / `GateGeneratorSpec.accepts_ctx` (default `False`). Built-in
  generators set it and receive `ctx=`. Legacy plugins run inside `ctx.scope()`, so the
  `warp_pipeline` helpers they call inherit the stream. Their own direct `wp.launch` calls
  still use Warp's current stream.
- **Generation can be recorded into the caller's CUDA graph** (roadmap PR3).
  `TrackGenerator.generate()`, `GateGenerator.generate()`, `Course.generate()` and the new
  `Course.regenerate()` called inside `wp.ScopedCapture`, `wp.capture_begin(external=True)`
  or `torch.cuda.graph` are recorded into that graph and replay correctly.
  - Recording enqueues the seed copy and the whole pipeline on the capturing stream. There
    is no self-capture, no `wp.capture_launch`, no host sync and no allocation, so it also
    works with Warp's mempool disabled.
  - Every replay reads the rng's current seeds: `rng.set_seeds_warp(new, None)` between
    replays gives a new batch without re-capturing.
  - Outside a capture, the facades still replay their own graph on CUDA and run eagerly on
    CPU.
  - Generator outputs (`Track` / `GateSequence`) are byte-identical to the previous
    release for the same seeds, on CPU and CUDA. `Course` sizes its sub-tools differently
    (see Behaviour changes), which can change the shape of its checkpoint arrays.
- New facade API on `TrackGenerator` and `GateGenerator`:
  - `generate(num_or_ids=None, *, stream=None, sync=True)`. `sync=False` returns without
    blocking.
  - `enqueue(*, stream=None)`: recorded under a capture, otherwise eager on the resolved
    stream. It never builds a self-graph and never blocks.
  - `graph`: the generator's own `wp.Graph`, `None` until the first CUDA `generate()`
    outside a capture.
  - `config`: a copy of the construction config (a `wp.Device` value is shared, not
    copied).
  - A `stream=` constructor argument, the default stream for calls that pass none. It
    outranks torch adoption: inside `torch.cuda.graph`, call such a generator with
    `stream="torch"` (or construct it without `stream=`).
- New `Course` API:
  - `regenerate(*, stream=None, sync=True)`: generation plus the downstream refresh as
    one body.
  - `enqueue_generate(*, stream=None)`: the same body, recorded or eager, never
    self-graphed.
  - `graph`: the course's own regeneration graph.
  - `generate(seeds=None, *, stream=None, sync=True)`.
- `track_gen.exec.CaptureError`, `NotCapturableError` and `LegacyPluginStreamError`, and
  `track_gen.exec.GraphRunner`, the record / eager / self-graph policy the facades share.

### Behaviour changes

- **Library calls made inside `torch.cuda.graph(g)` are now recorded into `g`.** They used
  to run eagerly on Warp's stream and were missing from the graph. Calls on a side
  stream are recorded too: pass `stream=s` or use `LaunchContext(device, s).scope()`.
- **The generator warm-up moved to construction.** `TrackGenerator` and `GateGenerator`
  construction now runs the pipeline once eagerly. That pass loads every kernel the hot
  path uses, and it leaves the batch for the rng's current seeds in the result. The first
  CUDA `generate()` then only records the self-graph and launches it.
  - The old first call is gone: three eager warm-ups, a `wp.synchronize()`, then the
    capture. A generator body now runs twice before the first replay instead of four
    times.
  - **Cost:** construction pays one full generation, a full iteration budget for
    `repulsive`, and blocks until it has finished on its stream, so a later call on
    another stream (or a readback) never races the warm-up.
  - Construction raises `CaptureError`, before allocating anything, when its resolved
    stream or Warp's current stream is capturing, or torch is capturing in the calling
    thread (construction allocates and zero-fills on Warp's current stream).
- **`generate()` no longer calls `wp.synchronize()`.** With the default `sync=True` it
  waits only for the stream it ran on, and only outside a capture. `sync=False` returns at
  once, with stream-ordered results. Under a capture it never waits, nor while torch is
  capturing in the calling thread. On Warp >= 1.18 the wait runs with the thread's CUDA
  capture mode relaxed, so it cannot invalidate a capture open in another thread.
- **The process-wide `_CAPTURE_LOCK` is gone from the built-in path.** Each generator and
  `Course` captures its self-graph once, on its own private stream, and launches it on
  the call's stream. Independent generators on different threads no longer serialize.
  - Each instance is single-writer: do not run the same generator or `Course` from two
    threads at once.
  - If the one-time capture is invalidated (e.g. by another thread using the legacy
    default stream), it is retried once before raising.
  - Plugins without `accepts_ctx=True` keep the lock and capture on Warp's current
    stream, which all threads share. Until every such producer has captured its
    self-graph, calls of other producers that resolve to Warp's current stream (the
    default) take the lock too and decide their mode under it, so they cannot mistake
    the legacy capture for their caller's. Calls on any other stream stay lock-free.
  - Outside a capture, a legacy-plugin producer whose call resolved another stream
    (`stream=`, constructor `stream=`, a scope) runs its whole body on Warp's current
    stream, ordered after and before the resolved stream with events: the plugin's own
    launches stay ordered with the library's stages.
- **`Course` builds everything at construction.** `course.result` holds the first batch
  and every sub-tool exists right after construction, so `step()` and `reset()` only need
  `bind()`.
  - Sub-tools are sized from the construction batch (`config.seeds`), no longer from the
    first `generate(seeds=k)` batch. That includes an auto-derived `max_checkpoints`, so
    `Course(cfg); course.generate(seeds=k)` can give checkpoint arrays of another shape
    than before (and a ring can now be truncated, see `checkpoint_sampler.truncated`).
    Pass `max_checkpoints` explicitly for seed-independent sizing.
  - Construction then runs the regeneration body once more as its warm-up, so it costs
    about two generations.
  - Generation and refresh are now ONE graph (`course.graph`); the separate refresh graph
    and the private `_refresh_graph` attribute are gone.
  - `generate(seeds=int)` no longer allocates: the seeds are expanded on device into a
    persistent buffer. **Under a caller's capture the int is baked into the graph**, so
    every replay reseeds with that value. To vary seeds between replays, pass a `wp.array`
    and rewrite it in place.
- **Generators snapshot their config.** Mutating the config object after construction no
  longer affects the generator, and `config` returns a copy (a `wp.Device` value is shared,
  so `device=wp.get_device(...)` keeps working). The private `_graph` is now a read-only
  alias of `graph`.
- **torch stream adoption no longer creates a throwaway `wp.Stream` per call.** Inside
  `torch.cuda.graph` (and for `stream="torch"`) the library reuses Warp's current stream
  when it is torch's stream, else the stream an open Warp capture was started on, else one
  wrapper per torch stream kept for the process. A per-call wrapper unregistered the handle
  from Warp for the whole process when it was collected (Warp does not reference-count
  it): an open Warp capture on that handle then failed at `capture_end`, and Warp's
  current stream became NULL.
- Under a caller's capture, a `capturable=False` generator raises `NotCapturableError`
  before launching anything. A plugin without `accepts_ctx=True` raises
  `LegacyPluginStreamError` unless the capture stream is Warp's current stream (recipe in
  the error message and in `docs/how-it-works/cuda-graph.rst`).

### Testing

- `tests/test_launch_stream.py` (CPU) fails if any library `wp.launch` / `wp.copy` call
  omits `stream=`, or any library `zero_()` / `fill_()` runs, across every
  `test_no_host_sync` driver, construction included.
- `tests/test_stream_capture_cuda.py` (CUDA) records every track and gate pipeline on a
  non-current side stream, and `Course.step()` both on an explicit side stream and inside
  `torch.cuda.graph`. It compares poisoned replays against eager twins.
- `tests/test_caller_capture_cuda.py` (CUDA) records `generate(stream=s)` of every track
  and gate generator on a non-current side stream, with Warp's mempool disabled and
  `wp.capture_launch` forbidden while recording. Poisoned replays must equal eager twins,
  before and after an in-place reseed. It also covers `torch.cuda.graph` adoption, a
  `Course` graph holding `regenerate()` + `step()` + `reset(mask)`, `NotCapturableError`
  and the self-graph path.
- CPU tests check that construction runs the generator body exactly once, the config
  snapshot (including `device=wp.get_device(...)`), and that `Course.regenerate()` /
  `generate(seeds=...)` match a fresh `Course`. On a fake CUDA device they also check the
  legacy-plugin lock and stream rules, the construction-time capture guard, the sync
  rules, and that torch adoption never creates a throwaway wrapper.
  `tests/test_generate_concurrent_cuda.py` now passes without the process-wide lock.
- `tests/test_caller_capture_cuda.py` also runs a legacy plugin that issues its own
  stream-less `wp.launch` (eager, constructor `stream=`, self-graph, recording), the
  documented `ScopedStream(stream_from_torch(s))` + `torch.cuda.graph(g, stream=s)` recipe
  followed by a second Warp capture on the same wrapper, and a Warp capture on a
  torch-current stream.

## 0.2.0.dev0 (2026-10-21, tag `v0.2.0.dev0`)

Development line for the library-readiness roadmap (`docs/library-readiness/`). Racing
projects that need stability should pin the `v0.1.0` tag.

### Behaviour changes

- **Tracks truncated at `N_max` now fail validity.** `_cs_scan_k` records a per-env flag
  from the unclamped point count, the validity gate fails flagged envs, and the flag is
  exposed as `Track.truncated` ([E] int32, covered by `clone()`). Before, a truncated track
  could ship with `valid=1`, and on the CUDA graph path nothing reported the truncation. The
  host-side `RuntimeWarning` now fires only in `resample_constant_spacing`'s allocating
  mode; in-place calls (the pipeline path) never read back. Size
  `N_max >= max(perimeter) / spacing + 1`. **Action needed:** at `half_width=0.03` the
  default `N_max=384` covers only ~98% of tracks, so about 2% of envs that used to ship
  clipped but valid now fail; set `N_max` to at least 512 (see
  `docs/configuration/tuning.rst`).
- **The `repulsive` generator is CUDA-graph capturable** (`capturable=True`). The per-window
  host readback and global early exit are gone, and freeze is absorbing, so each env depends
  only on its own seed, not on the rest of the batch. **Output change:** envs that froze
  before the loop ended keep their at-freeze shape and no longer get later periodic
  resamples. Envs that never freeze are byte-identical. **Cost:** every call now runs the
  full fixed iteration budget (there is no all-frozen early exit), and on CUDA the first
  `generate()` also pays the generator's 3 eager warm-ups plus the capture of a ~4k-node
  graph; later calls replay that graph.
- `apply_z_profile` raises `ValueError` on an unknown `z_profile`. It used to fall through
  silently to `"noise"`.
- `TrackGenerator` raises `ValueError` instead of `AssertionError` for an unsupported
  `relax_solver` or `smooth_finish=True`. It also checks that the rng has exactly `num_envs`
  seeds, as `GateGenerator` does.
- `GateGeneratorSpec` gains `capturable` (default `True`). On CUDA, `GateGenerator` runs a
  non-capturable gate generator eagerly instead of capturing it.
- The pipeline scratch no longer falls through to generator scratch by attribute name, so a
  generator scratch field named `w`, `kappa`, `z`, `area_a` or `area_b` can no longer shadow
  the pipeline's own buffers.
- **Utilities, pipeline wrappers and generator bodies no longer block the host.**
  On CUDA they used to call a global `wp.synchronize()` after their launches. They now only
  enqueue work on the device's current Warp stream. This covers `query()` / `update()` /
  `reset()` / `sample()` / `bake()` / `refresh()` on every utility, `Course.step()` /
  `reset()`, every `warp_pipeline` / `warp_gate` / `warp_relax` wrapper, and every
  generator's pipeline body.
  - Unchanged: consumers on the same stream, `wp.array.numpy()` (a blocking copy), and torch
    on its legacy default stream still see finished results, because Warp streams are created
    blocking.
  - **Action needed:** a consumer on a different stream (for example a torch side stream or
    a simulator's own stream) must order itself explicitly, e.g. with `wp.synchronize_device()`
    or a recorded `wp.Event`. Benchmarks must synchronize before reading a timer.
  - All of these calls can now be recorded into a caller's CUDA graph without any flag.
  - Debugging aid: `TRACKGEN_DEBUG_SYNC=1` makes every library call block on its stream
    again, so asynchronous CUDA errors surface at the call that caused them. It is skipped
    while the stream (or a Warp capture on the device) is capturing; do not combine it with a
    foreign capture on another stream, which it cannot detect.
  - Not changed yet: `TrackGenerator.generate()`, `GateGenerator.generate()` and
    `Course.generate()` keep their self-capture and blocking sync on CUDA.
- **`track_gen.set_capturing()` is deprecated** and is now a no-op that emits a
  `DeprecationWarning`. It will be removed in 1.0. The private `runtime._CAPTURING`,
  `warp_pipeline._CAPTURING` and `warp_gate._CAPTURING` names stay defined but are always
  `False` and are no longer read. `warp_relax.xpbd_solve_inplace` / `band_l0_inplace` accept
  and ignore `capturing=`. `warp_pipeline._sync` / `_init` and `warp_gate._sync` / `_init`
  remain importable for generator plugins.

### Packaging

- `warp-lang` is pinned to `>=1.14,<2`.
- `track_gen._experimental` is no longer shipped in the wheel.

### Testing

- New `.github/workflows/tests.yml`. The CPU lane runs on every PR
  (`pytest -m "not cuda and not slow and not benchmark"`, warp-lang 1.14.0 and latest).
  The GPU lane (`pytest -m cuda` with `TRACKGEN_REQUIRE_CUDA=1`) needs a self-hosted runner
  and is off until the `TRACKGEN_GPU_RUNNER` repository variable is set to `true`.
- CUDA tests are gated by the `cuda` marker on `wp.is_cuda_available()` (`tests/conftest.py`),
  replacing the `torch.cuda.is_available()` checks that silently skipped them on CPU-only
  torch builds. With `TRACKGEN_REQUIRE_CUDA=1`, a skipped `cuda` test fails the session.
- `tests/test_no_host_sync.py` drives every hot path on CPU with `wp.synchronize*` and
  `wp.array.numpy` replaced by tripwires, and checks that every pipeline wrapper is driven.
  Every generator hot path, repulsive included, is readback-free, so the known-readback
  allowlist is empty.
- `tests/test_shape_variety.py` read the vec3f centerlines as vec2f, which defeated the
  registry-wide degeneracy gate; fixed, with a self-test that a circle scores ~1.0.
- New `tests/test_truncation.py` (fail-closed truncation on CPU and under CUDA capture) and
  repulsive capture/replay and batch-independence tests.
