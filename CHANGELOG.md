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
- **Device-side reseed, masked regeneration and in-graph auto-reset** (roadmap PR4). One
  caller graph can now hold `[physics] -> course.step() -> done mask ->
  course.regenerate(mask=done, advance=True)`. Every replay gives new courses to the
  masked envs and resets their per-env state, with no host involvement.
  - **`track_gen.seeds.SeedState`**: per-env `seeds` and `epoch`, both `[E]` int32. The
    seeds are owned (built from `None` or an int, or with `copy=True`) or bound by pointer
    (a caller `wp.array`, or a `PerEnvSeededRNG`'s `seeds_warp`); `SeedState` and the
    generators never write bound seeds, and `set()` on them raises `RuntimeError`. (The
    rng does: `set_seeds_warp`, and so `Course.generate(seeds=...)`, overwrites
    `rng.seeds_warp` in place, which is the caller's array when the rng or
    `CourseConfig.seeds` was given a `wp.array`.) The epoch is owned, zero-initialised
    and changed only on device: `advance(mask=None)`, `reset_epoch(mask=None)` and
    `set(seeds, ids=None)` are allocation-free launches on the call's stream (by default
    the `default_stream=` given at construction; the facades pass their own `stream=`),
    so they can be recorded. `track_gen.seeds.effective_seed(seed, epoch)` is the host
    twin of the prologue below.
  - **Effective-seed prologue.** Every generation body opens with one kernel (after the
    epoch advance, if any) that derives the seed the pipeline reads from `(seed, epoch)`:
    epoch 0 uses the seed unchanged, any other epoch a MurmurHash3-finaliser hash of the
    pair (`track_gen.ops.hash32`). The pair fully reproduces an env's output. A masked
    generation re-derives the masked envs' effective seeds only.
  - `TrackGenerator` / `GateGenerator`: `seeds` (a `SeedState` bound to
    `rng.seeds_warp`), `inputs` (`{"regen_mask": ...}`, a persistent zero-initialised
    `[E]` int32 array with a stable address), `generate(..., advance=False, mask=None)`
    and `enqueue(..., advance=False, mask=None)`. `advance=True` advances the epoch of
    every env, or of the envs with `mask[e] != 0`, before generating; `mask` requires
    `advance=True`, must be a contiguous `[E]` int32 array on the generator's device, and
    is validated on the host before any launch. With a mask, only the masked envs get new
    output: the others keep the effective seed their output was built from, so a reseed or
    epoch reset made in between reaches them at their next masked generation, or at the
    next one without a mask. Outside a capture each `(advance, mask array)` combination
    records and caches a self-graph of its own, up to 16 per producer
    (`track_gen.exec.runner.MAX_KEYED_GRAPHS`; further ones run eagerly with a one-time
    `RuntimeWarning`), except on a legacy-plugin producer, which runs every
    `advance`/`mask` variant eagerly; `graph` stays the one of a plain `generate()`.
    `GraphRunner.run()` takes the cache `key=`.
  - `Track.generation` / `GateSequence.generation`: `[E]` int32, the epoch each env's
    current output was built with (the epoch, not a change counter: a reseed does not
    change it, and an epoch reset brings earlier values back). Optional (default `None`,
    so results built by hand still construct), allocated by the facades and copied by
    `clone()`.
  - `Course.regenerate(*, mask=None, advance=False, stream=None, sync=True)`,
    `Course.enqueue_generate(*, stream=None, mask=None, advance=False)`, `Course.inputs`
    (`{"regen_mask": ...}`) and `Course.seeds` (the generator's `SeedState`). With a mask,
    only the masked envs advance and have their progress (and localizer) state reset;
    unmasked envs keep their course, derived sub-tool data and runtime state bitwise; an
    all-zero mask is an exact no-op. Both hold even after an out-of-band reseed or epoch
    reset, which takes effect for an env when a mask selects it (or for every env at the
    next unmasked regeneration). Without `advance`, the masked envs are rebuilt from
    their current `(seed, epoch)` and respawned: they get new courses only if you wrote
    new seeds for them. The pipeline has fixed shapes, so it still recomputes every env:
    unmasked envs are recomputed from the effective seed their course was built from, to
    identical values. A masked commit that skips that recompute is roadmap PR10.
  - **Conditional skip.** `LaunchContext.skip_unless(flag, body)` and
    `track_gen.ops.any_set(ctx, mask, flag)`. `Course.regenerate(mask=...)` puts its body
    behind them, so a step where no env is masked skips the regeneration on device. The
    skip is a CUDA conditional graph node, used only inside a capture Warp registered on
    that stream when `wp.is_conditional_graph_supported()` holds (CUDA driver and
    toolkit 12.4+). Everywhere else the body runs unconditionally, which is correct
    because it must be an exact no-op when the flag is 0. Warp's eager `capture_if`, which
    reads the flag back on the host, is never used (a flag on another device raises
    `ValueError`). A caller graph holding the conditional node cannot be used as a CUDA
    child graph.
  - **`track_gen.interop.torch`**: `registered(stream=None, *, graph=None,
    capture_mode=None, conditional=False)`, entered inside `torch.cuda.graph(g)`,
    registers torch's capture stream with Warp (an external capture on the library's
    long-lived wrapper, opened only if neither that handle nor the capture it joined is
    registered yet) and yields it, so lifetime anchoring works under torch capture; it
    raises `CaptureError` outside one. `conditional=True` also allows the conditional
    skip. It is opt-in because a conditional node restarts the stream's capture under a
    new CUDA capture id, after which torch's caching allocator no longer routes the
    graph's later torch allocations to its private pool: with `conditional=True`, record
    the masked regeneration after every torch op that allocates.
    `retain(torch_graph, *owners)` keeps the owners alive as long as the torch graph
    (`graph=g` does it for the producers recorded in the block). Torch is imported only
    when these functions are called; `import track_gen` still does not import it.
  - `PerEnvSeededRNG.sample_*_warp(..., out=None)`: with `out=`, a preallocated array of
    the exact output shape, dtype and device (`ValueError` otherwise, before any launch),
    the samplers write into it and return it instead of allocating, so sampling can be
    recorded.

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
- **The seed copy at the start of every generation is now the effective-seed prologue**
  (roadmap PR4): one kernel reads the rng's seeds and the generator's epochs instead of a
  `wp.copy` of `rng.seeds_warp`. At epoch 0, which is where every env stays unless you
  advance it, it writes the seeds unchanged: generator outputs are byte-identical to the
  previous release, and the goldens are unchanged.
- **The rng must live on the generator's device.** The facades now bind
  `rng.seeds_warp` by pointer (`SeedState`) instead of copying it into their seed buffer
  every generation, so a `PerEnvSeededRNG` on another device (e.g. a `cuda` rng feeding a
  `cpu` generator, which used to run) raises `ValueError` at construction. Build the rng
  on the generator's device.
- **Reseeding through the rng keeps the epochs.** After an advance,
  `rng.set_seeds_warp(new, None)` generates the batch of `(new, epoch)`, which differs from
  a fresh generator seeded with `new`. Call `gen.seeds.reset_epoch()` as well to reproduce
  a fresh generator. `Course.generate(seeds=...)` now does both: it also zeroes every
  env's epoch, so it reproduces the courses of a fresh `Course` built with those seeds.
- **`PerEnvSeededRNG.set_seeds_warp` no longer loses its reseed to a later partial
  sample.** It now mirrors the new states into the samplers' double buffer. Before, a
  sample with explicit `ids` copied the whole stale buffer back, so the envs it did not
  select reverted to stale states from before the reseed. Their later samples therefore
  differ from earlier releases: they now follow the reseed. Generator outputs are not
  affected (the pipeline seeds its own RNG from the seeds, not from these states).

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
- PR4 CPU tests cover epoch-0 byte identity
  (`test_epoch0_is_byte_identical_to_the_pr3_seed_copy_path`: every registered track and
  gate generator, every output field, against the PR3 seed-copy body),
  `advance(mask)` (masked envs change, unmasked envs stay bitwise identical, `(seed,
  epoch)` reproduces an env, `generation` equals the epoch), the `SeedState` bound and
  owned rules, the `set_seeds_warp` mirror fix, sampler `out=` (same values as the
  allocating call, no allocation), and `Course.regenerate(mask=...)` (masked envs get new
  courses and a progress reset, unmasked envs and their state are unchanged, an all-zero
  mask is an exact no-op, also after an out-of-band reseed or epoch reset). The
  no-host-sync and launch-stream drivers include the advance, masked-regeneration and
  `out=` paths. On the fake CUDA device: the bounded keyed self-graph cache,
  `skip_unless` lowering (including a registration that forbids conditionals and the
  flag device check) and `interop.torch.registered()` (conditional opt-in, a stream that
  joined a registered capture).
- `tests/test_inplace_regen_cuda.py` (CUDA) records on a non-current side stream with
  Warp's mempool disabled and `wp.capture_launch` forbidden, and compares poisoned
  replays with eager twins driven the same way:
  - caller graphs holding `generate(advance=True, mask=m)` of both facades (default
    generator; mask from `inputs["regen_mask"]` or the caller): a new batch per replay
    for masked envs, unmasked envs bitwise unchanged. A mismatch is an xfail only when the
    eager reference itself is shown not to be bitwise reproducible;
  - self-graphs cached per `(advance, mask)`, and an invalid mask refused before any
    launch under capture;
  - `Course.regenerate(mask=..., advance=True)` with an all-zero then a partial mask, in
    the conditional and the forced-unconditional lowerings, and under `torch.cuda.graph`
    with and without `interop.torch.registered(conditional=True)`, against twins run
    eagerly (`enqueue_generate()`, never a graph); torch ops that allocate after the
    regeneration under the default `registered()` keep allocating from the torch graph's
    pool;
  - a 100-step RL loop in ONE caller graph (agent stub, `course.step()`, a device done
    rule writing `inputs["regen_mask"]`, `regenerate(mask=done, advance=True)`) against
    an eager twin, in track and gates mode and both lowerings.

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
