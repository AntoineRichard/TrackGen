# Changelog

## 0.2.0.dev0 (unreleased, `v2` branch)

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
