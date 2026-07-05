# Repulsive-Growth Centerline Generator — Design Spec

**Date:** 2026-07-05
**Status:** Design (pre-implementation)
**Depends on:** the pluggable generator framework (2026-06-21), the constant-width
relaxation + inflation tail (2026-06-17), the graph-capture-break precedent
(2026-06-18 relaxation quality).
**Evidence base:** `docs/superpowers/spikes/2026-07-05-repulsive-growth-phase1/`
(validated pure-Warp growth port `grow_warp.py`; 64/64 valid through the standard
tail; ~0.18 s @ E=64, ~5.1 s @ E=8192 on an RTX 4090).

---

## 1. Goal & context

Promote the validated *repulsive-growth* spike into a first-class registered
generator named `"repulsive"`. It grows each env's centerline from a small circle
under a hard ratcheting length constraint while a tangent-point (TP) energy keeps
the curve self-avoiding, confined to a disc domain seeded with per-env random disc
obstacles, coarse-to-fine `N = 64 → 128 → 256` with area-based stall-stop. The
result is paper-quality serpentine tracks (Henrich et al., *Generating Race Tracks
With Repulsive Curves*; Yu/Schumacher/Crane, *Repulsive Curves* SIGGRAPH 2021).

It is the first generator that is **not CUDA-graph-capturable**: the growth loop
records a fresh `wp.Tape` per iteration (autodiff gradients) and reads back an
area-convergence scalar every `K` iterations to drive stall-stop. Both are illegal
inside a capture region. On CUDA it therefore runs **eagerly** every call — the
spike's ~0.18 s / ~5.1 s numbers are already eager-CUDA numbers, so the generator
works on CUDA today; it just forfeits the ~1000× replay win the captured generators
enjoy. That cost is the headline caveat this spec surfaces everywhere a user meets
the generator (config docstring, docs page, README, benchmarks).

The downstream tail is untouched: the generator writes the grown closed centerline
at `config.num_points` and the standard pipeline (constant-spacing resample to
`0.6·half_width` → XPBD relax → inflate → validity) runs verbatim. The spike's
central finding — that feeding the tail a `0.6·hw`-spaced curve is what lifts yield
to 100% — is now **automatic**, because that resample is exactly what the pipeline
already does; the generator does nothing special for it.

## 2. Decisions (locked)

| Decision | Choice |
|---|---|
| Capture contract | Add `capturable: bool = True` to `GeneratorSpec`. `"repulsive"` registers `capturable=False`. `TrackGenerator.generate()` branches: capturable generators keep the auto-capture-then-replay CUDA path; a non-capturable generator runs the **whole pipeline eagerly** on CUDA every call (same code path as `cpu`), NOT an error. Transparent eager fallback beats a raised error because the generator genuinely runs on CUDA — an error would break CUDA usage that the spike proved works. |
| Guard surface | `tests/test_warp_graph.py` filters its `_ALL_GENERATORS` parametrization to `capturable` specs only (else `test_autocapture_replay_matches_eager[repulsive]` would attempt an illegal capture and fail). A new registry test asserts `reg.get("repulsive").capturable is False` and that the CUDA facade takes the eager branch for it. |
| Module | `track_gen/_src/warp_generate_repulsive.py`, registering `"repulsive"`. Host-side loop control (stage transitions, per-window stall readback, early exit) is acceptable given `capturable=False`. Scratch is allocated **once** (sized for the max stage `N=256`); coarse stages operate on the `[0:E·N_stage]` prefix of the same buffers. The one unavoidable per-iteration allocation is the `wp.Tape` — the very reason for `capturable=False`, documented as future work (persist one tape or hand-write analytic adjoints). |
| Obstacle layout | Seed-driven, device-side, deterministic per `(seed, env)` — a Warp kernel `wp.rand_init(seeds[e]·_OBSTACLE_SALT + off)`, matching every other generator. Angular-stratified disc placement (one disc per `2π/k` wedge + per-env phase) with an **analytic radial band** `rad ~ U(r_init + r_disc + 0.05·r_dom, c_frac·r_dom)` — **no rejection loop**. Replaces the spike's host torch rejection sampling. Wall + inner discs are point rings (wall weight 1.0, discs 0.25). |
| World scale anchor | **[AMENDED post-spec]** Absolute sizes anchor directly to `config.scale` like every other generator (no bezier coupling, no per-batch measurement): `P_ref = _DOMAIN_SCALE_REF · config.scale` with a fixed constant `_DOMAIN_SCALE_REF = 4.9029` (the spike's *actual* build_setup median at `scale=1`, the validated E=64 config — the spec's original bezier-coupled `_BEZIER_PERIMETER_REF ≈ 5.05` has drifted with the bezier defaults, exactly the risk §5 flagged). `r_dom = repulsive_domain_frac · P_ref`, `r_init = r_dom / repulsive_domain_init_ratio`. At `scale=1.0` the geometry is numerically identical to the spike's validated 64/64 regime — **no final rescale** (barycenter-pinned growth already centers it). Mirrors how `polar`/`voronoi` anchor to the bezier extent `1.44`. |
| Config | `repulsive_*` fields on `TrackGenConfig`, defaults = the spike's tuned config, validated in `__post_init__`, house-format docstrings including an explicit cost warning. `repulsive_stages[-1]` must equal `num_points` (the tail input resolution). |
| Output + validity | The generator writes the closed grown centerline `[E·num_points]` into `out_centerline` and sets `out_valid = 1` for every env (like `polar`/`voronoi`); the shared downstream validity gate (turning ≈ 2π, thickness ≥ `(1−relax_tol)·hw`, no-NaN, width floor, optional border check) decides real validity. A self-crossing born during growth drives thickness → 0 → invalid there, so no generator-side self-intersection retry is needed. |
| Coarse-to-fine upsample | Between stages, upsample `N → 2N` with the pure-Warp NaN-aware `warp_pipeline.arc_length_resample_inplace` (input stride `M=N_old`, output stride `num=N_new`) — device→device, torch-free. The periodic in-stage resample (same `N`) uses `warp_pipeline.resample_uniform`. Per-stage circulant Sobolev rows + the closed-form stage schedule are precomputed **once on host** with numpy at `alloc_scratch` time. |
| Obstacles as content | Out of scope. The per-env disc layouts die with generation (a resume-point note only). Exporting them as props / `DiscChecker` obstacles is deferred. |
| torch-free | The production module imports only `warp` + `numpy` (host precompute: `numpy.fft.irfft` for the Sobolev circulant rows, `numpy.log1p` for the stage schedule). No torch anywhere in the import graph; a `sys.modules` guard in the test suite enforces it. |

## 3. Architecture

### 3.1 Config additions (`track_gen/_src/types.py`)

New `TrackGenConfig` fields (defaults = the spike's tuned config), namespaced
`repulsive_*`, documented in the class docstring in the existing house format:

- `repulsive_grow_mult_min: float = 4.5` — target perimeter as a multiple of the
  init-circle perimeter (lower bound of the per-env `U` draw).
- `repulsive_grow_mult_max: float = 5.5` — upper bound. Overfill `= grow_mult /
  domain_init_ratio`; higher → denser mazes at some yield cost.
- `repulsive_domain_frac: float = 0.35` — domain radius as a fraction of `P_ref`;
  sets the grown curve's absolute scale vs `half_width`. Tighter → richer folds.
- `repulsive_domain_init_ratio: float = 4.0` — `r_dom / r_init` (reference: 4).
- `repulsive_obstacle_count_min: int = 8` / `repulsive_obstacle_count_max: int = 12`
  — per-env inner disc count `k ~ randi[min, max]` (~9.9/env).
- `repulsive_obstacle_radius_min_frac: float = 0.02` /
  `repulsive_obstacle_radius_max_frac: float = 0.045` — disc radius as a fraction
  of `r_dom` (brackets the reference's 0.025 ratio).
- `repulsive_ratchet_rate: float = 0.012` — per-iteration length growth factor.
  `≤ 0.013` holds 64/64 with folds; `≥ 0.016` collapses folds (physical limit).
- `repulsive_alpha: float = 3.0` / `repulsive_beta: float = 6.0` — TP energy
  exponents; obstacle inverse power `p = beta − alpha`.
- `repulsive_tau: float = 0.4` — normalized flow step size.
- `repulsive_w_len: float = 30.0` — small inert length regularizer weight.
- `repulsive_stages: tuple[int, ...] = (64, 128, 256)` — coarse-to-fine schedule;
  last entry must equal `num_points`.
- `repulsive_settle_iters: int = 40` — settle-phase iteration budget above ratchet.
- `repulsive_resample_every: int = 25` — periodic arc-length reparameterization.
- `repulsive_stall_window: int = 16` — iters between stall checks / early-exit
  readbacks.
- `repulsive_stall_area_tol: float = 0.05` — freeze an env once past target length
  AND its enclosed-area relative change over a window is below this
  (reparameterization-invariant shape-convergence tolerance).
- `repulsive_deactivate_obstacles: bool = True` — zero inner-disc weights once an
  env reaches its target length (wall kept live); closes the halos.

`__post_init__` validation (all raise `ValueError` with the offending value):
range fields ordered (`min ≤ max`) and positive; counts `≥ 1`; `domain_init_ratio
> 1`; `0 < domain_frac`; `ratchet_rate > 0`; `alpha, beta > 0`; `stages`
non-empty, strictly increasing, each a positive multiple of 4; and — only when
`generator == "repulsive"` — `repulsive_stages[-1] == num_points`.

### 3.2 Capture contract (`generator_registry.py`, `track_generator.py`)

- **`GeneratorSpec`**: add `capturable: bool = True` (frozen dataclass field with a
  default, so the four existing registrations are unchanged). Update the class
  docstring: `generate` is "pure Warp, in-place, zero-alloc, no host sync **when
  `capturable=True`**; a `capturable=False` generator may use host-side control and
  per-call allocation and is run eagerly on CUDA."
- **`TrackGenerator.__init__`**: cache `self._capturable = self._generator_spec.capturable`.
- **`TrackGenerator.generate()`**: the CUDA branch becomes
  `if _is_cuda and self._capturable:` → the existing warm-up + `wp.ScopedCapture` +
  `wp.capture_launch` path; `else:` → eager `self._run()` (covers `cpu` and
  non-capturable CUDA). `_CAPTURING` stays `False` on the eager path, so `_sync`
  performs real device syncs and the `resample_constant_spacing` N_max readback runs
  normally. Determinism, the fixed-batch contract, and the returned-`Track`-instance
  contract are all unchanged.

### 3.3 Generator module (`track_gen/_src/warp_generate_repulsive.py`)

Ported verbatim from the spike's `grow_warp.py` growth kernels (proven at parity),
with three production changes: (a) obstacle layout is a seed-driven Warp kernel, not
host torch rejection sampling; (b) scratch is allocated once and stage buffers are
slices; (c) the stage upsample uses `arc_length_resample_inplace`, not the torch
oracle. Pieces:

- **Differentiable energy kernels** (recorded under one `wp.Tape`): `_tp_energy_k`
  (dense `O(N²)` tangent-point pairs, constant `±2` circular exclusion),
  `_obstacle_energy_k` (inverse-power ring repulsion with in-kernel per-env
  deactivation branch), `_length_penalty_k` (inert regularizer). Gradient via
  `tape.backward` (matches torch autograd ~2e-7 rel).
- **FFT-free Sobolev preconditioner**: `_conv_k` circular convolution against the
  fixed circulant row `h` (numpy `irfft` of `1/(λ_k^s + ε)`, precomputed per stage
  on host at alloc time).
- **Optimizer-step kernels** (outside the tape): `_ratchet_k`, `_length_grad_k`,
  `_numden_k`/`_project_k` (Sobolev-orthogonal length projection), `_gmean_k`/
  `_gmax_msl_k`/`_step_k` (barycenter pin + normalized step), `_perim_bc_k`/
  `_rescale_k` (hard rescale to the ratcheted target), `_freeze_update_k`
  (area-convergence stall detector).
- **`_sample_obstacles_k`** (new, seed-driven): one thread per env; draws `k`,
  per-env phase, disc radii, and analytic-band radial distances from
  `wp.rand_init(seeds[e]·_OBSTACLE_SALT + off)`; writes the wall ring
  (weight 1.0) and `k` inner-disc rings (weight 0.25) into `obs_pts`/`obs_mw`,
  zeroing unused disc columns. Deterministic per `(seed, env)`; no rejection loop.
- **`_init_circle_k`** (new): writes the radius-`r_init` circle into the coarsest
  stage prefix.
- **`RepulsiveScratch`** + `repulsive_alloc_scratch(config)`: per-env state arrays
  (`L_target/L_init/L_final/reached/frozen/area_prev`, reductions), `obs_pts/obs_mw`
  `[E·M]`, N-dependent buffers (`g/lg/ainv_lg/rs_out/rs_seg/rs_s`) sized at `N=256`,
  and the per-stage circulant rows + closed-form stage schedule (host numpy).
- **`generate_repulsive_warp(seeds_wp, config, out_centerline, out_valid_wp, scratch)`**:
  sample obstacles → seed `L_final` from per-env `grow_mult` draw → init circle →
  the host-driven coarse-to-fine + stall-stop growth loop → final periodic resample
  → copy the `N=256` closed centerline into `out_centerline` → fill `out_valid = 1`.
- **Registration**: `register(GeneratorSpec(name="repulsive",
  alloc_scratch=repulsive_alloc_scratch, generate=generate_repulsive_warp,
  capturable=False))`, plus the import line in `generator_registry._ensure_loaded`.

### 3.4 Module boundaries

Growth is self-contained in `warp_generate_repulsive.py`; it reuses only
`warp_pipeline`'s `resample_uniform` / `arc_length_resample_inplace` / `_fill_i32_k`
and `rng`-style per-env `wp.rand_init` seeding. It writes only the two orchestrator-
owned output buffers. The tail (`resample_constant_spacing → warp_relax → inflate →
validity`) is unchanged and unaware the generator is non-capturable. No other
`_src` module imports the repulsive module (it self-registers lazily like the rest).

## 4. Verification strategy

- **Registration + capturable flag:** `reg.get("repulsive")` present, callables,
  `capturable is False`; the CUDA facade takes the eager branch (no `wp.Graph`
  built) for it.
- **Determinism:** two same-seed runs give byte-identical `gen_centerline`
  (per device); distinct seeds give distinct loops. Obstacle layout is bit-identical
  per `(seed, env)`.
- **Output contract:** `gen_centerline` shape `[E·num_points, 2]`, finite, closed
  (last→first gap ≤ 3× step); post-tail `Track` respects the NaN-tail contract
  (finite `< count[e]`, NaN after).
- **Yield + shape:** post-tail valid fraction `> 0.5` (target 64/64 at defaults,
  matching the spike); median post-relax compactness `< 0.85` so the auto-enrolled
  `test_shape_variety.py::test_no_registered_generator_is_degenerate` passes; the
  repulsive family lands in the foldy band (compactness ≈ 0.15).
- **Capture suite untouched:** `test_warp_graph.py` filters to capturable specs;
  the four existing generators still pass replay==eager. A regression check confirms
  `repulsive` is excluded, not silently capturing.
- **CPU + CUDA:** the per-generator test runs `@pytest.mark.parametrize("dev", DEVS)`
  over `["cpu"] + (["cuda"] if available)`; CUDA is exercised on the local 4090 and
  MUST NOT be filtered out.
- **torch-free:** `import track_gen` then `assert "torch" not in sys.modules`; the
  repulsive module imports only warp + numpy.
- **Docs/UI/assets:** docs build clean with the new `generators/repulsive` page in
  the toctree; `viz.render_readme_assets` emits `generator-repulsive.png` and the
  refreshed strip/grid; `test_readme_assets.py` and `test_param_explorer.py` updated
  and green.

## 5. Risks & open questions

- **Speed limitation is the defining trait.** ~0.18 s @ E=64, ~5.1 s @ E=8192 vs
  ~2 ms for bezier (~1000×+). It is surfaced in the config docstring, a docs
  `.. warning::` callout, the README feature note, and the benchmarks row, with the
  recommended usage being **regeneration-cadence / staggered-slice** (regenerate a
  slice of envs per step, or regenerate rarely and hold the batch) rather than
  every-frame regeneration. Graph capture (persist one tape or go analytic) is the
  future work that would erase most of the per-iter overhead.
- **World-scale anchor constant. [AMENDED post-spec]** Anchoring is now scale-only:
  `P_ref = _DOMAIN_SCALE_REF · config.scale`, `_DOMAIN_SCALE_REF = 4.9029` — a FIXED
  constant, not a per-batch bezier measurement (removing the bezier coupling this bullet
  originally worried about). The value is the spike's own build_setup median at `scale=1`,
  measured today; the spec's earlier `≈5.05` had already drifted with the bezier defaults
  (current measured median ≈4.90), so the fixed spike-derived constant is used to keep the
  generator identical to the validated ground truth. Determinism footnote [RESOLVED]: the
  growth gradient is now hand-written analytic adjoints (per-vertex gather, no atomics),
  replacing the per-iter `wp.Tape`. Output is byte-identical run-to-run PER DEVICE on both
  CPU and CUDA (the tape's ~2e-6 atomic gradient noise, which used to amplify chaotically on
  CUDA, is gone). Cross-device equality is NOT claimed (fp32 rounding differs). The analytic
  adjoints validate against a float64 reference + finite differences to ~2e-6 rel — tighter
  than the tape's own fp32 error. Removing the tape also dropped per-iter cost (E=64 CUDA
  full generate ~206→126 ms, E=1024 ~882→335 ms) and removed the interior CUDA-graph-capture
  blocker (capture itself is still future work — see §6).
- **Eager-CUDA numbers are throttle-sensitive** (±1.5× from GPU clock/pool state, per
  the spike). Benchmarks report them as indicative, not contractual.
- **`num_points` coupling.** `repulsive` requires `num_points == repulsive_stages[-1]`
  (256 default). A user changing `num_points` for repulsive must update `stages`;
  validated in `__post_init__`.

## 6. Out of scope

- CUDA graph capture of the growth loop (per-stage graphs). The interior per-iter-tape
  blocker is now REMOVED (analytic adjoints shipped); what remains is host-side control
  flow (stage transitions, stall readbacks, early exit), so capture is a wiring exercise.
- BVH far-field for the `O(N²)` TP + obstacle sums.
- Obstacles as gameplay content (props / `DiscChecker` export).
- Shipping the actual runtime Warp tail *inside* the spike (already the production
  path here — the generator feeds the standard tail directly).
- Per-env style-range sampling for `repulsive_*` fields (scalar knobs only, like
  `voronoi`).

## 7. Definition of done

- `GeneratorSpec.capturable` exists (default `True`); the four existing generators
  are unchanged and still graph-capturable; `test_warp_graph.py` filters to
  capturable specs and stays green.
- `"repulsive"` is registered with `capturable=False`; `TrackGenerator.generate()`
  runs it eagerly on CUDA (no `wp.Graph` built) and on CPU; determinism, fixed-batch,
  and same-`Track`-instance contracts hold.
- `repulsive_*` config fields exist with the spike's tuned defaults, validated in
  `__post_init__`, documented with an explicit cost warning; `repulsive_stages[-1]
  == num_points` is enforced for the repulsive generator.
- Obstacle layout is a seed-driven Warp kernel, bit-identical per `(seed, env)`, no
  rejection loop; the module imports no torch (`sys.modules` guard passes).
- `tests/test_generate_repulsive.py` passes on cpu and cuda: registration +
  `capturable=False`, output shape/finite/closed, NaN-tail, determinism + diversity,
  post-tail yield `> 0.5` (64/64 at defaults), compactness in the foldy band.
- Param explorer exposes the `repulsive_*` controls (visibility guards in sync);
  `test_param_explorer.py` green.
- Docs: `generators/repulsive.rst` (with the speed `.. warning::`) in the toctree;
  `overview.rst`/`benchmarks.rst`/`index.rst` count + rows updated;
  `writing-a-generator.rst` notes the `capturable` flag; docs build clean.
- README feature table + pipeline strip line updated; `render_readme_assets`
  `GENERATORS` extended; `generator-repulsive.png` + refreshed strip/grid rendered;
  `test_readme_assets.py` green.
- `benchmarks/compare_generators.py` auto-includes `repulsive` (registry-driven);
  the `benchmarks.rst` row is filled and flags the wall-clock.
- Full suite green on the 4090 (`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest
  -q`, no `-m "not cuda"`); torch-free guard passes; docs build + asset render
  succeed.
