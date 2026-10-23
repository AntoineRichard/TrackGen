# TrackGen library-readiness review

> **Status on the `v2` branch.** This review describes `v0.1.0` (`1d8a6e5`). On `v2` (tags `v0.2.0.dev0`–`v0.2.0.dev3`, then 0.2.0.dev4 in progress), the §1 quick wins and roadmap PR0–PR12 have landed: capture (CI lanes, no host syncs, `LaunchContext` and explicit streams, `GraphRunner` caller capture, `SeedState` with masked in-graph regeneration), fail-closed truncation (PR5), the public plugin surface (one registry with entry points, `track_gen.ops`, the `Stage` protocol with static/dynamic `Params`, the `track_gen.testing` conformance kit), the open pipeline (planner and `PipelineSpec`, pluggable relax/width/elevation/validity stages, fail bits and masked commit in PR10; gate stages, the ordering registry and the `stations` adapter in PR11), and PR12 (the BOUNDED tier: `ctx.bounded_loop` with its host-poll / conditional / unrolled lowerings, repulsive as a BOUNDED Stage that stops once every env has frozen; HOST stages with `HostOps`, segmented execution and `require_caller_capturable`). The SDF collision box query now takes the exact minimum over the box outline. Still to do: PR13–PR16 and the PR15b docs restructure. See `CHANGELOG.md` for what changed; the analysis below is kept as written.

**Date:** 2026-10-21 · **Commit:** `1d8a6e5` · **Scope:** `track_gen/` (~16k lines), tests, benchmarks, viz, docs, packaging

**Questions asked:**
1. Which design decisions would stop users from adding new track-generation algorithms?
2. Is everything CUDA-graphable? This library is CUDA-first, so every core API must support graph capture.

**Companion documents:**
- [`findings.md`](findings.md): all 138 verified findings, with file:line references and fixes.
- [`target-architecture.md`](target-architecture.md): the proposed target architecture, the design decisions behind it, and a 17-PR roadmap.

**Method:** see §8. In short: static analysis only, since no GPU was available. Fifteen parallel reviewers produced the findings, and every finding was checked by two adversarial verifiers. Warp semantics were checked against the Warp sources at `v1.14.0` (the current floor) and `1.19.0.dev1`.

---

## 1. Bottom line

### Extensibility

**Adding a new centerline generator is possible today, but only through private imports.** It works if you:
- import `track_gen._src.generator_registry`;
- subclass `TrackGenConfig` to carry your parameters;
- follow conventions that are not written down anywhere (a closed loop, exactly `num_points` finite `vec2f` points, an extent of `1.44*scale`, ad-hoc RNG salts).

Everything beyond the centerline itself requires editing library internals:
- a different relaxation solver;
- variable width;
- a new elevation profile;
- an extra validity check;
- a gate generator that controls its own z, tangent or size;
- a new utility inside `Course`.

The kernels are not the problem. The obstacles are architectural:
- private registries;
- a 107-field monolithic config;
- a hard-coded stage sequence;
- closed string enums with `if/elif` dispatch;
- fixed result schemas;
- a gate pipeline that reaches into the track generators' private kernels;
- a `Course` closed over two modes.

### Graphability

**The kernels are capture-clean; the APIs are not.** Kernel launches use fixed shapes, outputs are preallocated with stable pointers, seeds live on the device, and there is no per-call allocation. That is a good foundation. But:

| Public API | Can a caller record it in their own graph? |
|---|---|
| `TrackGenerator.generate`, `GateGenerator.generate`, `Course.generate` | **No, under any setting.** Each one opens its own `wp.ScopedCapture`, calls `wp.capture_launch`, calls `wp.synchronize()` unconditionally, and takes a process-wide lock. |
| `generate()` with `generator="repulsive"` | **No.** It cannot even self-capture, because of a host readback and `break` at `warp_generate_repulsive.py:845`. |
| `Course.step/reset` and every utility's `query/update/sample/bake/reset` | **Only after `track_gen.set_capturing(True)`.** This process-global flag suppresses the `wp.synchronize()` that otherwise follows every launch. Replays are then correct. |
| `PerEnvSeededRNG.set_seeds_warp` | Yes. |
| RNG `sample_*`, `localize.curvature/speed_profile`, `*.clone()` | Not cleanly. They allocate on every call. |

Several things also give **silently wrong results under capture**:
- **Configs:** they are mutable and held by reference, while CUDA graphs bake their values. After the first `generate()`, CPU and CUDA silently diverge.
- **Seeds:** there is no device-side way to reseed. A captured generation therefore replays the identical batch every time.
- **Truncation:** tracks truncated at `N_max` pass validity. The only warning is a host readback, and it never runs on the graph path.

### What to do first

These are the top actions, mapped to roadmap PRs in §6.

1. **Separate enqueuing from execution policy.** Give every producer (generators, Course) a pure `enqueue(stream)` that only launches work, with no sync, lock, allocation or nested capture. Make self-capture a convenience built on top of it. Warp cannot nest graph launches inside a capture (§4.5), so this is the only design that lets callers record generation.
2. **Remove the host syncs and the three capture flags.** Detect capture from the stream actually used (`Stream.is_capturing`), and pass an explicit `stream=` to every launch.
3. **Move regeneration state onto the device.** Add a seed `epoch`, `advance(mask)`, and a masked commit, so a recorded graph produces new tracks. Make truncation fail validity.
4. **Publish a plugin surface.** One registry with entry points, public sync-free primitives (`track_gen.ops`), per-stage `Params` dataclasses, and a shipped conformance kit (`track_gen.testing`).
5. **Open the pipeline.** Turn relaxation, width, elevation, validity and the gate stages into registered stages, and add per-point attribute channels.

### Quick wins (small, independent, low risk)

- **Repulsive:** delete the readback at `warp_generate_repulsive.py:845-846`, then set `capturable=True`. The rest of the loop already follows a fixed schedule that does not depend on device data (§4.6).
- **Test CI:** add a CI job that runs the tests. Today the only workflow builds the docs, and CUDA tests skip silently when torch is CPU-only.
- **Truncation:** have `_cs_scan_k` write a `truncated[e]` flag instead of silently clamping (`warp_pipeline.py:401`).
- **Elevation profiles:** raise on an unknown `z_profile`. Today it silently falls through to `"noise"` (`warp_zprofile.py:216`).
- **Validation:** replace the `assert relax_solver == "xpbd"` with a `ValueError` (`track_generator.py:76`). The assert is stripped under `python -O`.
- **Gate registry:** add `capturable` to `GateGeneratorSpec`.
- **Scratch:** remove the `_Scratch.__getattr__` fall-through (`warp_pipeline.py:1480-1495`).
- **Packaging:** exclude `_experimental` from the wheel.

---

## 2. Architecture today

```
TrackGenConfig (107 fields, mutable)   GateGenConfig (38 fields, duplicates generator knobs)
        │                                         │
TrackGenerator ──registry.get()──► GeneratorSpec   GateGenerator ──► GateGeneratorSpec
  │ on CUDA: lock → 3× warm-up → ScopedCapture → capture_launch → wp.synchronize()
  ▼                                               ▼
_run_pipeline (fixed):                            _run_gate_pipeline (fixed):
  generate → resample_constant_spacing →            generate(_GateStaging vec2f) → sphere relax →
  XPBD relax → inflate_warp(resample, frames,       arclen → z-profile → lift → tangents → frames/validity
  offset[const width], z-profile[if/elif], fused validity, lift→vec3f)
  ▼                                               ▼
Track (10 fixed fields, [E*N_max], NaN-padded)    GateSequence (9 fixed fields)
        └───────────────┬─────────────────────────┘
                        ▼
  Utilities: CollisionChecker · DiscChecker · FrameChecker · PropSampler · CheckpointSampler
             ProgressTracker · TrackLocalizer · HeightFieldBaker · CourseLine
             (allocate once, bind once, query() without arguments; _sync() after every launch)
                        ▼
  Course (mode ∈ {"track","gates"}; hard-wired tool set; owns refresh Graph B)

Global state: runtime._CAPTURING (public via set_capturing), warp_pipeline._CAPTURING,
              warp_gate._CAPTURING, runtime._CAPTURE_LOCK (process-wide threading.Lock)
```

### What is already good, and should be kept

- **Pointer-stable results.** `Track`, `GateSequence` and every utility result are allocated once, rewritten in place, and returned as the same instance. That is exactly what downstream graphs need.
- **Capture-friendly kernels.** Launch shapes are fixed (`E`, `E*N_max`), per-env `count` is device-side, and tails are NaN-padded. There is no Python branching on device data in the hot paths, apart from repulsive. Per-env fallbacks are device masks (voronoi, checkpoint best-of-K).
- **Seeds are device data.** Per-env RNG is stateless (`wp.rand_init(seed*K + c)`), so replay is driven by the seed buffer.
- **The utilities follow a bind-once, `query()`-without-arguments pattern.** All mutable per-step state lives on the device (progress `prev_pos`, localizer warm start). Once the syncs are gone, replay is correct.
- **Some seams already exist and deserve to be generalized:**
  - `GeneratorSpec.capturable`;
  - the orchestrator owns the generator outputs while the generator owns its scratch;
  - `warp_relax` receives `capturing=` by injection instead of reading a global.
- **Graph testing technique is strong.** Poisoned-buffer replay and eager-twin trajectory comparisons are the right methods and should be extended.

---

## 3. Design decisions that block new generation algorithms

### 3.1 No public plugin surface (high)

- **Private registries.** `GeneratorSpec` and `register` exist only in `track_gen._src.generator_registry`. `track_gen/__init__.py` does not re-export them, and `tests/test_public_api.py` asserts the public surface exactly.
- **Hard-coded loaders, no entry points.** `_ensure_loaded` hard-codes six imports: *"Add one import line per new generator"* (`generator_registry.py:57-63`). The gate registry has its own fixed tuple (`gate_generator_registry.py:27-32`). `pyproject.toml` declares no entry points.
- **Inconsistent duplicate handling:**
  - The track `register()` silently overwrites. A plugin that registers a built-in name before the first lookup is then clobbered by the lazy import.
  - The gate `register()` raises on a cross-module duplicate. A plugin that pre-registers a built-in name breaks every later lookup.
- **Every shipped generator uses private helpers** from `warp_pipeline`: `_pipe._sync`, `_pipe._fill_i32_k`, `_pipe._arc_resample_inplace`, `_pipe.self_intersections_inplace`, `_pipe.GenScratch`, `_pipe._ccw`. No public primitives exist.

### 3.2 The configuration monolith (medium)

- **One huge config.** `TrackGenConfig` has 107 fields (`types.py:537-674`):
  - about 48 are specific to one generator (`checkpoint_*` ×8, `repulsive_*` ×19, `voronoi_*`, `polar_*`, bezier style ranges, …);
  - about 21 are never read at runtime (`energy_*`, `tp_*`, `smooth_finish*`, the Fourier fields, `relax_chunk_size`, `relax_use_warp`, `max_regen_iters`, `min_angle`, `output_mode`).
- **A plugin cannot declare its own parameters.** The working pattern is to subclass `TrackGenConfig`. Two such plugins cannot be combined. Registry-wide tools (`tests/test_shape_variety.py`, `benchmarks/compare_generators.py`) build a base `TrackGenConfig` for every registered name, and crash on plugins whose parameters live on a subclass.
- **Validation is coupled to specific names.** `__post_init__` hard-codes generator names (e.g. `generator == "repulsive"` at `types.py:776`) and closed string enums (`z_profile`, `relax_solver`, `relax_accel`, `output_mode`, `gate_ordering`, `gate_align`).
- **The gate config duplicates the track knobs.** `GateGenConfig` repeats them (`rad`, `edgy`, `polar_*`, `voronoi_*`, `checkpoint_*`) so it can be duck-typed into the track samplers. A new generator therefore needs fields in two configs.

### 3.3 A fixed pipeline (high)

- **Fixed stage sequence.** `_run_pipeline` (`warp_pipeline.py:889-968`) always runs generate → constant-spacing resample → XPBD → `inflate_warp`. `inflate_warp` (`1699-1950`) has its own fixed order. Nothing can be inserted, swapped or reordered.
- **Relaxation.** `TrackGenerator` asserts `relax_solver == "xpbd"` (`track_generator.py:76`), although the config documents `{"xpbd","energy","tp_sobolev"}`. The validity thickness gate is tuned to XPBD's separation target, so another smoother would collapse yield even if you could wire it in.
- **Elevation.** `apply_z_profile` is an `if/elif` whose `else` silently runs `noise` (`warp_zprofile.py:202-222`). The allowed names are a literal set (`types.py:24`), and "is this profile knot-based" is a name tuple in two places (`warp_pipeline.py:1505`, `1899`). The module calls itself "pluggable", but the only way to plug in is to edit it.
- **Width.** The half-width is a scalar everywhere:
  - XPBD target and `R_min` (`warp_relax.py:384`);
  - band setup (`551`);
  - `scratch.w` is filled with a constant (`warp_pipeline.py:1869`), so the validity `w_floor` check is a tautology;
  - the offset kernels take `half_width: float` (`52-131`).
- **Validity.** It is one fused kernel that writes one bit (`_validity_k`, `625-712`). It requires the turning number to be exactly ±2π (`657`), which rules out figure-8 or grade-separated layouts even when elevation is present.

### 3.4 A narrow, partly undocumented generator contract (high/medium)

- **Fixed output shape.** The output is `[E*num_points] vec2f` plus `[E] valid`, and nothing else (`generator_registry.py:27-29`). There is no channel for z, per-point width, banking, surface tags or speed hints.
- **Hidden invariants.** All `num_points` points must be finite: `_cs_scan_k` sums every slot, so a single NaN gives `count=0` and a silently invalid env (`warp_pipeline.py:395`). The loop is implicitly closed and must be simple, and the size convention is `1.44*scale`.
- **Undocumented execution model.** On the first CUDA call, a generator's Python body runs **4×** (3 warm-ups plus 1 record) and its kernels execute 4×. Later calls run the Python body 0× and the kernels 1×. CPU runs both 1× per call. A stateful generator therefore drifts silently between CPU and CUDA.
- **No per-call device inputs** other than seeds. Per-env difficulty or curriculum values can only be passed through config, which is a baked host scalar.
- **Scratch name collisions.** `_Scratch.__getattr__` searches the generator's scratch *before* the inflate buffers (`warp_pipeline.py:1480-1495`). A plugin scratch field named `w`, `kappa`, `z`, `area_a` or `area_b` silently hijacks the pipeline's buffer.

### 3.5 The gate pipeline is a separate silo (high)

- **Private, 2D-only contract.** Gate generators fill a private `_GateStaging(vec2f position, count)` (`warp_gate.py:777-790`) and must call the private `finish_ordered_gates` / `alloc_order_scratch` themselves.
- **Built-in gate generators import track generators' private kernels.** For example, `warp_generate_polar_gates.py:6-11` imports `_polar_controls_k`, `_BASE_RADIUS` and `_BEZIER_EXTENT`. `warp_generate_gates.py:38` passes a `GateGenConfig` where a `TrackGenConfig` is expected. A new track generator does not get a gate variant for free.
- **Whatever the generator supplies is overwritten.** After `generate()`, the pipeline:
  - overwrites z from the profile (`853-861`);
  - recomputes tangents by closed-loop central differences (`354`);
  - uses a roll-free orientation (`406`);
  - applies a uniform `half_size` (`407`).
- **Spacing and validity are XY-only.** So 3D-native layouts (vertical loops, split-S, stacked gates), banked gates, per-gate sizes and open sprint courses cannot be expressed. For a drone-gate library, this is the main blocker for new algorithms.
- **No capturable opt-out.** `GateGeneratorSpec` has no `capturable` flag, so every gate generator is force-captured.
- **Fragile scratch.** The gate scratch is a positional 6-tuple, and `GateGenerator` reads `scratch[5]`.

### 3.6 Course and the utilities are closed (medium)

- **Closed modes.** `Course` accepts only `mode ∈ {"track","gates"}`. It builds its own RNG and generator; it cannot be given one.
- **Hard-wired tools.** The tool set is fixed, and Course dispatches with `isinstance` and private attribute access (`course.py:431-447`, `524-604`). There is no `Component` protocol.
- **Inconsistent utility interfaces.** Method names differ by class (`bind_inputs` vs `bind`; `bake` vs `sample` vs `refresh`).
- **`Track` is used as a universal representation.** `CourseLine` fakes the band fields with NaN, and no consumer checks which fields it actually needs.
- **Closed-loop topology is hard-wired in every kernel:** relax, z-profiles, collision, props, the localizer and progress all wrap at the seam.

### 3.7 Tooling hard-codes the generator list (medium)

- **Hard-coded tools.** `viz/param_explorer.py` hard-codes every generator's controls and gate counts. Tests and benchmarks enumerate fixed names. There is no conformance kit, and each generator's test file re-implements the same contract checks by reading private scratch.
- **The "one registry line" promise does not hold.** The original design spec promised that adding a generator collides "only on the single registry line". In practice every shipped generator edited 5-10 shared files.
- **The private ABI is stable only by accident.** The unmerged `grammar` generator branch would still run as an out-of-tree plugin, because the private helpers it uses have not changed. Nothing defines or pins that ABI.

### 3.8 Simulated third-party plugins

A reviewer wrote real plugin sketches against the current code. They are kept in the session scratchpad, not the repo.

| Scenario | Possible without editing `track_gen`? | Main friction |
|---|---|---|
| (a) Capturable track generator with its own params | Only via private imports | private registry; `TrackGenConfig` subclass; undocumented contract; `__getattr__` collisions; explorer shows no controls |
| (b) Gate generator from (a) | Only via private imports | second config subclass; private `_GateStaging` and helpers; no reuse of (a) |
| (c) Replace XPBD / disable relaxation for one generator | Only via private imports | `assert`, hard-coded relax branch; validity tuned to XPBD; options are overriding private `_run` or monkeypatching |
| (d) Variable-width inflation | **No** | scalar width baked through the contract, resample, XPBD, offset, validity and `Track` |
| (e) New elevation profile | Only via private imports | closed name set; requires monkeypatching `apply_z_profile` process-wide, which also changes gates |
| (f) New runtime utility usable in `Course` | Standalone yes; in `Course`, only via private overrides | needs `set_capturing`; no `Component` protocol; fixed `StepResult` |

---

## 4. CUDA graph capture

### 4.1 Critical blockers

1. **Generation facades** (`track_generator.py:182-208`, `gate_generator.py:165-187`). Every CUDA `generate()`:
   - takes `_CAPTURE_LOCK`;
   - on the first call, runs 3 warm-ups, then `wp.synchronize()`, then its own `wp.ScopedCapture`;
   - on every call, ends with `wp.capture_launch(self._graph)` followed by `wp.synchronize()`.

   Inside a caller's capture, the nested capture and the sync raise. Removing the sync would not help either: a graph launch cannot be recorded into another capture (§4.5).
2. **`Course.generate`** (`course.py:451-521`). It inherits the above, and adds its own nested refresh graph (Graph B). It also:
   - allocates and uploads seeds from the host when given an integer (`481-482`);
   - builds every sub-tool on the first call, including host readbacks used to size buffers (`checkpoints.py:264`).
3. **Repulsive** (`warp_generate_repulsive.py:845`). `if int(s.frozen.numpy().sum()) >= E: break` is a device-to-host readback plus a host branch on device data. With `capturable=False`, `TrackGenerator` runs the *whole* pipeline eagerly, including the resample's readback.

### 4.2 Structural causes

- **Three private, process-global capture flags:** `runtime._CAPTURING` (public via `set_capturing`), `warp_pipeline._CAPTURING` and `warp_gate._CAPTURING`. `set_capturing` does not reach the pipeline flags. Which flag guards which sync depends on knowing the call chain.
- **About 48 `_sync()` sites call a global `wp.synchronize()`.** That synchronizes *every* GPU, raises if any Warp capture is active on the device, and stalls unrelated simulator streams.
- **No `stream=` anywhere.** There are 187 `wp.launch` calls in `track_gen/_src` and none passes a stream, so all work goes to Warp's process-global current stream:
  - Under a default `torch.cuda.graph`, which captures on a private side stream, library work is silently left out of the graph, or the capture is invalidated by a context sync.
  - `area_a.zero_()` (`warp_pipeline.py:983`) is a latent wrong-stream bug, because Warp's memset uses the current-stream sentinel.
- **A process-wide, non-reentrant `threading.Lock`** serializes all CUDA generation on all GPUs. Its premise is "there is one CUDA device stream". It also forces `Course.generate` to be non-atomic.
- **Self-capture is the only mode, and capturability is all or nothing.** One non-capturable generator makes the capturable tail run eagerly too.

### 4.3 Silent wrong results under capture

| Issue | Where | Consequence |
|---|---|---|
| Mutable config held by reference; graph bakes scalars and sizes | `types.py:53`, `track_generator.py:84`, `warp_pipeline.py:917` | After the first CUDA `generate()`, changing `half_width` is ignored on replay but applied on CPU. Changing a size field (`N_max`, `num_points`, …) causes out-of-bounds writes on the eager paths. |
| No device-side reseed or advance | `track_generator.py:188` (host `wp.copy` of seeds) | A recorded generation replays the identical batch. There is no masked per-env regeneration, so in-graph auto-reset is impossible. |
| N_max truncation is silent | `warp_pipeline.py:401`, `1127` | The scan clamps `count` and records nothing. The only warning is a host readback, which is skipped whenever `_CAPTURING` is set, and that includes warm-up. Relaxation then smooths the chord into a plausible loop that passes validity. The README and quickstart use `half_width=0.03` with the default `N_max=384`, which is the documented truncating regime. |
| Invalid envs leave finite geometry with `count≥3` | `warp_pipeline.py:709-712`, `1931` | Every utility's degenerate path keys on `count<3`, so it computes plausible garbage for invalid envs. The conventions also disagree: collision is conservative, frame checker permissive, and disc checker has no validity input. Expected invalid rate at E=512: bezier 0.78%, checkpoint 2.34%. |
| Sampler capacities sized by reading back the first batch | `checkpoints.py:258`, `props.py:348` | Capacity is fixed by whatever the first batch happened to need. A static bound exists, but `Track` carries no layout metadata. |
| Warm-up executes plugin bodies 4× | `track_generator.py:196-207` | Stateful generators diverge between CPU and CUDA. |

### 4.4 Utilities

Every per-step utility is otherwise ready: one or two fixed-shape launches, no allocation, device state, stable pointers. Removing `_sync()`, or gating it on the resolved stream's capture state, makes all of them caller-capturable without a global flag.

`localize.curvature/speed_profile`, the RNG `sample_*` methods and `clone()` need `out=` variants.

### 4.5 Warp facts that constrain the fix

These were checked against the Warp sources at `v1.14.0` and `1.19.0.dev1`.

- **`wp.capture_launch` cannot be recorded into another capture.** It is a plain `cudaGraphLaunch`, and the repo itself has hit error 900 here. Splicing a graph is possible only through `capture_if`/`capture_while` bodies. **So caller capture requires an enqueue-only path that re-records the launches. Removing the trailing sync is not enough.**
- **`wp.synchronize()` behaves differently for registered and unregistered captures.** It raises under any Warp-registered capture on any GPU. Under an *unregistered* capture on another stream (the default `torch.cuda.graph`), it calls `cuCtxSynchronize`, and CUDA invalidates the torch capture.
- **`Device.is_capturing` is the wrong probe.** It is true for any registered capture on the device from any thread, including the library's own self-capture. Probe `Stream.is_capturing` on the stream actually used.
- **`device.captures` is keyed by `wp.Stream` object identity**, and `stream_from_torch` returns a new wrapper on every call. Look up registration by capture id or stream handle.
- **`capture_if`/`capture_while` have strict requirements.** They need a Warp-registered capture plus a CUDA 12.4+ toolkit and driver. In an unregistered capture they fall back to host polling with `synchronize_stream`. Warp ≥1.16 fixes GH-1641, where allocations made during capture were freed too early.
- **Warm-up must use real launches.** At 1.14, `load_module`/`force_module_load` after a CPU launch preloads the wrong CUDA variant (GH-564). Generic kernels also add overloads at launch time.
- **A `wp.Graph` keeps alive only modules, not buffers.** The objects that own the buffers must outlive any caller graph.
- **Version floor.** Recording into a caller's capture works at the 1.14 floor. Conditional lowerings should be feature-gated, and the floor raised to ≥1.16 at 1.0.

### 4.6 Repulsive can be made capturable

The only data-dependent host control flow is the readback at `:845`. Stage transitions, resample cadence and stall checks all depend only on the host loop counter, and the views are fixed slices.

Fully unrolled, the loop is about 4.1k graph nodes at defaults (269 iterations × 15 launches, plus resamples). Envs that have converged are frozen, and the O(N²) kernels already skip frozen envs, so the extra iterations after convergence cost milliseconds. Making the freeze absorbing (masking the periodic resample by `frozen`) gives identical results with and without the early exit. It also removes today's dependence on which other envs are in the batch.

A port to `capture_while` is *not* a drop-in replacement. The resample cadence (every 25 iterations) and the stall-check cadence (every 16) are coprime, and `capture_while` fails inside torch captures.

---

## 5. Recommended direction

[`target-architecture.md`](target-architecture.md) has the full design, with signatures, code sketches and alternatives considered. It is a synthesis of three independently drafted proposals (capture-first, plugin-first and incremental), scored by three judges with different priorities. Its core ideas:

| Layer | Module | Role |
|---|---|---|
| L0 | `track_gen.exec` | `LaunchContext` is the only way to emit device work: stream-explicit `launch/copy/fill`, plus `skip_unless` and `bounded_loop`. `run()` picks Record / Eager / SelfGraph on each call from `Stream.is_capturing` of the resolved stream. `prepare()` is a real-launch warm-up. A private `_compat` is the only code that touches Warp internals. |
| L1 | `track_gen.results`, `track_gen.seeds` | `Layout` metadata; additive `Track` fields (`half_width`, `fail_mask`, `truncated`, `generation`, `attributes`); capability protocols (`HasCenterline`, `HasBand`, `HasGates`); `SeedState(seeds, epoch)` with a device-side `advance(mask)`. |
| L2 | `track_gen.ops` | Public, sync-free, allocation-free primitives that take `ctx` first: NaN-aware resampling, channel carry-through, geometry checks, samplers, curves, `@wp.func` helpers, hashed RNG salts. |
| L3 | `track_gen.plugin` | `Stage` with declared reads/writes, `Params` with `static` (baked) vs `dynamic` ([E] device arrays) fields, `CaptureClass` (STATIC / BOUNDED / HOST), one `Registry` with entry points, duplicate policy, quarantine, `describe()`. |
| L4 | `track_gen.pipeline` | `PipelineSpec` (an ordered, rewirable list of stages), presets, a planner that allocates once, `Program` (enqueue, masked commit, `inputs`, `params`, `tap`). |
| L5–L6 | facades, `track_gen.tools`, `Course` | Thin `TrackGenerator/GateGenerator` with `generate(stream=, advance=, mask=, sync=False)` and `enqueue()`. A `Component` protocol. `Course` becomes a composition root with a single `regenerate(mask)` body that can be recorded in the caller's graph. |
| L7 | `track_gen.interop.torch`, `track_gen.testing` | Torch capture registration and lifetime retention; a conformance kit run over everything registered, in this repo's CI and in plugin authors' CI. |

**The contract every capturable API obeys:**
- no host sync, readback, allocation, lock, nested capture, `capture_launch`, `ScopedStream` or `zero_()`/`fill_()` on the hot path;
- one resolved stream per call;
- anything that changes between replays is device data;
- HOST-class stages raise `NotCapturableError` before launching anything when a caller is capturing.

**Gates for free.** Any centerline generator gets a gate variant through a `stations` adapter. Native gate generators can declare `provides={"z","tangent",…}`, and the pipeline then skips the stages that would overwrite those values.

---

## 6. Roadmap summary

The full roadmap is in [`target-architecture.md`](target-architecture.md#roadmap-17-pr-sized-steps-in-7-phases). Goldens stay byte-identical except for four deliberate, release-noted changes:
- truncation now fails validity;
- repulsive uses an absorbing freeze;
- hashed RNG salts at 1.0;
- invalid envs are poisoned by default at 1.0.

| Phase | PRs | Outcome |
|---|---|---|
| 1. Unblock caller capture | PR0 CI lanes and capture probes · PR1 delete syncs and flags · PR2 `LaunchContext` and explicit streams · PR3 `Program`/`run()` and enqueue-capable facades/Course | **Every generation API except repulsive, and every utility, can be recorded in a caller's graph** (Warp, external and torch captures) |
| 2. Replay correctness | PR4 `SeedState`/epoch/`advance(mask)`, torch interop · PR5 truncation fails closed | Recorded graphs produce new tracks; no silently clipped tracks |
| 3. Plugin surface v1 (parallel with 1–2) | PR6 Registry and entry points · PR7 public `ops`, deduplicated built-ins · PR8 `Stage`/`Params`, `describe()`-driven explorer · PR9 `track_gen.testing` and author docs | **Out-of-tree generators without private imports** |
| 4. Open pipeline | PR10 planner and track stages, masked commit · PR11 gate stages, ordering registry, `stations` · PR12 BOUNDED tier, repulsive capturable | Relax/elevation/validity/gate stages are pluggable; repulsive is graphable |
| 5. Channels | PR13 per-point channels, width registry, `Track.layout/validate/view`, ingestion stage | Variable width, attributes, caller-owned outputs |
| 6. Components | PR14 `Component` protocol and adapters · PR15 `Course` as composition root, in-graph regeneration | RL-style `[step → done → regenerate(mask)]` entirely in one graph |
| 6b. Docs | PR15b docs restructure (getting started → user guide → extending → reference → internals), migration guide 0.1 → 0.2, regenerated figures, CI-checked snippets | Docs published with the `v2` merge |
| 7. 1.0 | PR16 hashed salts, poison default, frozen configs, shims removed, Warp ≥1.16 | Clean surface |

Phases 1–2 and Phase 3 can proceed in parallel; Phase 3 depends only on PR0 and PR2.

---

## 7. Decisions needed from you

1. **Warp floor.** Which Warp versions do Isaac Sim, Isaac Lab and Newton bundle? The answer decides whether the floor can move to 1.16 at 1.0, and whether conditional-node lowerings can be on by default.
2. **Async `generate()`.** The proposal makes `generate()` non-blocking by default, with `sync=True` as an opt-in. This changes behaviour for any caller that reads results from another stream without a sync.
3. **Invalid-env default.** Should invalid envs be poisoned by default (NaN geometry, `count=0`), or keep finite geometry with a `fail_mask`? And what sentinel should progress use (frozen with `dist_to_next=0`, or NaN)?
4. **Retry vs candidate axis for low-yield generators.** Is "regenerate with `advance(mask=1-valid)`" enough, or is a candidate axis worth its memory cost (`E*K` candidates, then select the first valid)?
5. **Topology.** Is a batch-static `closed` flag enough, or do you need per-env open/closed in mixed batches?
6. **Plugin naming.** Should third-party names be required to be dotted (`pkg.name`)? Should built-ins get namespaced aliases?

---

## 8. Method and caveats

**Static analysis only.** No Warp install and no GPU were available on the review machine. Warp and torch behaviour was checked against local source checkouts (Warp `v1.14.0` and `1.19.0.dev1`; torch 2.14.1). Every claim about runtime behaviour on hardware, especially torch side-stream capture and the cost of a 4.1k-node graph, should be confirmed on a GPU. Roadmap PR0 adds the probes that do this.

**Pipeline:**
1. 10 subsystem reviewers and 5 cross-cutting sweeps: host syncs, allocations and baked scalars, global state and streams, a caller-capture matrix, and a third-party plugin walkthrough.
2. The first pass produced 317 raw findings. Many overlapped, and merging duplicates left 106 unique ones.
3. Two rounds of a completeness critic added 8 targeted investigations. These covered Warp version semantics, ingestion and serialization, mixed batches, the plugin execution model, truncation, invalid envs, repulsive's capture options, and plugin ABI stability measured against the unmerged `grammar` generator. They contributed 44 more findings.
4. Each of the 150 findings was checked by a factual verifier and an impact skeptic. 138 survived and 12 were dismissed (listed in [`findings.md`](findings.md)).
5. The architecture was produced by three independent proposals, three judges with different priorities, and a synthesis.

The previous review ([`../code-review-2026-06-27.md`](../code-review-2026-06-27.md)) found RNG defects. Some of them are still open, and are listed in [`findings.md`](findings.md) as low-severity carry-overs.
