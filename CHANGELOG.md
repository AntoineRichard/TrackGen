# Changelog

## 0.3.0rc3 (release candidate, tag `v0.3.0rc3`)

The third candidate repairs Repulsive's growth and full-road obstacle contract,
then completes the racing documentation with current examples, visual setting
comparisons, algorithm and solver movies, and native light/dark media. Repulsive
seeded outputs change; the other generators retain their existing byte references.
PR4's auto-spacing finding #4 remains deferred pending the user's decision.
This candidate requires downstream racing validation before a final release.

### Repulsive repair

- Require the entire finished road band to clear every sampled filled disc and
  stay inside its circular wall after resampling, relaxation and inflation.
  Preserve internal barrier metadata and require the final check in custom
  pipelines. Cell convex-hull certificates are conservative for concave/folded
  strips; discs in the central loop hole remain allowed.
- Guard swept optimizer/growth/resampling transitions, preserve edges on
  refinement, correct edge-tangent energy and Sobolev order, and use double
  precision for energy reductions, length arithmetic and metric operations.
  Raw validity requires finite feasible target-reaching geometry; optional
  freezing requires genuine small movement and projected stationarity. A valid
  budget-limited curve need not be converged.
- Remove `repulsive_w_len` and `repulsive_deactivate_obstacles` without
  compatibility. Require `alpha > 1`, `alpha + 2 <= beta < 2 * alpha + 1` and
  positive `repulsive_tau`. Lower default growth bounds from 4.5–5.5 to 2.0–3.0.
  Repulsive `size` now participates in unscaled width reservation and can change
  raw obstacle layout and curve shape.
- Repulsive seeded outputs change; other generators' outputs and all RNG salts
  and draw order stay unchanged. Public result fields and poison/keep policy
  stay unchanged. Repulsive programs add `repulsive_disc` (bit 8) and
  `repulsive_wall` (bit 9) failure reasons. Inspect validity and `fail_mask`;
  aggressive tuning can fail after the shared tail. See the
  [migration guide](docs/library-readiness/archive/upgrading/from-0.2.rst).

## 0.3.0rc2 (release candidate, tag `v0.3.0rc2`)

The second candidate addresses review findings #1–#3 and #5–#14 from PR #4.
Generator outputs and existing byte goldens are unchanged from rc1. See the
[0.2 to 0.3 migration guide](https://github.com/AntoineRichard/TrackGen/blob/v0.3.0rc2/docs/upgrading/from-0.2.rst)
for the current API replacements.

- Repair the track-gallery, Fourier-ablation and PDF-report tools; remove ignored
  regeneration options and identify historical report measurements.
- Share RNG initialization between the randomizer and plugin examples; remove
  unused pipeline composition, self-drawn parameter machinery and obsolete test filters.
- Restore the Taubin math symbols and complete GraphRunner execution contracts.
- Add a default-lane CPU statistics gate against an exact rc1 reference for all
  built-in generators, elevation and random gate ordering, with statistical tolerances.
- Auto-derived config spacing retains its existing behavior: when replacing
  `half_width`, also pass `spacing=None` to derive it again. Review finding #4 remains
  deferred pending the user's decision on that public config contract.

The candidate awaits downstream racing-repository validation. Final 0.3.0 still
requires the user's explicit approval.

- Remove `keying=` from `ops.sample_counts` / `sample_grid_points`: salts are
  hashed uint32 namespaces, and draws differ from 0.2 for the same salt. Built-in
  generator outputs remain identical to rc1; they already used the hashed path.

- Remove `registry.describe(...).accepts_ctx`: every generator receives `ctx`;
  conformance reports no longer repeat this invariant flag.

## 0.3.0rc1 (release candidate, tag `v0.3.0rc1`)

The first release candidate for 0.3 includes the legacy API removals, hashed built-in
RNG streams without a reproduction mode, poison defaults, frozen configurations,
required result storage, and the updated documentation from dev1-dev4. See the
[0.2 to 0.3 migration guide](https://github.com/AntoineRichard/TrackGen/blob/v0.3.0rc1/docs/upgrading/from-0.2.rst)
for every break and its replacement. Package metadata and `track_gen.__version__`
both report `0.3.0rc1`.

Final release requires validation in the downstream racing repositories and the
user's explicit approval.

## 0.3.0.dev4 (documentation milestone, `v3` branch)

Document the complete 0.2 → 0.3 migration in `docs/upgrading/from-0.2.rst`, covering
all removals, changed RNG streams, poison defaults, frozen config/subclass recipes
and complete result storage. Current guides and executed examples use the active
contracts; pre-switch figures and tuning measurements are labeled historical.
The rc and downstream racing-repository validation remain pending before 0.3.0.

## 0.3.0.dev3 (defaults and contracts milestone, `v3` branch)

- Default `on_invalid="poison"` everywhere: Track/Gate configs, `BatchSpec`, presets,
  Program construction, and `/2` dictionaries that omit the policy. Invalid envs
  have `count=0`, NaN geometry and padded attributes; failure diagnostics and
  generation remain readable. Set `on_invalid="keep"` explicitly to inspect failed
  geometry. Runtime utilities retain their invalid-env sentinels.
- Freeze `TrackGenConfig` and `GateGenConfig`. Use `dataclasses.replace` for changes
  and decorate config subclasses with `frozen=True`. To recouple point spacing after
  changing half-width, use `replace(cfg, half_width=..., spacing=None)`.
- Require Track's `truncated`, `generation`, `fail_mask`, `half_width` and `layout`,
  and GateSequence's `generation`, `fail_mask` and `layout`. All their concrete array
  fields and layouts reject `None`, including through validation and `out=`. Use
  `Track.alloc(Layout(...))` / `GateSequence.alloc(...)` for complete manual storage.
  Metadata mappings still default empty; Centerline and capability-source layout
  inference are retained. A no-offset Track retains a required NaN half-width array.
- Retain the static `edgy` host transform and all other surviving geometry arithmetic;
  this milestone changes invalid-result defaults and data contracts, not RNG streams.

## 0.3.0.dev2 (RNG milestone, `v3` branch)

Built-in RNG streams now use hashed salts. Track and gate geometry changes for the
same seed; rebaseline generated-output goldens and cached datasets once. No legacy
RNG config field, compatibility path or byte-identical reproduction mode is provided.
The migration guide is `docs/upgrading/from-0.2.rst`.

- Add public `track_gen.ops.rng`: device `init` / `init_at` and host twins
  `key_host`, `init_host` / `init_at_host`, with formulas frozen for plugin API 1.
- Switch all six track generators, all five gate generators, `order.random_pairs`
  and uniform/random-walk/noise elevation to fixed named salts. Names do not depend
  on Stage IDs; gate generators share their matching track generator's streams.
  Effective seeds, epochs, `PerEnvSeededRNG` and `randomize.uniform` are unchanged.
- Add keyword-only `keying="hashed"` to public ops samplers. Their default remains
  `keying="linear"`, preserving draws for existing plugin callers.
- Replace the old-keying byte comparison with a distribution gate covering every
  built-in track/gate generator, nonflat elevation and random-pair ordering. All 133
  distributions across 19 cases passed with 8192 envs per case (two blocks of 4096)
  on the same GPU. The largest absolute yield delta was 0.00146484375 (about 0.1465
  percentage points). It checks validity, counts, length, compactness, curvature and
  elevation; byte goldens are rebaselined separately.
- A `Course` capacity derived from newly sampled geometry may change. Pass explicit
  `max_checkpoints=` to preserve the desired checkpoint buffer capacity.

This milestone changes valid geometry. The following milestone changes defaults and
contracts; historical figures have not all been remeasured for the new seeds.

## 0.3.0.dev1 (`v3` branch)

The removal milestone of the 0.3 cleanup. Surviving configurations retain their
geometry and RNG keying. The hashed RNG switch, poison defaults, frozen configs and
required result fields follow in separate milestones. The complete migration guide
is `docs/upgrading/from-0.2.rst`.

### Breaking removals

- Remove the `_src` registry modules/views, helper re-exports and module hooks;
  use `track_gen.plugin.registry` and `track_gen.ops`.
- Remove frozen `generate_*_warp`, `*_alloc_scratch`, `apply_z_profile*` and scalar
  sampling helpers, with their scalar/style kernels. Use generator facades or presets.
- Remove `GateGeneratorSpec`, its adapter, staging alias and legacy gate ordering
  helpers. Gate plugins are Stages.
- Retain `GeneratorSpec` as the documented simple track-plugin path: static
  generators need only an allocator and a generate callable, using the same pipeline
  and conformance kit. Its generate callable must accept `ctx=`. Remove the
  `accepts_ctx` constructor flag and the context-free five-argument adapter, with
  `LegacyPluginStreamError`, legacy-stream introspection/options and the shared
  legacy capture lock.
- Remove `style_sampling` and its three range fields from configs and Bezier Params;
  use `randomize=`. Remove the five built-in Stage-class `.params` aliases; use `.Params`.
  Runtime `gen.params` and `GeneratorSpec(params=...)` remain supported.
- Remove the 21 unused track config fields (oracle energy/tangent-point/smoothing,
  regeneration, old Fourier and dead relaxation knobs), `output_mode`, and the five
  inert gate config parity fields. Removed keys now fail rather than being ignored.
- Remove inert `validity.gate_default.gate_width` / `check_crossing` params;
  set width on `gate_frames.default`. Nonzero bars remain checked per env.
- Remove `set_capturing`, `Course.set_capturing`, `_CAPTURING`, the ignored private
  `capturing=` arguments, `CourseLine.track`, and `tools.Progress(checkpoints=)`.
  Use launch contexts, `CourseLine.centerline` and `Progress(source=)`.
- Refuse `track_gen.pipeline/1` dictionaries; load and re-save with 0.2 as `/2` first.

No compatibility shims, tombstones, warning cycles or reproduction modes are added.

## 0.2.0 (2026-10-24, tag `v0.2.0`)

The first release of the 0.2 line: everything in 0.2.0.dev0 to 0.2.0.dev6 below (the
`v2` branch, merged into `main`). Upgrading from 0.1: the migration guide
(`docs/upgrading/from-0.1.rst`, "Upgrading from 0.1" on the documentation site), organised
by impact: silent changes first. Highlights: every generation, course and utility can be
recorded into your CUDA graph; one plugin registry with entry points and a conformance kit;
a pluggable track and gate pipeline (generators, relax, width, elevation, validity,
ordering, stations); per-point channels and variable width; per-env dynamic knobs, size and
curricula in the graph; `Course` as a composition root over components; invalid envs read
as sentinels (gate on `StepResult.valid`); the SDF box query takes the exact minimum over
the box outline; Warp >= 1.18. 0.2 is the warning release for the 1.0 removals listed in
the migration guide.

## 0.2.0.dev6 (2026-10-24, tag `v0.2.0.dev6`)

Roadmap PR14: the component protocol (`track_gen.tools`) and the adapters of the runtime
utilities, static capacities from the generator's plan, generation-aware self-reset, one
invalid-env sentinel table, `out=` on `curvature()` / `speed_profile()`, and
`check_component` in the conformance kit. Roadmap PR15: `Course` as the composition root
(`Course(generator, components)` over any producer, the presets `Course.track` /
`Course.gates` / `Course.from_config`, `StepResult` mapping access by component name,
`bind(**arrays)` routed by declared inputs, the regeneration as one graph over every
component with a stable `inputs["reset_mask"]`, the course anchoring its components).
Generator outputs and launch sequences do not change; the utilities' outputs for valid
envs are byte-identical; `Course(CourseConfig)` keeps its outputs and launch sequences
(PR15 changes none of them; construction launch streams under an ambient scope
excepted: migration item 20). Roadmap PR15b: the documentation restructure and the
0.1 -> 0.2 migration guide (`docs/upgrading/from-0.1.rst`), with the documentation
examples run in the CPU test lane and a link check; docs, docs tooling and CI only (the
library diff is docstrings and comments: see Documentation).

### Migration (`Course` and runtime-utility users)

Downstream code that uses `Course` / `CourseConfig` or the runtime utilities sees the
changes below. Each one is detailed further down (Added, Behaviour changes). Items 1-8
come from roadmap PR14, items 9-20 from roadmap PR15.

**1. Invalid envs read like poisoned envs in every utility.** For an env whose result has
`valid[e] == 0`, every utility now writes what it writes for an env with no track
(`count = 0`, NaN geometry), whatever the generator's `on_invalid`.
`TrackGenConfig` / `GateGenConfig` expose no `on_invalid`, so **every `Course` runs on
`"keep"` and sees this change**; so does every direct utility on such a result.

- How many envs: measured at the defaults (`E = 256`, seeds 0, CPU): `bezier`, `polar`,
  `voronoi` 0 %; `hull` 1.2 % (3 envs, thickness); `checkpoint` 0.4 % (1 env, turning
  and thickness); `repulsive` 1.6 % (4 envs, thickness, one also turning). `bezier` at
  size 4: 19 % (49 envs, truncated at `N_max`); none at sizes 1 to 3. Gates mode
  depends on the bar width, up to 91 % of the batch: **see item 2**. Rates grow with the
  size and other settings that truncate or thin the tracks: measure yours with
  `(course.result.valid.numpy() == 0).mean()`.
- Every field that changes for an invalid env (`Course` attribute, then the utility):

  | Field | 0.2.0.dev6 | 0.2.0.dev5 (`"keep"`) |
  |---|---|---|
  | `res.contacts` (track mode, `BoxContact`; `CollisionChecker`, both methods) | `oob = 1`, NaN `distance` / `nearest` / `normal`, `boundary = -1` | the answer on the failed geometry |
  | SDF bake (`collision="sdf"`) | NaN texels, boundary id `-1`, box `(-1, -1)`-`(1, 1)` | the failed band's field and box |
  | `res.contacts` (gates mode, posts, `DiscContact`) | `hit = 0`, `disc = -1`, `depth = 0`, NaN `nearest` | hits on the invalid gates' posts |
  | `res.events` (`ProgressEvents`) | frozen: `next_checkpoint` / `laps` / `progress` keep their values, `passed = 0`, `checkpoint_passed = -1`, `wrong_way = 0`, `wrong_checkpoint = -1`, **`dist_to_next = 0`** | tracking on the invalid course (or, with no checkpoints, NaN `dist_to_next`) |
  | A `TrackLocalizer` you build on a `Track` (e.g. `TrackLocalizer(course.result)` in track mode) | NaN `s` / `n` / `n_up`, `segment = -1` | the projection on the invalid centerline |
  | `course.checkpoints` (track mode, sampled) | `count = 0`, NaN slots, `valid = 0`; the sampler's `step` NaN, `truncated = 0` | the invalid track's ring |
  | `course.heightfield.result` (the `HeightField`; also what `bake()` returns) | NaN `height` (as before), `lo = (-1, -1)`, `hi = (1, 1)` | NaN heights, the failed band's box |
  | `PropSampler` | `count = 0`, NaN poses, `step` NaN, `truncated = 0` | the invalid track's ring |
  | `localize.curvature` / `speed_profile` | NaN rows | values |

  Unchanged: `FrameContact` (gates mode, frames: already `hit = 0`, `depth = 0`), the
  course line (`course.course_line.centerline`: already `count = 0`, NaN), so `res.frame`
  (gates mode: the localizer reads the course line: already NaN `s` / `n` / `n_up`,
  `segment = -1`), `CheckpointSet.from_gates` (a view of the gates). With
  `on_invalid="poison"` (a `presets` pipeline) the only change is `dist_to_next`: NaN
  -> 0.
- `valid` is the invalid-env signal (new: `StepResult.valid`, `CheckpointSet.valid`).
  `dist_to_next` now means two things: `0` for an invalid env, NaN for a VALID env with
  no checkpoints (a hand-built set without `valid`). Code that detected a poisoned env
  through `isnan(dist_to_next)` must read `valid`.
- What breaks in an RL loop: a loop that penalises `oob` or ends episodes on it without
  gating on `valid` penalises / ends the invalid envs at every step, and
  `course.reset(done)` respawns them on the same invalid course forever; a NaN in one
  env's observation makes a batch mean NaN. **The recipe** (the docs' four RL loops,
  `docs/utilities/course.rst`, `tutorials/batch-of-tracks.rst`,
  `tutorials/gate-sequences.rst`, `tutorials/cuda-graph-in-a-sim.rst`, are rewritten
  with it):

  ```python
  res = course.step()
  valid = wp.to_torch(res.valid).bool()           # read it before regenerate (it is live)
  reward = torch.where(valid, reward, 0.0)        # a select: never reward * valid (NaN * 0 is NaN)
  obs = torch.where(valid[:, None], obs, 0.0)
  done = done | ~valid                            # done |= 1 - valid
  if done.any():                                  # a host read; see below
      wp.to_torch(course.inputs["regen_mask"]).copy_(done.int())
      course.regenerate(mask=course.inputs["regen_mask"], advance=True)  # a NEW course for done envs
  ```

  In a Warp kernel: `wp.where(valid[e] != 0, r, 0.0)`. Guard the eager call: a
  `regenerate(mask=...)` reruns the whole generator pipeline whenever it runs, even with
  an all-zero mask (on CPU at `E = 256` the step loop gets about two orders of magnitude
  slower). The unguarded call belongs in a CUDA graph recorded through a capture Warp
  registered (`wp.ScopedCapture`, `wp.capture_begin`, the course's self-graph, or a
  `torch.cuda.graph` inside `track_gen.interop.torch.registered(conditional=True)`) with
  conditional nodes (`wp.is_conditional_graph_supported()`: CUDA driver and toolkit
  >= 12.4), where an all-zero mask is skipped on device; anywhere else (eager, a CPU
  capture, a plain `torch.cuda.graph`, an older CUDA) it costs a full generation per call
  (`docs/tutorials/cuda-graph-in-a-sim.rst`). Keep `course.reset(mask)` for valid envs
  that crashed (respawn on the same course); never for an invalid one.

**2. Gates mode: the bar width decides how many envs are valid.** Every gate pipeline
checks each env's bars for crossings (fail bit 4, the per-env check of 0.2.0.dev5).
0.2.0.dev5 still tracked such an env (progress, post hits); 0.2.0.dev6 freezes it and it
hits nothing (item 1), so an invalid env is a slot your agents cannot train on until it
is regenerated. Measured invalid fraction (`GateGenConfig` defaults, `E = 256`, seeds 7,
CPU):

| Gate generator | `gate_width` 0.05 | 0.1 | 0.2 | 0.3 | 0.4 |
|---|---|---|---|---|---|
| `bezier` (the default), `scale = 1` | 0 % | 8 % | 50 % | 78 % | 91 % |
| `hull`, `scale = 1` | 0 % | 9 % | 48 % | 76 % | 91 % |
| `voronoi`, `scale = 1` | 0 % | 1 % | 17 % | 46 % | 80 % |
| `checkpoint`, `scale = 1` | 0 % | 3 % | 13 % | 30 % | 48 % |
| `polar`, `scale = 1` | 0 % | 0 % | 0 % | 5 % | 18 % |
| `bezier`, `scale = 2` | 0 % | 2 % | 15 % | 34 % | 51 % |
| `bezier`, `scale = 3` | 0 % | 0 % | 4 % | 16 % | 29 % |

- **The gates tutorial and the quickstart used `gate_width = 0.4` before 0.2.0.dev6**:
  copied as is, about 91 % of a `bezier` batch at `scale = 1` is invalid (59 of 64 envs
  at `E = 64`, `Course(seeds=7)`; 60 of 64 for the tutorial's generator examples,
  `PerEnvSeededRNG(seeds=0)`). Their `gate_width = 0.4` examples now use 0.1 (the
  configuration and Full Example of `tutorials/gate-sequences.rst`, its `Course`
  example, `getting-started/quickstart.rst`): 2 of 64 for the `Course`, 8 of 64 for the
  generator examples. The bar-width curriculum example keeps 0.4 at `size = 2.0`. With
  the recipe of item 1 at `gate_width = 0.2`, about half of the batch is regenerated at
  every step.
- Lower `gate_width`, or raise `scale` (the gates spread out, so wider bars fit), and
  check the fraction after construction: `(course.result.valid.numpy() == 0).mean()`.

**3. The default `CourseConfig.max_checkpoints` is the plan's bound.** `None` (the
default) now gives `max(3, ceil(max_loop_length / checkpoint_spacing))` with
`max_loop_length = 1.5 * N_max * spacing + dZ` read from the generator's plan (no
readback; `dZ = 0` at flat elevation), instead of `ceil(1.5 * longest valid centerline of
the construction batch / checkpoint_spacing)`.

- Numbers at the `TrackGenConfig` defaults (`N_max = 384`, `spacing = 0.06`, flat:
  `max_loop_length = 34.56`): **70** checkpoints at `checkpoint_spacing = 0.5` (0.2.0.dev5:
  13 for `E = 1`, 17 for `E = 6` or 8, 19 for `E = 32`, 15 to 23 across the five
  sampling generators at `E = 64`, seeds 0), **58** at 0.6, **116** at 0.3 (was 29 at
  `E = 6`), 35 at 1.0. It no longer depends on the seeds, the batch or the course size.
- Memory: every `course.checkpoints` array grows to `[E * M]` (52 bytes per slot and env:
  `position`, `left`, `right`, `tangent`, `up_half`; plus 8 bytes per env for the new
  `valid` / `generation`): 28.5 MiB at `E = 8192`, `M = 70`, against 7.8 MiB at
  `M = 19`. The bound scales with `N_max * spacing`: `N_max = 2048`, `spacing = 0.02`,
  `checkpoint_spacing = 0.05` gives `M = 1229`, 262 MB at `E = 4096`. The sampling kernel
  launches `E * M` threads.
- Outputs: an env's real rows, count, progress and laps do not depend on `M` while its
  ring fits, so every env that fit the 0.2.0.dev5 capacity is byte-identical. Envs the
  old capacity clipped (`course.checkpoint_sampler.truncated == 1`; a size curriculum
  constructed at size 1: 44 % / 94 % of the valid `bezier` envs at size 2 / 2.5) now get
  their full ring, so their progress and laps change.
- When the plan declares no bound, `Course` keeps the 0.2.0.dev5 readback: an elevation
  whose z knobs declare no range and no `z_valid_grade` (every elevated
  `TrackGenConfig`: `z_min` / `z_max` are unbounded dynamic knobs, so an elevated default
  course keeps exactly its 0.2.0.dev5 capacity), a generator that provides its own
  altitudes (`provides` `z`) without a `z_valid_grade` or its own half-widths
  (`provides` `half_width`), a width other than `width.constant`, a plugin geometry stage
  after the constant-spacing resample. The built-in `validity.default`'s `z_valid_grade
  = g > 0` declares a bound (`dZ = g * 1.5 * N_max * spacing`: `M = 277` at `g = 3`,
  `checkpoint_spacing = 0.5`).
- A new `UserWarning`, once at construction, when the defaulted capacity (bound or
  readback) clips a valid env of the construction batch (a cold readback of two `[E]`
  int32 arrays, after the warm-up). Never with an explicit `max_checkpoints`.
- **To keep the 0.2.0.dev5 shape**, pass the old capacity explicitly: print
  `course.checkpoints.position.shape[0] // E` under 0.2.0.dev5 and pass it as
  `CourseConfig(max_checkpoints=...)`. An explicit `max_checkpoints` behaves as before.
  `course.checkpoint_sampler.max_checkpoints` reads the capacity in use.

**4. New fields** (additive; positional construction unchanged):
`StepResult.valid` (the generator result's `[E]` int32 `valid`, the same array;
`clone()` copies it); `course.checkpoints.valid` / `.generation` (`[E]` int32: owned and
written by the sampler with its ring in track mode, the gates' arrays in gates mode);
`course.course_line.centerline.generation` (an `[E]` int32 written at every refresh from
the gates' `generation`; was `None`).

- **`StepResult.valid` is producer state**: it aliases `course.result.valid`, and in gates
  mode `course.checkpoints.valid` / `.generation` alias the `GateSequence`'s. Never write
  into them, and exclude them from code that zeroes or poisons step outputs:
  `dataclasses.fields(StepResult)` and `track_gen.testing.capture.result_arrays(res)`
  now list `valid` (that helper returned `{}` for a `StepResult` before). A masked
  regeneration restores only the masked envs, so an overwritten `valid` stays wrong for
  the others.
- **It is live**: after a `regenerate(mask=...)` in the same iteration it describes the NEW
  courses, while `events` / `contacts` / `frame` still describe the step. Read it before
  regenerating (as the recipe does), or `clone()` the result.
- **Generic field iteration**: `StepResult` gains a raw `wp.array` field next to its
  result dataclasses, and `CheckpointSet` two per-env `[E]` fields next to its `[E * M]`
  ones: code that recurses into every `StepResult` field as a dataclass, or reshapes
  every `CheckpointSet` field to `(E, M, ...)`, must special-case them.
- **New read-only properties**: `.result` on every utility, `CollisionChecker.method`,
  `CheckpointSampler.max_checkpoints`, `PropSampler.max_props`, `CourseLine.result`. A
  subclass or wrapper that assigned attributes of those names (`self.result = ...`) now
  gets `AttributeError`: rename them.

**5. Rebinding after a capture warns.** `Course.bind` and the utilities' `bind` /
`bind_inputs` / `bind_window` emit `track_gen.exec.RebindAfterCaptureWarning` (a
`UserWarning` subclass) when a bound array is replaced after a hot method ran while its
stream (or a CPU capture on its device) was capturing: the recorded graph keeps reading
the old arrays. Record the graph again after rebinding. Binding the same arrays again is
silent. The recipe "rebind, then record again" warns once too: filter it
(`warnings.simplefilter("ignore", RebindAfterCaptureWarning)`) when you re-record on
purpose. A test suite that turns warnings into errors (`-W error`) sees it.

**6. Progress and the localizer reset themselves after a regeneration.** `ProgressTracker`
(through its `CheckpointSet.generation`) and `TrackLocalizer` (through its source's
`generation`) reset an env's runtime state on device at the first `update()` / `query()`
after its `generation` (its epoch) changed, exactly as `reset(mask)` would. `Course`
still resets explicitly (no change in its outputs); a direct user who regenerated with
`advance=True` and deliberately kept the progress state now sees it reset.

**7. Malformed optional members raise at construction.** A hand-built source whose
`valid` or `generation` is present but of the wrong shape, dtype or device now raises
(`ValueError` / `TypeError`, the required members' messages) in `CollisionChecker`,
`TrackLocalizer`, `curvature`, `speed_profile`, `CheckpointSampler`, `CourseLine`,
`CheckpointSet.from_gates` and `ProgressTracker` (on its set); they used to be ignored.

**8. Gates-mode post collision** scans each env's `2 * count[e]` real posts and is given
the gates' `valid` (it scanned `2 * max_gates` posts, skipping the NaN padding): the same
contacts for valid envs, item 1 for invalid ones.

**PR15: `Course` as the composition root.** For `Course(CourseConfig)` code, **outputs
and launch sequences do not change** (construction and every call: byte-identical state
after every step and identical launch traces, buffer roles included, on ten track and
gates variants; CPU goldens `tests/goldens/pr15_course.json` taken at the PR14 commit),
except on CUDA for a course built inside an ambient `LaunchContext.scope(other)` on a
stream other than Warp's current one, whose construction launches change streams (item
20), **no warning is added**, and every 0.2.0.dev5 / PR14 `Course` test passes
unchanged. What you can observe:

**9. The constructor signature** is `Course(generator=None, components=(), *,
stream=None, prepare=True, config=None)`. `Course(cfg)` and `Course(config=cfg)` (the
0.x forms) build exactly the 0.x course, silently: they are `Course.from_config(cfg)`.
`Course(cfg, components)`, `stream=` or `prepare=False` with a `CourseConfig` raise
`TypeError` (as before, with another message). `inspect.signature(Course)` changes.
**Subclasses:** every front door now runs `__init__` (the presets call `cls(generator,
components, stream=..., prepare=...)`), so a subclass's `__init__` runs for
`MyCourse.track(...)` too and must accept that signature; one written as
`__init__(self, config)` keeps working for `MyCourse(cfg)` and `MyCourse.from_config(cfg)`
(it calls `cls(config)`); the presets need the new signature.

**10. `StepResult` gains read-only mapping access** (it does not become a `Mapping`):
`res[name]` (a `KeyError` listing the entries), `res.get(name, default=None)`,
`keys()`, `values()`, `items()`, `name in res`, `dict(res)`. The entries are the
per-step components by name; on a config course exactly the sources of the 0.x fields:
track `progress` [, `collision`]; gates `progress` [, `collision`], `localizer`
(`res.events is res["progress"]`, `res.contacts is res["collision"]`, `res.frame is
res["localizer"]`).

- **Unchanged:** the dataclass and its four fields (`dataclasses.fields`, `asdict`,
  positional construction, `==`), the annotations (`events: ProgressEvents`), truthiness
  (always `True`), `iter(res)` (`TypeError`), `isinstance(res, collections.abc.Mapping)`
  (`False`), the `repr` (the dataclass-generated one, on every course, so `pprint` and
  `rich` lay it out as before), `clone()`. Generic tree utilities that test `Mapping`
  before dataclasses (Lightning's `apply_to_collection` / `move_data_to_device`,
  `dm-tree`'s `map_structure`) see the dataclass they saw, and the mapping access on
  what they return agrees with its fields: assigning a 0.x field (`res.events = x`, as
  they do on their copy) updates its entry on that instance only (`None` removes it;
  copy on write, so a `copy.copy` shares nothing it reassigns).
- **Visible:** `vars(res)` gains `_items` (the entries, a plain instance attribute);
  `dataclasses.replace(res, ...)` rebuilds from the four fields, so it keeps the 0.x
  entries only (on a config course: all of them); duck-typed code that probes `keys()` /
  `__getitem__` now finds them, so `**res` and `dict.update(res)` succeed (they raised
  `TypeError`); `hasattr(res, "__iter__")` is `True` (the attribute is `None`, so
  `iter(res)` still raises); `isinstance(res, collections.abc.Container)` is `True`
  (`__contains__`). Logging or serialization helpers that sniff these to pick a code
  path take another one. A hand-built `StepResult(events, contacts, frame, valid)` gets
  the entries of its non-`None` fields under `progress` / `collision` / `localizer`.

**11. `course.inputs` gains `"reset_mask"`**: an `[E]` int32 array, zeroed at
construction, with a stable address, for your kernels to write the mask of
`reset(mask)` into (`course.reset(course.inputs["reset_mask"])`). Code that expects
exactly one key, or iterates `inputs`, sees two. One more `[E]` int32 allocation at
construction (no launch).

**12. `reset(mask=None, stream=None)`**: the mask is optional; `None` resets every env.

**13. `bind(position, orientation=None, half_extents=None, box_position=None,
**arrays)`**: same positional order and the same 0.x checks and messages first
(`RuntimeError` for a missing `orientation` / `half_extents` / `box_position`, the
shape / dtype / device `ValueError`s). On a config course `bind()` still raises
`TypeError` (now the library's `"Course.bind() missing 1 required argument:
'position'"`) and `bind(None)` the 0.x error (`RuntimeError` when the course has a box
/ disc collision checker, else `ValueError`). An unknown keyword raises
`TypeError("Course.bind: unknown inputs [...] (component inputs: [...])")` instead of
Python's unexpected-keyword `TypeError` (same type, another message); the name of a
generator input raises `TypeError` with the in-place recipe (Deferred). A
refused bind, including a component's `on_bind` raising part-way (the component that
raised is bound again to its previous arrays too), leaves every previous binding in
place and the rebind-after-capture warning armed: a later real rebind after a capture
still warns.

**14. New public members** on every course: `components` (a read-only `name ->
component` mapping, in execution order), `prepare()`, and the classmethods `from_config`,
`track`, `gates`. A subclass that defines class attributes or methods of these names
shadows them, but `components` is a read-only property: assigning an instance attribute
of that name (`self.components = [...]` in a subclass's `__init__`, a plausible name for
reward or observation parts) now raises `AttributeError` (Python 3.11+: `property
'components' of ... object has no setter`; 3.10: `can't set attribute 'components'`);
rename yours. Likewise `TrackGenerator.rng` / `GateGenerator.rng`
are read-only properties now (the `PerEnvSeededRNG` the facade was built with, else
`None`): `gen.rng = ...`, or a facade subclass that stores `self.rng`, raises
`AttributeError`; keep your rng under another name.

**15. Lifetime.** `step()` and `reset()` recorded into a capture Warp registered
(`wp.ScopedCapture`, `wp.capture_begin`, `interop.torch.registered()`) now anchor the
course on that graph, as `regenerate` / `generate` / `enqueue_generate` already did: a
graph that records only `step()` / `reset()` keeps the course, its generator, its
components and its bound arrays alive. The arrays a recorded call reads that the course
does not own are anchored with it: `step()` the binding it read (`course._bind_args`;
after a rebind the course drops its references to the old arrays, the graph keeps
them), `reset(mask)`, `regenerate(mask=)` and `enqueue_generate(mask=)` your mask,
`generate(seeds=array)` your seed array. A CPU graph (`wp.ScopedCapture("cpu")`) needs
no anchor (Warp keeps every array its recorded launches touch); the course object itself
is released there. A plain `torch.cuda.graph` still needs `interop.torch.retain(g,
course, ...)` (pass those arrays too).

**16. Private internals** (some tests and downstream code read them) keep their meaning:
`_cfg`, `_runner`, `_components`, `_by_name`, `_posts`, `_step_result`, `_device`, `_E`,
`_needs_boxes()`, `_refresh_geometry(ctx, mask)`, `_reset_runtime(ctx, mask)`,
`_refresh`, `_fill_posts`, `_enqueue_regenerate`, `_regen_call`,
`_validate_seed_array`, `_check_arr`. **Changed:**

- `_reset_order` is removed: resets run over every component in reverse order (the same
  launches: localizer, then progress).
- `_build_subtools` is removed (its work moved into the common setup): a subclass
  override or a monkeypatch of it no longer runs.
- `_apply_bind` is no longer called by `bind()` or by construction (routing moved into
  `_bind`): calling it re-routes the last binding, but a subclass override or a
  monkeypatch of it no longer runs.
- The regeneration body calls `generator.program.enqueue` directly: an override or a
  monkeypatch of the facade's `_enqueue` no longer runs.
- `_cp_default` is replaced by the checkpoint component's `defaulted`. The checkpoint
  component of a config course with a defaulted capacity is built with
  `max_checkpoints="auto"` (its `max_checkpoints` attribute reads `"auto"`;
  `course.checkpoint_sampler.max_checkpoints` is the same int as before).
- `_step_result` exists from construction (the first `step()` created it).
- `_bind_args` holds only the arrays given to the last successful `bind` (`{name:
  array}`, no `None` values, extra keywords included): it was a dict of the four 0.x
  names with `None` for an absent one, so `course._bind_args["box_position"]` (or
  `["orientation"]` on a course without collision) now raises `KeyError`; use `.get(name)`.
- `_bind(position, orientation, half_extents, box_position)` (PR14) is `_bind(given)`,
  `given` a `{name: array}` dict: a subclass override of the old signature breaks.
- New attributes: `_producer`, `_needs_bind`, `_superseded` (a `bool`), `_reset_mask`,
  `_gen_inputs`, `_components_view`, `_zero_mask` (an array with `prepare=False`, else
  `None`). New methods: `_setup`, `_init_from_config`, `_set_legacy_attributes`, `_ctx`,
  `_ready`, `_check_owner`, `_check_mask` (and the classmethod `StepResult._of`). A
  subclass that defines a method or an attribute of one of these names now shadows the
  library's (`__init__` calls `self._setup(...)`, `step()` calls `self._ctx(stream)`):
  rename yours.

**17. The bezier `style_sampling` `DeprecationWarning` text**: its `Course` clause
("Course (until it accepts a generator, PR15): keep style_sampling, or draw per-env sizes
yourself ...") now reads "Course: with style_sampling=False,
Course.track(legacy.to_pipeline_spec(cfg).with_randomize({'generator.size': (a / scale,
b / scale)}), checkpoint_spacing=...) (track_gen.pipeline.legacy), or draw per-env sizes
yourself ...". Filters that match the warning's category or its first sentence are
unaffected.

**18. One course per generator.** Building a `Course` over a generator that a live course
drives (the same object: producers are tracked by identity, never by `==` / `hash`)
takes it over once the construction completes: the older course raises `RuntimeError`
at its next `step`, `reset`, `regenerate`, `enqueue_generate` or `generate` (its
components would read a source the newer course regenerates). A construction that raises
(a warning turned into an error by `-W error` included) takes nothing over. A config
course builds its own generator, so only `Course(old.generator, ...)` reaches this.
Rebuilding a course over the same generator (`c = Course(gen, ...)` run twice) works.

**19. Moving to the presets (optional; `CourseConfig` stays through 0.x, not
deprecated).** `Course.track` / `Course.gates` build the components a config course
builds (`docs/utilities/composing-a-course.rst`):

| `CourseConfig` | `Course.track` / `Course.gates` |
|---|---|
| `mode="track"` / `"gates"` | the preset |
| `gen` | `source` (a config, a `PipelineSpec`, or a built `TrackGenerator` / `GateGenerator` / `Program`) |
| `seeds` (int / array) | `seeds=` (int / array). `course.rng` is `None` unless `seeds=PerEnvSeededRNG(...)`: read `course.seeds.seeds` for logging. With a bound array and no rng, `generate(seeds=)` raises `RuntimeError` (the library never writes a bound seed array): pass `seeds=PerEnvSeededRNG(seeds=arr, ...)` to keep the 0.x write-in-place |
| `checkpoint_spacing`, `max_checkpoints` | the same names (track) |
| `collision` (default `None`) | `collision=` (default `None`, not `"segments"`): `"segments"` / `"sdf"` / a component named `"collision"` |
| `sdf_resolution`, `max_boxes` (track) | `collision=tools.BandCollision(method, max_boxes=k[, sdf_resolution=R])`, `method` `"segments"` or `"sdf"` (the string form `collision="segments"` means `max_boxes=1`) |
| `heightfield_resolution` | `extra=[tools.HeightFieldBake(R)]` |
| `post_radius`, `max_boxes` (gates) | `collision=r`, or `collision=tools.DiscCollision(r, max_boxes=k)` |
| `frame_collision`, `agent_radius`, `frame_thickness`, `frame_depth` | `collision=tools.FrameCollision(agent_radius, frame_thickness, frame_depth)` |
| `samples_per_gate`, `localize_window` | the same names (gates) |
| `bind(position, orientation, half_extents, box_position)` | the same call; names no component declares are ignored. The errors are the components': a missing `orientation` / `half_extents` of a box or disc collision raises `TypeError("collision: missing inputs [...]")` (a config course: the 0.x `RuntimeError`), and `None` means "not given" (`bind(None)` raises `TypeError("progress: missing inputs ['position']")`, a config course: the 0.x `RuntimeError` / `ValueError`): code that catches `RuntimeError` / `ValueError` there catches `TypeError` after moving |

- `Course.gates(cfg, collision=r, seeds=PerEnvSeededRNG(seeds=s, num_envs=E,
  device=d))` is byte-identical to `Course(CourseConfig(mode="gates", gen=cfg, seeds=s,
  post_radius=r))`.
- A preset's `BandCollision` / `HeightFieldBake` read the plan's largest size
  (`size_hi="auto"`) and warn at construction when the SDF / heightfield resolution
  does not cover it; a config course never warns (Behaviour changes). The check reads
  the construction range only: a curriculum that later widens
  `params["randomize.generator.size"]`'s `hi` is not checked again, so pass the
  curriculum's final `hi` as `size_hi=`.
- **The size curriculum without `style_sampling`** (removed at 1.0, roadmap PR16): with
  `style_sampling=False` and `scale_range=(a, b)` translated as in the warning,
  `Course.track(legacy.to_pipeline_spec(cfg).with_randomize({"generator.size": (a /
  scale, b / scale)}), checkpoint_spacing=...)`.
- The clip warning of a defaulted checkpoint capacity reads `"...; pass a larger
  max_checkpoints"` on a preset (a config course keeps `"...; pass a larger
  CourseConfig.max_checkpoints"`).

**20. Construction streams under an ambient scope (CUDA).** A `Course(CourseConfig)`
built inside an ambient `LaunchContext.scope(other)`, on a stream other than Warp's
current one, now attaches its components on Warp's current stream, ordered after and
before `other` (`other` waits for it before the warm-up), as a `stream=` course does:
0.2.0.dev5 / PR14 attached on `other`, racing the components' own `wp.zeros` memsets on
Warp's current stream. The kernels and the outputs are the same; the streams of those
construction launches change (the CPU goldens cannot see streams). Every call after
construction runs on the stream it ran on before.

### Added

- **`track_gen.tools`** (public, in `track_gen.__all__`; `import track_gen` imports it and
  still loads no generator or plugin module): the component protocol of the runtime
  utilities (`docs/contributing/writing-a-component.rst`).
  - **`Component`**, a plain base class (not a `typing.Protocol`: the library duck-types
    components and never `isinstance`-checks them) with immutable defaults: the
    declaration (`name`, `source` (the read target: `None` for the course result, else
    the name of the component whose `result` it reads), `requires` / `optional`
    (protocols of `track_gen.results`, strict / read when present), `topology`,
    `stateful`, `inputs`, `after`), the attributes `layout` (set before `attach`) and
    `result`, and the lifecycle `attach(source, cc)` (cold, once: ends with one
    `on_generate(cc.ctx, None)` and loads its hot kernels' modules), `bind(**arrays)`
    (validates, routes defaults, then calls the `on_bind` hook; a routed input's error
    names the argument it came from), and the hot `on_generate(ctx, mask)`, `step(ctx)`
    and `reset(ctx, mask)`.
  - **`ComponentContext(layout, capacity=None, *, ctx=None)`**: what `attach` is given
    (`layout`, a read-only `capacity`, the cold `ctx`, `component(name)`, `names`), and
    `add(component, source)`, which checks dependencies, unique names and input
    conflicts, resolves the read target, checks it with `check_source` against
    `requires` (strict) and `optional` (present members; with `optional` alone, against
    the target's layout or, without one, the layout its present members give: `num_envs`
    from `count` or a per-env member, the stride from a per-point member, e.g. `M` on a
    `CheckpointSet`; the context's fills what they do not), refuses a producer-result
    target of another batch (`num_envs` or device: `ValueError`), sets
    `component.layout`, then attaches.
    **`ComponentContext.for_producer(producer, *, ctx=None)`** takes a `TrackGenerator`,
    a `GateGenerator`, a `Program` or a `Course` (its generator) and reads its layout and
    plan (never its arrays); a result (a `Track`) raises `TypeError` (it has no plan:
    `ComponentContext(track.layout, capacity)` for a hand-built source).
  - **The capacity key `max_loop_length`** = `1.5 * N_max * spacing + dZ`: a bound with
    measured headroom on the 3D length of the centerline and either boundary loop of
    every valid env. `dZ` from the elevation knobs' declared `bounds()`: 0 when flat,
    `K * (z_max_hi - z_min_lo)` for `uniform` / `random_walk`, the smaller of
    `2 * H * range` and `4 * A_hi * H / H_H` for `noise`, `g * 1.5 * N_max * spacing`
    with the built-in `validity.default`'s `z_valid_grade = g > 0` (a plugin check's
    field of that name does not count). Omitted (consumers need an explicit capacity)
    when no term is finite, when the generator provides its altitudes (`provides` `z`:
    `lift.elevated` with no elevation stage) and no grade is declared, when no
    `width.constant` follows the resample (another width, or a generator that provides
    `half_width`: both unbounded per point), with a plugin geometry stage after the
    constant-spacing resample, or without that resample (gate pipelines, `external`). 34.56 at the `TrackGenConfig` defaults, 38.4 for
    `presets.track(n_max=512, spacing=0.05)`.
  - **`InputSpec(dtype, slots=1, required=True, default=None, doc="")`** (frozen; `doc`
    not compared): a bound input, `[E * slots]` of `dtype`. `default` names the input
    used when this one is not given (`box_position` defaults to `position` at one box);
    an absent optional input (`required=False`) reaches `on_bind` as `None`.
  - **Ten adapters**, each wrapping one legacy utility (`adapter.utility`, an adapter
    attribute outside the component contract, every legacy name kept; `adapter.result`
    its `.result`): `BandCollision` (`CollisionChecker`),
    `DiscCollision` (`DiscChecker`; posts mode builds the gate posts and their count on
    device and passes the gates' `valid`; raw mode, `topology="any"`), `FrameCollision`
    (`FrameChecker`, `num_envs` from the layout, windowed on the progress component's
    public `next_checkpoint`), `Props` (`PropSampler`), `Checkpoints`
    (`CheckpointSampler`), `GateCheckpoints` (`CheckpointSet.from_gates`, zero copy),
    `Progress` (`ProgressTracker`, reading the component named by `source=`, default
    `"checkpoints"`; `checkpoints=` is a 0.x alias), `Localizer`
    (`TrackLocalizer`, stateful in both modes), `HeightFieldBake` (`HeightFieldBaker`)
    and `CenterlineFromGates` (`CourseLine`, the roadmap's `centerline_from_gates`: its
    `result` is the `Centerline`). `Props` / `Checkpoints` with no capacity take
    `max(3, ceil(max_loop_length / spacing))` (`ValueError` naming `max_props` /
    `max_checkpoints` without the key).
  - **The SDF / heightfield resolution check**: `BandCollision(method="sdf", size_hi=,
    reference_resolution=128)` and `HeightFieldBake(..., size_hi=,
    reference_resolution=128)` emit a `UserWarning` at `attach` when `resolution <
    ceil(reference_resolution * size_hi)`, naming the resolution that keeps the accuracy
    and its memory; a lower `reference_resolution` accepts a coarser grid; `size_hi <= 1`
    and `segments` never warn. Since PR15 the default is `size_hi="auto"` (the plan's
    bound, below); `Course(CourseConfig)` passes `size_hi=None`.
- **Protocols** `HasValidity` (`valid`) and `HasCheckpoints` (`position`, `left`,
  `right`, `tangent`, `up_half`, `count`: what `ProgressTracker` reads), frozen, in
  `track_gen.results.__all__` and re-exported by `track_gen`; `up_half` joins the member
  schema (per point, `float32`). **`check_source(..., optional=())`**: members checked
  when present (the required members' rules and messages), never required.
- **`ctx=`** (keyword-only, a `LaunchContext`) on every hot method of the utilities
  (`query`, `update`, `reset`, `sample`, `bake`, `refresh`); with `stream=` a
  `TypeError`, on another device a `ValueError`.
- **`.result`** (read-only) on every utility: the preallocated result its hot method
  returns. **`CollisionChecker.method`** (read-only). **`CheckpointSampler.max_checkpoints`**
  and **`PropSampler.max_props`** (read-only).
- **`DiscChecker(..., valid=None)`**: an `[E]` int32 validity, checked and read in place;
  an env with `valid[e] == 0` reports no hit.
- **`CheckpointSet.valid` / `.generation`**: optional trailing fields (`[E]` int32); the
  sampler owns and writes both at `sample()` (`_place_checkpoints_k`, no new launch),
  `from_gates` aliases the gates' (or leaves `None`); `clone()` copies them.
- **`StepResult.valid`** (trailing field), `course.course_line.centerline.generation`.
- **`out=` for `localize.curvature(track, window=2, ctx=None, *, out=None,
  scratch=None)` and `speed_profile(..., ctx=None, *, out=None)`**: the same kernels into
  caller arrays (contiguous `[E * N_max]` `float32` on the track's device), no
  allocation, no fill, no host sync, recordable; byte-identical to the allocating form.
  `curvature(out=)` with `window > 0` needs `scratch=` (not aliasing `out`);
  `speed_profile(out=)` needs `kappa=`. Without `out=` both allocate as before.
- **Generation-aware self-reset** in `ProgressTracker` and `TrackLocalizer`: an owned
  `seen_generation` `[E]` per env, compared with the read `generation` at the top of the
  step kernel; a changed env first gets exactly `reset(mask)`'s writes (idempotent with an
  explicit reset). The tracker reads the `CheckpointSet`'s `generation`, written by the
  sampler at `sample()`, so an `update()` between a regeneration and the resample does not
  reset against the old ring.
- **`track_gen.exec.RebindAfterCaptureWarning`** (`UserWarning`), exported by
  `track_gen.exec`.
- **`track_gen.testing.check_component(factory, *, source="track", upstream=(),
  devices=None, num_envs=8, steps=32, seeds=None, drive=None, modes=CAPTURE_MODES)`**:
  the conformance kit for components (`construct`, `requires.complete`, `topology`,
  `result.stable`, `determinism`, `statelessness` / `reset.exact`, `regenerate.masked`,
  `regenerate.self_reset`, `invalid.sentinel`, `capacity.clamp`, `source.view`, the
  `hot_path.*` checks, and on CUDA `capture.caller` / `capture.self_reset`), with
  `CHECKS["component"]` and the pytest fixture `track_gen_check_component(factory,
  **kw)`. Every `wp.array` reachable from the result (dataclasses, mappings, named and
  plain tuples, lists, `__slots__` and attributes, recursively) is found, owned and
  aliased ones told apart by pointer; only owned arrays are poisoned and compared.
  `construct` also checks the attach contract (a chain attached without a warm-up steps
  like a warmed one) and refuses a source with no valid env (every check would be
  vacuous); the `hot_path.*` guard covers a chain's FIRST hot calls (a throwaway chain
  loads the modules); a component that declares no read is attached to an empty read
  target (reading one fails `requires.complete`); `regenerate.masked` and
  `invalid.sentinel` run for every component, `regenerate.self_reset` for every stateful
  one whose read target carries `generation` (`optional` declarations included)
  (`docs/contributing/testing-a-plugin.rst`).
- **`Course(generator, components=(), *, stream=None, prepare=True)`** (roadmap PR15;
  `docs/utilities/composing-a-course.rst`): a course over any producer, a
  `TrackGenerator`, a `GateGenerator`, a built `Program` (one built with
  `prepare=False` is prepared first) or any object with the `Producer` surface, and any
  components. A config or a `PipelineSpec` there raises `TypeError` naming the preset; a
  producer missing a required member, or a component missing one of `attach`, `bind`,
  `on_generate`, `step`, `reset`, raises `TypeError` listing them; an already-attached
  component raises `ValueError` (build a new one).
  - **Ordering**: a stable topological sort of each component's `after` and `source`
    over the given order (an ordered list, every config course's included, is kept);
    `attach`, `on_generate` and `step` run in that order, `reset` in reverse order.
    Duplicate names, an `after` / `source` naming no component of the course and cycles
    raise `ValueError` naming them; a name that is not a non-empty `str`, `TypeError`.
  - **`course.components`**: a read-only `name -> component` mapping in execution order.
    The 0.x attributes exist on every course: `progress`, `collision`, `heightfield`,
    `localizer`: the `utility` of the component of that name; `checkpoint_sampler`: of
    `checkpoints` (`course.checkpoints` is that component's `result`); `course_line`: of
    `centerline`; `None` when absent.
  - **`bind(**arrays)` routed by declared inputs**: one array to every component that
    declares its name; `InputSpec.default` routes (`box_position` from `position`), an
    explicit name beating a default route; every route validated before anything is
    rebound, and the components already rebound and the refusing one restored (with
    their rebind-after-capture probes) when an `on_bind` refuses; the
    four 0.x names accepted on every course and ignored when undeclared; any other
    unknown name a `TypeError`. A name declared by a component and by the generator
    raises `ValueError` at construction. `step()` raises `RuntimeError("call bind()
    before step()")` while a required input is unbound.
  - **The regeneration body over every component**: `producer.enqueue(ctx, advance,
    mask)`, then `on_generate(ctx, mask)` for each component, then `reset(ctx, mask)` in
    reverse order; with a mask, `any_set(mask)` first and the rest under
    `ctx.skip_unless`, so an all-zero mask is skipped on device where the capture allows
    conditional nodes and is an exact identity everywhere else. Self-graphed as ONE graph
    per `(advance, mask array)` key (the 0.x keys) or recorded into the caller's capture.
  - **`StepResult` entries**: the result of every component that overrides
    `Component.step` (per-step work), by name; per-generation components (checkpoints,
    course line, bakes, props) through `course.components[name].result`. The 0.x fields
    are the entries named `progress` / `collision` / `localizer` (`None` when absent).
    `clone()` deep-copies every entry (a `wp.array`, an object with `clone()`, a
    dataclass's `wp.array` fields; `TypeError` naming the component otherwise).
  - **`prepare=False`** skips the construction warm-up: the first regeneration call
    outside a capture prepares, one inside a capture raises `CaptureError`, and
    **`course.prepare()`** prepares explicitly (a no-op once prepared). The warm-up of an
    unprepared course runs the masked body on an all-zero private mask, so it changes no
    env's course or state, after steps too.
  - **A default stream**: `stream=s`, else the producer's `default_stream` (a generator
    built with `stream=s`). Construction attaches the components on Warp's current
    stream (where their allocations happen) after `s`'s work, then orders `s` after it.
    A call without `stream=` while another stream is capturing (Warp's current stream,
    or torch's in this thread; the library's own self-captures excluded) raises
    `CaptureError` instead of running eagerly on `s`, outside the caller's graph: record
    such a course with `stream=`.
  - **`generate(seeds=)`** follows the seed owner's rule: through the course's rng when
    it has one (the 0.x path, launch for launch); `SeedState.set` on seeds the generator
    owns (a copy and zeroed epochs: the courses of a fresh course with those seeds);
    `RuntimeError` for seeds bound to a caller array with no rng (with the
    `PerEnvSeededRNG` recipe); `TypeError` for a producer without `SeedState`. The
    refusals come before any launch.
- **Presets** `Course.track(source, *, checkpoint_spacing, collision=None,
  max_checkpoints=None, extra=(), seeds=None, stream=None, prepare=True)` and
  `Course.gates(source, *, collision=None, samples_per_gate=8, localize_window=None,
  extra=(), seeds=None, stream=None, prepare=True)`, and **`Course.from_config(config)`**
  (`== Course(config)`). `source`: a config or a `PipelineSpec` (built by the preset
  with `seeds=` / `stream=`), or a built `TrackGenerator` / `GateGenerator` / `Program`
  of the family (`seeds=` / `stream=` with one: `TypeError`; the other family:
  `TypeError` naming the other preset). Components: track `checkpoints`
  (`tools.Checkpoints(spacing, max_checkpoints=M or "auto")`), `progress`, [`collision`];
  gates `checkpoints` (`GateCheckpoints`), `centerline` (`CenterlineFromGates`),
  `progress`, [`collision`], `localizer` (`Localizer(warm_window=localize_window,
  source="centerline")`); then `extra`. `collision=`: track `"segments"` / `"sdf"` (a
  `BandCollision`) or a component; gates a post radius (a `DiscCollision`) or a component
  (`tools.FrameCollision(...)`); a component there must be named `"collision"`
  (`ValueError`). Host checks before any generator is built: `checkpoint_spacing > 0`,
  the `collision` value, a post radius `> 0`, `max_checkpoints >= 3`, the family, `extra`
  sorted with the preset's own names, and in gates `samples_per_gate >= 2`,
  `localize_window >= 1` and the construction `gate_frames.gate_width > 0` (read from
  the config, the spec or the built generator; when a randomizer draws the width, its
  construction `lo > 0` instead);
  `gates` then narrows `gate_width` to `> 0` as a config course does, on a built
  generator too.
- **`course.inputs["reset_mask"]`**: a stable `[E]` int32 mask for `reset(mask)`, next to
  `"regen_mask"`; **`reset(mask=None)`** resets every env.
- **`track_gen.pipeline.Producer`** (in `track_gen.pipeline.__all__`): a documented
  `typing.Protocol` (never `isinstance`-checked) of what a course reads from its
  generator: required `result`,
  `layout`, `enqueue(ctx, *, advance=False, mask=None)` (masked: unmasked rows bit for
  bit, an exact no-op on an all-zero mask, runnable inside a CUDA conditional-node body);
  optional `seeds`, `capture_class`, `prepared` / `prepare()`, `default_stream`,
  `capacity`, `bounds`, `inputs` / `input_info`. `Program` satisfies it.
- **`Program.prepared`** and **`Program.default_stream`** (read-only), and a read-only
  **`rng`** on `TrackGenerator` / `GateGenerator` (the `PerEnvSeededRNG` it was built
  with, else `None`), so a course reads no private member of a producer.
- **`ComponentContext.bounds`** (`ComponentContext(layout, capacity=None, *, ctx=None,
  bounds=None)`): a read-only mapping of the plan's declared bounds, separate
  from `capacity` (which the kit's `capacity.clamp` scales). One key, `size_hi`: a
  `randomize` stage's construction `hi` on `generator.size`, else the generator's
  construction `size`. `for_producer` fills it, and reads a producer's own `capacity` /
  `bounds` members before its plan.
- **`tools.Checkpoints(max_checkpoints="auto")`**: the plan bound, else the readback of
  the batch it attaches to (once, at attach); `defaulted` tells how a capacity the
  component chose itself was chosen (`None` for an explicit one). The presets and
  `Course(CourseConfig)` use it, and the course's clip warning applies to every
  checkpoint component whose capacity it chose (`None` or `"auto"`), whatever the front
  door. `None` keeps its PR14 meaning (the plan bound, else `ValueError`).
- **`size_hi="auto"`** (the new default of `tools.BandCollision` and
  `tools.HeightFieldBake`): `cc.bounds["size_hi"]`, so a preset or composed course with
  `collision="sdf"`, `collision=tools.BandCollision("sdf", ...)` or
  `extra=[tools.HeightFieldBake(R)]` warns at construction when `R <
  ceil(reference_resolution * size_hi)`. `Course(CourseConfig)` passes `size_hi=None`.

### Behaviour changes

Beyond the migration items above (which apply to the utilities used directly too):

- **`Course` builds its sub-tools as `track_gen.tools` adapters** (`course._components`,
  private) and drives them through one lifecycle: `on_generate` and `step` in list order,
  resets in the 0.2.0.dev5 order (localizer, then progress). Every public and tested
  attribute (`collision`, `checkpoints`, `checkpoint_sampler`, `progress`, `localizer`,
  `course_line`, `heightfield`) keeps its type and object; the launch order after
  construction is unchanged. Construction adds cold work before the warm-up
  regeneration: the first heightfield bake and course-line refresh at attach, and the
  module loads of the step kernels.
- **A default-capacity track `Course` at flat elevation no longer raises** when its
  construction batch has no valid env (0.2.0.dev5: `ValueError: max_checkpoints=None needs
  at least one valid env in the bound track batch`; e.g. `hull`, `E = 1`, seeds 13 or
  22): the capacity no longer comes from the batch, so it constructs (`M` from the plan
  bound) and the invalid envs read the sentinels (Migration item 1). The readback
  fallback (an elevation without declared bounds) still raises.
- **`check_source`'s missing-member message names the optional members apart**: a
  consumer that reads optional members reports `needs a HasBand source (center, outer,
  inner, count; optional: valid): ...` (0.2.0.dev5 listed every member of every protocol
  as if required, e.g. `(center, outer, inner, count, valid)`). Only the text changes.
- **The legacy constructors keep their sizing**: `CheckpointSampler(track, spacing)` /
  `PropSampler(track, spacing)` with `max_* = None` still read the bound batch back once
  at construction (a supported legacy convenience, not deprecated). The plan bound is the
  default of the components and of `Course`.
- **Optional members read once**: `valid` (now also read by `CollisionChecker`,
  `TrackLocalizer`, `curvature`, `speed_profile`, `CheckpointSet.from_gates`) and
  `generation` (`TrackLocalizer`, `CheckpointSampler`, `CourseLine`, `from_gates`), and the
  set's `valid` / `generation` (`ProgressTracker`), are checked at construction when
  present and bound like the required members; absent, every env is valid and nothing
  self-resets.
- **`generation` is the epoch** for the self-reset: an advancing regeneration changes it
  for the envs it advances; a regeneration without `advance` does not; a reseed changes it
  only when it zeroes a non-zero epoch (`Course.generate(seeds=)`, `SeedState.set`), not
  on a bare `rng.set_seeds_warp`; `reset_epoch()` followed by the same number of advances
  between two updates leaves it equal. Callers that reseed or roll epochs back must still
  `reset(mask)` (`Course` does).
- **`ProgressTracker` re-arms its previous position on invalid envs** (NaN), so an env
  that becomes valid again without a generation change (an un-advanced regeneration, a
  reseed at epoch 0, a `copy_from` of a valid snapshot) cannot emit a spurious crossing.
- **The bezier `style_sampling` `DeprecationWarning`** drops its clause "size Course's
  max_checkpoints for the largest size: ceil(1.5 * N_max * spacing / checkpoint_spacing),
  see CourseConfig.max_checkpoints" (now the default at flat elevation).
- **A deliberate exception to the hot-path rules**: the hot methods of the five utilities
  with bound inputs (`CollisionChecker`, `DiscChecker`, `FrameChecker`,
  `ProgressTracker`, `TrackLocalizer`) probe, until it is set once, whether their stream
  (or a CPU capture) is capturing, and set a host flag. It changes no launch (a kit check
  compares the traces with the flag set and unset).

- **`Course` is the composition root** (PR15): every front door ends in one setup over a
  producer and an ordered component list, `on_generate` / `step` in execution order and
  the resets over every component in reverse order (stateless components emit nothing,
  so a config course resets localizer, then progress, as before: the 0.2.0.dev5 order of
  PR14 is now this rule rather than a per-mode list). The regeneration body calls the
  producer's `enqueue` (`program.enqueue` for a facade, which is what the facade's
  `_enqueue` called).
- **The component contract in a course** (`docs/contributing/writing-a-component.rst`,
  "Contract in a Course"): (1) `on_generate` and `reset` read and write no bound input
  (the course's self-graph runs them keyed on `(advance, mask)` only; read inputs in
  `step`; a respawn snapshot of an input belongs in `step`, armed by `reset`); (2) they
  run inside a CUDA conditional-node body (the masked regeneration, in the course's own
  graph and in Warp-registered caller captures), so they may emit only kernels,
  `ctx.fill` and device-to-device `ctx.copy` on `ctx`'s stream (no copy to or from
  pageable host memory, no event, no other stream; CUDA accepts a copy from pinned,
  device-mapped host memory, measured on a GB10, but keep the body to device memory for
  portability); (3) `reset` reads only its own state and the mask (resets run in reverse
  order). A component that overrides `step` is a `StepResult` entry. Every built-in
  adapter complies.
- **`check_component` enforces the first two** (still 20 checks, the same names):
  `hot_path.buffers` traces `on_generate(ctx, None)`, `on_generate(ctx, m)` and
  `reset(ctx, m)` separately and fails one that references a bound input's pointer
  (also through an upstream result aliasing it); `capture.caller` and
  `capture.self_reset` record the chain as a course does (`any_set(ctx, done, flag)`,
  then `ctx.skip_unless(flag, [producer.enqueue(advance=True, mask=done), chain
  on_generate, chain reset])`, then `chain step`) in every mode, with an all-zero `done`
  every fourth step so the skipped branch replays (the owned arrays are poisoned in both
  runs before every step, as in PR14, and on such a step the eager twin skips the
  masked part exactly when the capture recorded it under a conditional node, so an
  array that only `on_generate` writes keeps the poison in both runs). A PR14 component
  that read an input in `on_generate` / `reset`, or used a pageable-host copy there, now
  fails them.
- **A refused conditional body** (Warp's check of a `capture_if` body) is re-raised by
  the course as `CaptureError` naming its components, with the
  `track_gen.exec.unconditional()` escape hatch.
- **`BandCollision` / `HeightFieldBake` default to `size_hi="auto"`** (PR14 components,
  unreleased): attached through `ComponentContext.for_producer` they read the plan's
  `size_hi`, so a coarse SDF / heightfield grid under a `generator.size` randomizer (or
  a construction `size > 1`) now warns at attach; `size_hi=None` disables the check.
- **`ComponentContext`'s `ctx` rule**: attach on the stream where allocations happen
  (Warp's current stream), after the producer's work; a context on another stream races
  the attach's own `wp.zeros` / `wp.full`. A `Course` with a default stream does this
  itself (docstring and guide; the PR14 text said "or pass the producer's stream").
- **Construction order on CUDA**: an open capture is refused before the producer's own
  `prepare()` runs (an unprepared `Program` or third-party producer launches nothing
  into the caller's capture); a course whose stream differs from its producer's
  `default_stream` (`Course(gen, comps, stream=s1)` over `TrackGenerator(...,
  stream=s2)`) makes `s1` wait for `s2` before the attach phase and the warm-up, so
  producer work still pending on `s2` (`gen.generate(sync=False)`) comes first; a
  `Course(CourseConfig)` built inside an ambient `LaunchContext.scope(other)` on another
  stream than Warp's current one attaches its components on Warp's current stream,
  ordered after and before `other`, as a `stream=` course does (d58ac55 attached on
  `other`, racing the components' own `wp.zeros` memsets on the current stream): the
  outputs are unchanged, the streams of that construction's launches change (migration
  item 20).
- **The producer surface is read strictly**: a facade's `program` is unwrapped only
  when it is a `Program`, and its `rng` is used only when it is a `PerEnvSeededRNG`, so
  a third-party producer's unrelated `program` / `rng` attributes are ignored
  (`course.rng` is then `None`). `generate(seeds=)` raises `TypeError` before any launch
  for a producer without `SeedState`, also when it has an rng.
- **Preset arguments get clear errors before anything is built**: `Course.gates(...,
  collision="posts")` / `collision=True` and `Course.track(..., collision=True)` (or a
  number) raise `ValueError("collision must be ...")`; `checkpoint_spacing` must be a
  number (`None` or a string: `ValueError`, not a bare `float()` error);
  `max_checkpoints="auto"` is accepted as `None`, another string or a `bool` raises
  `ValueError`; a single component as `extra` raises `TypeError` (as
  `Course(generator, component)` does).
- **`Course.gates` with a randomized bar width**: when a randomizer draws
  `gate_frames.gate_width` (`spec.with_randomize({"gate_frames.gate_width": (lo,
  hi)})`), its construction `lo` must be `> 0` and the construction width is not
  checked (`presets.gates(gate_width=0.0)` with such a randomizer builds; a `lo` of 0,
  whose draws could give never-passable gates, is refused).
- **Defaults kept against the roadmap sketch**: `regenerate(advance=False)` (the sketch's
  `advance=True` would change every `regenerate()` of every 0.x course) and `sync=True`;
  `Course.track`'s `collision=None` (as `CourseConfig.collision`; the sketch said
  `"segments"`, which would add a collision checker the config did not have and make
  `bind(position)` raise).
- **The parameter explorer and the benchmarks use the public API only**
  (`viz/param_explorer.py`, `benchmarks/*.py`): imports from `track_gen`, `track_gen.ops`
  and `track_gen.exec` instead of `track_gen._src`; the explorer's thickness statistic
  through `ops.thickness` (the same kernel and arguments) and its gate-center target as
  `2 * gate_radius`; the collision benchmark's SDF cell size from public data (the
  documented auto padding rule on `track.outer` / `track.count`, a float64 recomputation
  of the same box). Their outputs do not change beyond float rounding of that metric.
  The other `viz/` scripts (`make_report`, `plot_ablations`, `plot_tracks`,
  `plot_tracks_3d`, `plot_z_profiles`, `plot_gate_courses`) import the generators,
  configs, `Track` and `PerEnvSeededRNG` from `track_gen` too; `plot_gate_courses` keeps
  `track_gen._src.course_line` for `CourseLine`, which has no public path.

### Changed (outputs)

- Generator outputs: none. Fingerprints, the PR10a / PR11 / PR13a parity suites and
  goldens, the PR9 boundary bytes and every launch sequence of every pipeline are
  byte-identical.
- `Course(CourseConfig)` (PR15): none. Byte-identical state after every step and
  identical launch traces against the PR14 commit (`tests/goldens/pr15_course.json`, ten
  track and gates variants: construction, a bind and 40 steps with every 0.x call
  interleaved; the launch digests are checked on every platform, the state digests on
  platforms with an entry). Construction launch streams under an ambient scope excepted:
  migration item 20.
- Utilities, valid envs: none. Every utility's outputs for a valid env are byte-identical
  to 0.2.0.dev5 at the same capacities (CPU goldens taken at the 0.2.0.dev5 commit, and
  the twelve edited kernels' 0.2.0.dev5 copies launched next to the new ones on the same
  inputs: on CPU in `tests/test_components.py`, on `cuda:0` in
  `tests/test_components_cuda.py`). A default `Course`'s checkpoint arrays change shape (Migration
  item 3); their real rows do not, except for rings the old capacity clipped.
- Utilities, invalid envs: the sentinel table (Migration item 1).

### Documentation (roadmap PR15b)

The site is published from `main` when `v2` merges; there are no versioned docs.

- **Structure.** Nine sections (Getting started, Upgrading, Generating tracks and gates;
  Courses, simulators and RL; Extending TrackGen, Reference, Internals, Development,
  Background) and a landing page with four reading orders (training an RL agent,
  building datasets, writing a plugin, upgrading from 0.1). New pages:
  `getting-started/concepts` (the data flow and a glossary of 23 terms),
  `upgrading/from-0.1`, `user-guide/generating`, `user-guide/curricula`,
  `user-guide/rl-auto-reset`, `contributing/writing-a-gate-generator`,
  `contributing/packaging-a-plugin`, `contributing/plugin-abis` ("GeneratorSpec and the
  0.1 plugin ABIs", with a status table), `how-it-works/gate-pipeline`, and one
  reference page per area (`reference/results`, `pipeline`, `plugin`, `ops`, `runtime`,
  `exec`, `testing`; `reference/api` keeps the module map, the top-level names, an
  errors-and-warnings table and the environment variables). The pipeline preview page is
  promoted: its user-facing parts moved into the user guide and the rest is
  `how-it-works/pipeline-stages`; `how-it-works/results` is `user-guide/results`. Every
  page URL published from `main` (43) keeps its path, and every section label of the
  previous site is kept; deep links into `reference/api.html#...` move to the
  per-module pages.
- **Content.** Presets first everywhere (`presets.track` / `presets.gates`,
  `Course.track` / `Course.gates`), results read with `.numpy()` (torch in interop
  sections), every RL loop gated on `valid`. New homes: "Invalid envs"
  (`user-guide/results.html#invalid-envs`: fail bits, causes, the knob that lowers each,
  `keep` vs `poison`, measured rates), the bar width vs invalid fraction table
  (`tutorials/gate-sequences`, re-measured by a checked example), the rules for
  recording into your graph and a capture troubleshooting table
  (`tutorials/cuda-graph-in-a-sim`), RL auto-reset (the two masks, the host and
  in-graph recipes, cost), collision backends (choosing, CPU accuracy, cost), the
  built-in stage catalogue, and the curricula page (dynamic knobs, `randomize=`, live
  params, size, a course curriculum). No roadmap ids, `0.2.0.devN` versions or
  "preview" notes outside `upgrading/*`; the per-release "as shipped" notes are folded
  into the pages as current behaviour.
- **Migration guide** (`upgrading/from-0.1`): an upgrade procedure, a triage table
  (silent changes first, with the patterns to grep your code for), 25 items in four
  groups (12 silent changes, 7 breaking, 4 deprecations, 2 optional) each with who it
  affects and before / after code, the smaller changes, a table for users of the
  0.2.0.devN previews (with the spec `/1` -> `/2` section), what 1.0 removes and what did
  not change. Each of the 212 migration-relevant bullets of the 0.2.0.dev0 to dev6
  entries was classified (audience, kind) and placed; the 0.1 behaviour it states was
  measured on `v0.1.0` and on this tree, on CPU.
- **Checked examples** (`tests/_doc_snippets.py`, `tests/test_doc_snippets.py`, in the
  CPU lane): every `code-block:: python` under `docs/`, every python fence of
  `README.md` and the examples in the public docstrings run, the `"cuda"` literals
  rewritten to `"cpu"` inside `wp.ScopedDevice("cpu")`. Metadata is an RST comment
  (`.. snippet: group= requires= register= setup after cuda slow skip=<reason>
  expect-warning= expect-error=`); `cuda` groups run only in the GPU lane; a
  `DeprecationWarning` / `FutureWarning` attributed to an example fails it (also in a
  block that expects another warning; `expect-warning` needs that category attributed to
  the example, and `expect-*` refuse the root classes); a group is stopped after 120 s
  (a `BaseException`, so an `except Exception` cannot swallow it), and the `linecache`
  entries it replaced are restored afterwards. A Python block in a form the extractor
  does not run (`code:: python`, `code-block:: python3` / `py`, an indented or `py`
  fence) is refused, and a skipped block that uses a CUDA-only API must be a `cuda`
  group instead. Today: 230 blocks on 45 pages (30 docs pages, the README, 14 modules'
  docstrings), 203 run and 27 skipped with a reason (12 signature listings, 9
  plugin-package layouts, 4 fragments, 2 0.1 "before" blocks); 111 groups, 7 of them
  `cuda`.
- **Build checks** (`docs/_ext/trackgen_docscheck.py`, failing `sphinx -W`): every
  public symbol documented (the 30 public modules of
  `docs/_ext/trackgen_public_modules.py`, every name of their `__all__`, member
  docstrings, members inherited from private bases; every non-underscore module
  classified public or internal; exemptions with reasons in `docs/conf.py`); no roadmap /
  version jargon outside `upgrading/*`, docstrings included; every data file present.
  Nitpicky mode is on (third-party targets ignored by pattern; two type-annotation
  targets in `docs/_tools/nitpick_allowlist.txt`, applied by the extension, which fails a
  full build on an entry that no longer silences anything). User pages also keep the
  internal terms `self-graph` and `plan bound` out (internals and `reference/exec`
  excepted); the `_modules/` source listings of `sphinx.ext.viewcode` show the library's
  comments as they are and are only counted. `docs/_tools/build_ci_env.py` runs the
  build on the checkout (its root first on `sys.path`, as `python -m sphinx` from the
  root) without pytest, torch, scipy and matplotlib, as the docs workflow does. The
  `docs` extra pins `sphinx>=8.1,<10`.
- **Links.** `.github/workflows/linkcheck.yml` (new): `sphinx -b linkcheck` weekly and on
  dispatch (strict), on pull requests touching `docs/` (informational), with a summary of
  the broken links (`docs/_tools/linkcheck_summary.py`, which marks the links expected to
  be broken until v2 merges, such as `blob/main/CHANGELOG.md`); it never gates the
  deploy. `README.md`'s links into the site are checked against the build
  (`docs/_tools/check_readme_anchors.py`) and, offline, against the sources
  (`tests/test_docs_links.py`).
- **Workflows.** `docs.yml` builds with `-E`, checks that the internal pages stay out of
  the site and the README anchors, uploads the Pages artifact on pushes only, grants
  `pages` / `id-token` to the deploy job only, and cancels superseded pull-request runs
  only (a push to main runs to its deploy). `tests.yml`: `workflow_dispatch`, and
  `timeout-minutes: 30` on the CPU job (the examples ride its existing `-m "not cuda and not slow and not benchmark"`
  run, their `cuda` groups the GPU lane's `-m cuda`).
- **Internal pages** stay in the repository and out of the site: `exclude_patterns`
  gains `library-readiness/**`, `code-review-*.md` and the tooling directories
  (`_ext`, `_tools`, `_data`); nothing was moved or deleted.
- **Figures and tables.** Every generated number the docs show lives in `docs/_data/`
  (pulled in with `csv-table :file:` / `literalinclude` / `include`, never retyped; the
  command of each file is in `docs/_data/README.md` and
  `contributing/rendering-assets`). Re-measured on CPU: the generator metrics
  (`generators/benchmarks`) and the collision accuracy. Re-rendered on CPU:
  `docs/assets/generator-repulsive.png` and `docs/assets/readme-generator-grid.png`, the
  only CPU figures whose pixels change with the code. Deviation from the roadmap: the
  README's figure (`readme-generator-strip.png`) is not stale (main's and this tree's
  code render it pixel-identically on one machine), so it is kept; the figures that
  differ from a fresh render on arm64 differ for `v0.1.0`'s code too (platform), and are
  kept. The GPU tables were measured on an NVIDIA GB10 (`docs/_data/machine-cuda.txt`).
- **Library diff: docstrings and comments only** (an `ast.dump` comparison with
  docstrings stripped matches the PR15 tree for all 51 changed `track_gen` files). The
  rendered docstrings lose their roadmap / version notes; the "Preview: names and
  signatures may change before 1.0" note of `track_gen.pipeline` is replaced by the
  deprecation-policy statement; 18 member docstrings are added (and `from_legacy` on the
  Params bases); docstring citations of moved pages point at labels
  (`testing/conformance.py`, `testing/components.py`) or at
  `docs/contributing/plugin-abis.rst` (`_src/generator_registry.py`). No message
  changes: the three library messages that send gate-generator authors to
  `docs/contributing/writing-a-generator.rst` stay valid through its "Gate generators"
  pointer section.

### Fixed

- Library-readiness findings closed by roadmap PR14: MEDIUM A34 "no common utility
  protocol; Course hard-codes the tool set" (`track_gen.tools`, `Course` drives adapters
  through one lifecycle; `Course(generator, components)` completes it in PR15); MEDIUM A38 "no
  regeneration signal" (generation-aware self-reset; the sampler's set carries the
  generation of its ring); MEDIUM A49 "utilities disagree on invalid envs" (one sentinel
  table, `StepResult.valid`, the posts' `DiscChecker` given count and `valid`); LOW A131
  "the invalid-env path is undocumented, untested, and contradicted by the tutorials"
  (the sentinel table in the docs, the RL loops gated on `valid`, forced-invalid tests);
  the rebind part of LOW A92 (`RebindAfterCaptureWarning`; the stable reset mask: PR15);
  LOW A99 "public helpers allocate per call" (`curvature` / `speed_profile` `out=`;
  `rng.sample_uniform_warp(out=)` stays open); LOW A100 "sampler capacities sized by host
  readback" (the plan's `max_loop_length`, the `Course` default, the components).
- Library-readiness findings closed by roadmap PR15: MEDIUM A33 "Course hard-codes two
  generator families and a fixed set of runtime utilities" (`Course(generator,
  components)`, the presets); MEDIUM A34 (completed); MEDIUM A35 "Course cannot be given
  a generator, RNG or result source" (any producer, the `Producer` protocol, presets
  over built generators and Programs, `seeds=` including a `PerEnvSeededRNG`); LOW A54
  "whole-batch-only regeneration" (the course level: one masked body over every
  component, skipped on device for an all-zero mask, `inputs["regen_mask"]` /
  `inputs["reset_mask"]`); LOW A86 "gate collision is hard-wired" (`Course.gates(...,
  collision=)` takes any collision component); the reset-mask part of LOW A92; LOW A93
  "CourseConfig is monolithic" (components that validate themselves, the presets;
  `CourseConfig` stays for 0.x); LOW A94 "bind() hard-codes one agent model"
  (`bind(**arrays)` routed by declared `InputSpec`s); the private-module reaches of LOW
  A108 / A111 (the explorer and the benchmarks use the public API); D13 / G6 "Course
  anchors its components" (`step()` / `reset()` anchor too).

### Deferred

- **Generator-input binding through `Course.bind`** (roadmap PR15, deferred): a dotted
  generator input name (`bind(**{"centerline.xy": a})`) raises `TypeError` with the
  recipe: write `course.generator.inputs[name]` in place (a stable array that every
  regeneration and every replay reads). Pointer binding arrives with `build(inputs=)`,
  and `bind` routes dotted names then, with no copy and no self-graph key change.
- `pipeline.MixedBatch`: not shipped; the `Producer` protocol allows it.
- `Course.describe()`; per-component invalid flags in `StepResult`; deprecating
  `CourseConfig` (roadmap PR16's decision); the default-stream capture guard on the
  generator facades.
- A typing-only `SupportsComponent` protocol for static duck typing; a component registry
  kind; fused step kernels.
- Removing the legacy readback sizing of `CheckpointSampler` / `PropSampler` (not
  planned; it would need a warning release first).
- Declared bounds on the elevation's `z_min` / `z_max` (so that elevated
  `TrackGenConfig`s get a `max_loop_length`), and a bound for variable width.
- PR15b, GPU: the GPU tables of `docs/_data/` were measured on an NVIDIA GB10 (an
  integrated-memory GPU); re-measure them on a discrete GPU when one is available (the
  commands are in `docs/_data/README.md`). Every documentation example, the `cuda`
  groups included, passes on that GPU (246 groups).
- PR15b, not done: a built copy of the 0.1 docs (the migration guide links the
  `v0.1.0` documentation sources); the two nitpick allowlist entries (type annotations
  naming private classes) go by 1.0; three planner messages still cite the release that
  introduced a rule ("since 0.2.0.dev4" / "dev5"), and the docs quote them without it.
- The migration guide links the changelog on `main`, which has none until `v2` merges:
  the link check reports it on pull requests (informational) until then.

### Testing

- `tests/test_components.py` (CPU): the base class, `ComponentContext` and `InputSpec`
  (immutable defaults, `bind` errors, default routing and optional inputs, `add`'s
  dependency, duplicate-name, input-conflict, optional-only and other-batch errors,
  read-target resolution, `for_producer`'s keys for flat, elevated, randomized, graded,
  variable-width, attribute-provider and gate plans and its `TypeError` on a `Track`,
  `import track_gen` loading no pipeline or plugin module); `check_component` for every
  built-in adapter; negative kit tests with toys that each fail exactly their check
  (first-call-only work, undeclared reads, a stateful optional-only component without
  self-reset, a named-tuple result, an attach without its closing `on_generate`); the
  `CurvatureLookahead` toy of the guide (public imports only) passing every check; byte
  identity of every utility for valid envs (CPU goldens `tests/goldens/pr14_utilities.json`
  taken at 0.2.0.dev5 by `tests/tools/make_pr14_goldens.py`, and the 0.2.0.dev5 kernels
  frozen in `tests/_pr14_base_kernels.py` launched next to the new ones); the
  adapter-driven `Course` against a hand-wired composition of the legacy utilities (8
  variants, 40 steps), `StepResult.valid`, the posts' count / valid, the course line's
  generation, the default checkpoint shapes (70 / 58 / 116 at spacing 0.5 / 0.6 / 0.3),
  real rows equal at the old capacity, the elevated readback fallback and the clip
  warning; the self-reset (masked and full advances, with and without reset, the
  `generate()` / `sample()` ordering, reseeds and the epoch ABA case); the sentinels
  (invalid == poisoned for every utility and adapter, the table values, inactive box
  slots, poison mode, no event on re-validation); capacities and the resolution warning;
  `out=` (bytes, hot-path guard on a `Track`, a `Centerline` and a source without
  `valid`, errors); `ctx=`; the rebind warning (a capturing context, a CPU capture,
  stacklevels, silence on the same arrays); the `CheckpointSet` fields; the docs' RL loop
  on a `Course` with one env forced invalid (no NaN reaches the reward or the
  observation, the env is done, one advancing masked regeneration takes it out);
  `check_source(optional=)`.
- CUDA (`tests/test_components_cuda.py`, collects on CPU): `check_component` for every
  adapter on `cuda:0` (`capture.caller` in the warp / external / torch modes,
  `capture.self_reset`); the eager-vs-captured trajectory twin in track and gates modes
  (`wp.ScopedCapture` and a registered `torch.cuda.graph`; 200 steps of `[course.step();
  course.regenerate(mask=..., advance=True)]` with host-written positions and a done
  schedule; only the step outputs are poisoned before each replay, the producer state and
  the sub-tools' buffers are compared unpoisoned); the old-vs-new kernel identity, the
  sentinels and the self-reset on `cuda:0`; `curvature(out=)` / `speed_profile(out=)`
  recorded; the rebind warning under a real capture and the probe's host cost on
  `Course.step` (printed ratio).
- Existing tests updated where PR14 required it: `tests/test_public_api.py` (`tools`,
  `HasValidity`, `HasCheckpoints` in `track_gen.__all__`); `tests/test_results.py` (the
  consumers' read tuples gain their optional members, so a malformed optional member is
  rejected at construction); `tests/test_no_host_sync.py` (the `out=` forms under the
  full guard); `tests/test_difficulty_axes.py` (the `style_sampling` warning's parts; a
  default-capacity `Course` under a size curriculum). The byte guards pass unchanged:
  `tests/test_generator_fingerprints.py` and its goldens, the PR9 boundary bytes,
  `tests/test_gate_pipeline_parity.py`, the PR13a golden and the
  `tests/test_legacy_plugin_abi.py` pins.
- PR15, CPU: `tests/test_course_composition.py`: the constructors (`Course(cfg)`,
  `Course(config=cfg)`, `from_config`: same type and bytes; every `TypeError`; a
  subclass's `__init__` for every front door); the presets (names and order,
  `Course.track` == a hand composition, `Course.gates(..., seeds=rng)` == the config
  course, built `TrackGenerator` / `Program` sources, `seeds=` / `stream=` refused with
  them, the other family refused, the `collision=` slot name, one capacity path across
  front doors incl. the elevated readback, the clip warning texts, `gate_width == 0`
  refused before any generator is built, `size_hi` warnings); ordering (sorting,
  stability, every error, attached components, `step` forward and `reset` reverse by
  `trace_launches`); `bind` (routing, defaults, the 0.x names, unknown and generator
  names, atomicity on a refused route and a refusing `on_bind`, construction-time
  conflicts, the config course's `bind()` / `bind(None)`); supersession; `StepResult`
  (entries, mapping methods, not a `Mapping`, truthiness, `iter`, fields, repr,
  `replace`, `clone`, positional construction); the body structure; the all-zero mask
  identity on every buffer; the RL graph twin recorded into `wp.ScopedCapture("cpu")`
  == the eager twin every step (track preset + Localizer + `CurvatureLookahead`, gates
  preset); the toy with public imports only (an AST check); the contract checks (a toy
  reading `position` in `on_generate` / `reset` fails exactly `hot_path.buffers`; no
  regeneration body or `reset` of the composed built-in courses references a bound
  array); a minimal producer (`result`, `layout`, `enqueue` only; `HOST` refused under a
  capture); `generate(seeds=)` on owned, rng-held, bound and missing seeds;
  `prepare=False` / `prepare()`; the hot-path guard; the CPU graph lifetime. After the
  PR15 review: a refusing `on_bind` that pinned an array first is restored and a later
  rebind after a capture still warns; a construction that raises (warnings as errors)
  supersedes nothing, and supersession follows producer identity (value-hashed and
  unhashable producers); `StepResult` fields and entries stay in sync (`copy.copy` plus
  a reassignment, Lightning's deepcopy-with-memo plus `setattr`), its repr is the
  dataclass one (`pprint`'s layout); a randomized gate width checked on its `lo`; the
  presets' argument errors; unrelated `program` / `rng` producer attributes ignored, a
  missing `SeedState` refused before any launch with an rng; an open capture refused
  before a third-party producer's `prepare()`; what each recorded call anchors (the
  binding, masks, seeds). `test_config_course_launches_match_the_pr14_goldens` (every
  platform) and `test_config_course_states_match_the_pr14_goldens` (platforms with an
  entry) recompute `tests/goldens/pr15_course.json` (written at the PR14 commit by
  `tests/tools/make_pr15_course_goldens.py`: per-call launch-trace digests with buffer
  roles, kept per platform and once more under the platform-independent key
  `"topology"`, and per-step state digests, ten variants). To add a platform's state
  digests (the Linux GPU workstation, Linux CI), run the tool there from a d58ac55 tree
  (its docstring gives the commands) and merge the new entry.
- PR15, CUDA (`tests/test_course_composition_cuda.py`, collects on CPU): the roadmap
  gate, an RL graph `[physics stub, course.step(), done kernel,
  course.regenerate(mask=course.inputs["regen_mask"], advance=True)]` replayed 200 steps
  == an eager twin in track and gates modes, under `wp.ScopedCapture`, a plain
  `torch.cuda.graph` and `interop.torch.registered(graph=g)` with and without
  `conditional=True` (the `_regen_flag` conditional counted, anchoring checked); one
  self-graph per key (a third-party component inside its conditional body); both
  lowerings of the zero-mask body; lifetime; the conditional-body contract (a component
  that records an event in `on_generate` fails `check_component`'s `capture.caller`, the
  course refuses it with a `CaptureError` naming its components, `exec.unconditional()`
  runs it; a copy from a pinned host array is accepted by CUDA, so it is no offence); a
  course with a default stream (its calls default to it; `CaptureError` without
  `stream=` under another capture, recording with it); attach ordering on a busy current
  stream, and on a producer stream still busy with an advancing generation (`wait_stream`
  spy, == a twin); a recorded `[step, reset(m), regenerate(mask=m')]` anchoring the
  binding and both masks (alive after a rebind and every caller reference dropped, freed
  with the graph); the host cost of `step()`'s anchor and default-stream probes (printed
  ratio).
- PR15 edits no existing test file: every `Course` test of 0.2.0.dev5 and PR14 passes
  unchanged on the new root. The shared toys (`tests/_component_toys.py`) gain the two
  that read a bound input in the regeneration body (each fails exactly
  `hot_path.buffers`), and the guide's `CurvatureLookahead` takes its namespaced default
  name, `"acme.lookahead"`.
- PR15b (CPU lane, no Sphinx needed): `tests/test_doc_snippets.py` runs every group of
  every page (`test_snippets_cpu[<page>::<group>]`; `test_snippets_cuda[...]` in the GPU
  lane) and checks the metadata grammar and layout on every page
  (`test_snippet_metadata`: unknown or misplaced tokens, a comment docutils would render
  as text, dangling `requires=`, an f-string containing `cuda` in a CPU group, the
  retired `page-pending` marker); its unit tests run the grammar and the runner on
  synthetic pages (`setup` / `after` / `register=` / `expect-*`, `requires=` and its
  cycles, docstring extraction, a deprecation attributed to an example failing it, the
  `wp.config` restore in a fresh interpreter, the refused block forms, a specific
  `expect-warning`, the timeout under `except Exception`, the `linecache` restore). `tests/test_docs_links.py`: the README's
  links into the site map to pages, its relative paths exist, no page links into the
  excluded directories. `tests/test_docs_tools.py` (fixtures in `tests/fixtures/docs_tools/`):
  `md_table_to_csv` on a real `compare_generators` output with Warp's banner,
  `strip_warp_log`, `check_readme_anchors`, `linkcheck_summary` (expected-broken links
  marked), the package blocker and checkout resolution of `build_ci_env` and the page
  owners of `check_labels`. `tests/test_generator_docs.py`
  reads the `Scatter` gate generator from `contributing/writing-a-gate-generator.rst`.
  The docs build (`sphinx -W`, nitpicky, the docscheck) runs in `docs.yml`.

## 0.2.0.dev5 (2026-10-24, tag `v0.2.0.dev5`)

### Added

- **The result-object surface** (roadmap PR13b). A new public module,
  **`track_gen.results`**: `Layout`, `Track`, `GateSequence`, `Centerline`, the
  capability protocols and `check_source`. `track_gen` re-exports `Track`,
  `GateSequence`, the new `Layout` and the five protocols (the same objects; also
  `track_gen.results` itself), and `track_gen.plugin.Layout` is the same `Layout`.
  `docs/how-it-works/results.rst` describes it all.
  - **`Layout(num_envs, stride, device, closed=True)`** (moved from `plugin/stage.py` to
    `_src/types.py`): `num_envs` / `stride` are coerced with `operator.index` (NumPy
    integers accepted, `bool` refused with `TypeError`, values below 1 `ValueError`),
    `device` must be a `str` and is stored as given (no device lookup at construction),
    `closed` must be a `bool`. Every library check compares layouts by their canonical form
    (the device resolved with `wp.get_device` at check time), so a hand-set
    `Layout(E, S, "cuda")` matches a pipeline's `"cuda:0"`; every layout the library makes
    stores the canonical device string.
  - **`Track.layout` / `GateSequence.layout`**, a trailing field (positional construction
    keeps working): set on every result a generator, `build`, `view`, `alloc` or `clone`
    returns, `None` on a hand-built one (a build with `out=` sets it when it is `None`).
    `Program.layout` is the pipeline's layout.
  - **`validate(need=()) -> Layout`** on `Track`, `GateSequence` and `Centerline`: every
    schema field that is not `None` must be a contiguous 1-D array of its dtype,
    `[E * stride]` per point or `[E]` per env, on the layout's device, and every attribute
    `[E * stride]` of `float32` / `int32` / `vec2f` / `vec3f`; the layout is the stored one
    or inferred from `count` and the first per-point field. It raises `ValueError` naming
    the field, the expected array and the one it got (`Track.inner: expected a contiguous
    [3072] vec3f ([E*stride] = [8*384]) array on cpu, got shape (3064,) vec3f on cpu`).
    Host only (a few microseconds), never in a constructor: `Track(...)` still accepts a
    partial result.
  - **`view(e0, e1)`**: a zero-copy env-slice view (per-point arrays `[e0*S : e1*S]`,
    per-env arrays `[e0 : e1]`, attributes alike, `fail_bits` shared, its own `layout`),
    callable anywhere, views of views included; `IndexError` outside `[0, E)`.
  - **`copy_from(src, *, stream=None, ctx=None)`**: one copy per held array, no
    allocation, no host sync, recordable; `src` must be of the same class (`TypeError`);
    layouts, held fields and attribute dtypes must match (`ValueError`), a partial overlap
    raises, a self-copy is a no-op,
    `stream=` with `ctx=` is a `TypeError`. The stream is
    `ctx`, else `stream=`, the ambient scope, the stream that produced `src` (its
    generator's default stream), torch's capture stream, Warp's current stream.
  - **`clone(*, stream=None)`** iterates the schema instead of listing the fields: it
    allocates on Warp's current stream, makes it wait on the producer's stream
    (`stream=`, else the generator's default), copies there, then waits for the copies
    on the host, and keeps `None` fields, the class, `layout` and the shared
    `fail_bits`. The clone is usable on ANY stream on return. The same bytes on the same
    stream as before when no producer stream is involved. Documented as cold (it
    allocates and blocks).
  - **`alloc(layout, *, attributes=None)`** (classmethod): every field allocated and set to
    the "no track yet" state (padding, zero counts and flags), on Warp's current stream,
    then a host wait: usable on ANY stream on return, so `snap = Track.alloc(layout)`
    then `snap.copy_from(track)` on the generator's stream needs no sync (no late fill
    lands on the snapshot, and the producer's stream never writes a block the current
    stream still uses). `attributes` maps plugin keys (`"<package>.<name>"`, else
    `ValueError`) to `wp` dtypes or their names (`"float32"`, as in
    `ExternalParams.attributes`). `CaptureError` under a capture. `copy_from` into an
    `alloc` result is the hot-path snapshot.
  - **`Centerline`**: a centerline without a road band (`center`, `tangent`, `arclen`,
    `length`, `count`, `valid`, optional `generation`, `layout`), with the same methods.
- **Caller-owned results: `out=` on the facades.** `TrackGenerator(..., out=None)` and
  `GateGenerator(..., out=None)` pass `out` to the planner, like
  `pipeline.build(out=)`: the Program's result is the caller's object, written from the
  warm-up on, at stable addresses. `out` is validated in full (messages name
  `out.<field>`) and its canonical layout must equal the pipeline's
  (`out: a Track for Layout(4, 384, 'cpu', True); this pipeline writes Layout(8, 384,
  'cpu', True)`); afterwards `out.layout` (when it was `None`), `out.fail_bits` and its
  producer stream are set.
  - **Mixed batches**: several generators write disjoint `big.view(e0, e1)` ranges of one
    result, recorded in one caller graph, each with its own mask slice; a masked
    generation of one leaves the others' rows untouched (findings A43, A120).
  - **Seeds on a view**: with `seeds=None` or an `int`, env `e` of a view starting at env
    `k` of its root result gets seed `base + k + e`, so a mixed batch is seeded like one
    batch and two halves of one generator never collide (finding A121); seed arrays,
    rngs and `SeedState`s are bound as given.
  - **Disjoint outputs**: a build whose `out=` overlaps the result of a live Program
    raises `PlanError` (`out: envs [0, 4) of this storage are already written by a live
    Program ...; to rebuild into the same storage, release the earlier generator /
    Program first (del gen)`); every build registers its result (weak references, keyed
    on the allocation of `count`) once its Program is usable (after the warm-up; a
    facade at the end of its construction), so `build(out=gen.program.result)` raises
    while `gen` is alive and a build that raised never refuses a retry. The overlap
    check runs after the capture refusal (its `gc.collect()` never runs in a capture).
  - **`out.attributes` with keys from another pipeline**, on a view of a larger result
    (a mixed batch): a key this pipeline exports is bound (as in 0.2.0.dev4); any other
    key is checked, filled once with its padding at build and never written, so
    generators that do and do not export an attribute can share its array. Binding is
    decided at the family's final write, where its export is known, so a family carried
    without being exported is padded too. On the whole of its allocation an unexported
    key still raises the 0.2.0.dev4 `ValueError` (typo protection), and an exported key
    of the wrong dtype reads `out.attributes['acme.tag'] is float32; this pipeline
    exports 'acme.tag' as int32`.
- **Capability protocols** in `track_gen.results` (`typing.Protocol`,
  `runtime_checkable`), each a capability a family of consumers shares (each consumer
  checks the subset it reads, `members=`), frozen once shipped (a new capability is
  a new protocol): **`HasCenterline`** (`center`, `arclen`, `length`, `count`),
  **`HasBand`** (`center`, `outer`, `inner`, `count`, `valid`), **`HasGates`**
  (`position`, `count`, `valid`), **`HasGateFrames`** (`HasGates` + `forward`, `left`,
  `right`, `orientation`, `half_size`) and **`HasGeneration`** (`generation`). No
  protocol has a `layout` member.
  - **`check_source(obj, protocol, *, consumer, topology="closed", members=None) ->
    Layout`**: reads the protocol's members (or the `members` a consumer reads) with
    `getattr`, once (a missing one raises `TypeError` naming them), uses
    `obj.layout` when it is a `Layout` (a TrackGen result's must be one or `None`; an
    unrelated `layout` attribute of another object is ignored) or infers it from the
    checked members only (`count`, else the first per-env member, gives `num_envs`; the
    first per-point member the stride, `1` when there is none), checks only those
    members with the `validate()` messages prefixed by the consumer, never calls the
    object's own `validate`, and refuses an open layout for a closed-only consumer.
    `protocol` may be a tuple (`(HasCenterline, HasGeneration)`: one call, one layout);
    a non-protocol raises `TypeError`, `members` outside the protocol `ValueError`.
  - Every runtime utility checks its source with it at construction, only the members it
    reads: `CollisionChecker`, `PropSampler`, `CheckpointSampler`, `HeightFieldBaker`
    (`HasBand`), `TrackLocalizer`, `localize.curvature`, `localize.speed_profile`
    (`HasCenterline`), `CourseLine` (`HasGates`), `FrameChecker` and
    `CheckpointSet.from_gates` (`HasGateFrames`). They accept any object with those
    members (a `Centerline`, a view, a `SimpleNamespace`, a proxy); their signatures are
    typed with the protocols. `FrameChecker`'s `num_envs` must equal the source's. Each
    binds the arrays it checked at construction and passes those to every call (stable
    pointers; a member reassigned on the source later is never read unchecked).
  - **`CourseLine.centerline`**: a gates-mode `Course` builds its spline as a `Centerline`
    (no NaN band any more, the same two refresh launches into the same buffers), and its
    localizer reads it.
- **`randomize.uniform`**: per-env uniform draws of another stage's dynamic `float32`
  field, inside the Program, at every generation (finding A15's mechanism; the built-in
  knobs it can draw are the dynamic built-in knobs below). **`presets.track(randomize=None)`** and
  **`presets.gates(randomize=None)`** take `{"<stage id>.<field>": (lo, hi)}` (split at
  the last dot) or a sequence of `StageRef`s, and insert
  `StageRef("randomize", "uniform", {"stage": s, "field": f, "lo": lo, "hi": hi},
  id=f"randomize.{s}.{f}")` right before stage `s`. **`RandomizeUniformParams`**:
  `stage`, `field` (static), `lo`, `hi` (dynamic, so a curriculum writes
  `params["randomize.<s>.<f>"]["hi"]` in its own graph).
  - The draw is `lo + u * (hi - lo)` keyed on the env's effective seed and the
    randomizer's salt (`b.salt("draw")` under its id), clamped to the target field's
    declared bounds: deterministic per `(seed, epoch)`, distinct across envs, redrawn by
    `advance=True` and reseeds, and exact under masks (an unmasked env redraws the same
    value).
  - Listing order is execution order: a randomizer is listed before its target, and only
    the generator's randomizers may precede the generator, which is the first stage that
    is not a randomizer (planner and legacy sites use one helper instead of
    `stages[0]`). `PlanError` for a randomizer after its target, another stage before the
    generator, an unknown or randomizer target, a static / `int32` / unknown field (the
    static message lists the pipeline's drawable fields), two randomizers on one field,
    initial bounds outside the field's.
  - The drawn field is a single unlatched array: `params[s][f]` reads the current draws,
    `set(f, ...)` raises `ValueError` naming the randomizer, `repr` marks it. The
    randomizer's `set("lo" | "hi", scalar)` refuses a value outside the drawn field's
    declared range (an array write is clamped by the kernel, documented). A plugin's
    static field gets its own refusal (`... is static (declared with plugin.static)`),
    and a `randomize=` key naming a missing stage says so (`randomize: key
    'gnerator.amp' names stage 'gnerator', ...`).
  - Two more launches per randomizer per generation (draw, `lo` / `hi` latch, minus the
    target's latch); none without one. `to_dict` / `from_dict` round-trip it (format
    `/2`, an additive kind). The catalogue gains the kind `randomize` (not open to
    plugins).
- **The parameter latch stays** (the PR10 question answered): dynamic params and
  `per="env"` inputs are still snapshotted at the start of each generation; drawn fields
  and `per="raw"` inputs are **unlatched** (read during the generation), held outside the
  latch pairs. Documented in `StageParams`, `Program.inputs` / `input_info`, the
  facades' `params` / `inputs` and the generator guide.
- **`generator="external"`: ingesting caller centerlines** (finding A42). A built-in
  `STATIC` Stage (registered through the same `register(reg)` as a plugin, public plugin
  surface only) that reads `[E * N]` `vec2f` `centerline.xy` and `[E]` `centerline.count`,
  plus `centerline.z` (`with_z`), `centerline.half_width` (`with_half_width`) and
  `centerline.<pkg>.<name>` attributes (**`ExternalParams.attributes`**: `(key, dtype,
  interp)` triples) from `program.inputs`. Each generation copies the inputs into
  `raw.*` (slots past `count[e]` repeat the last point: exact, the extra segments have
  zero length), then validates the copies (`3 <= count <= N`, finite points, finite `z`,
  finite and positive `half_width`; else fail bit 0). The initial inputs are circles of
  radius `0.6 * scale`. `seeded = False`, `preferred = {"relax": "none"}`, raw capacity
  `num_points`. It works through `presets.track`, `presets.gates(source="stations" |
  "relaxed_stations")` and `TrackGenConfig(generator="external")` (which keeps
  `relax_enable`).
  - **`track_gen.pipeline.load_centerlines(target, xy, *, count=None, z=None,
    half_width=None, attributes=None, envs=None, stream=None)`**: loads `[E, m, 2]`,
    `[E * N, 2]` or a list of `[m_e, 2]` arrays into any generator declaring the
    `centerline.*` inputs, with host checks (env count, counts, finite values, positive
    half-width, attribute keys, the kind of every value: numbers, and integers within
    `int32` for an `int32` attribute, never truncated; undeclared inputs; every declared
    per-point input given: `z=` for `with_z`, `half_width=` for `with_half_width`, each
    attribute, else the new rows would keep the previous ones) and a **sizing check**
    against the Program's spacing and `n_max` (`load_centerlines: env 3: perimeter 4012.0
    needs 66868 points at spacing 0.06 > n_max 384; ...`); uploads through pinned staging
    on the stream resolved like `generate`'s (`stream=`, an ambient `LaunchContext.scope`,
    the generator's constructor `stream=`, torch's capture stream, Warp's current
    stream), then waits; `CaptureError` under a capture.
  - **`Program.input_info`** / `gen.input_info`: a mapping from input name to a
    **`track_gen.pipeline.InputInfo`** (`name`, `dtype`, `per`, `length`, `latched`,
    `stage_id`).
- **Plugin surface for ingestion generators** (all public, invariant 4):
  **`StageBuilder.input(name, dtype, default=0, *, per="env")`**, where `per="raw"`
  (centerline generators) gives an unlatched `[E * raw]` input of `float32` / `int32` /
  `vec2f` / `vec3f`; **`Stage.declare(params) -> (writes, provides)`** (default: the class
  attributes; read for centerline generators only in 0.x, refused at registration on
  other kinds, validated at bind with the generator port rules, and routed through every
  site that read `writes` / `provides`: binding, the raw attribute families, the
  adapter's writes, the presets, `legacy`, `describe` at the default Params);
  **`Stage.seeded`** (`True` by default; `describe()` reports it for every generator);
  **`Stage.preferred`** opened to centerline generators with the key `"relax"`.
  **`presets.track(relax="auto")` / `presets.gates(relax="auto")`** is the new default:
  the generator's `preferred["relax"]`, else `"xpbd"` (unchanged for every built-in).
- **Topology** (`closed`): `BatchSpec.closed` must be a `bool` (`PlanError`), and
  `closed=False` is refused right after the spec's shape checks, before any generator is
  bound or `Params.validate` runs, from the stage classes, naming the stages whose
  `Stage.topology` is `"closed"` (every built-in; a legacy `GeneratorSpec` counts as
  closed). The utilities that read a result declare `topology = "closed"` and
  `check_source` refuses an open layout. A host-side contract: no kernel takes the flag
  (see Deferred).
- **Conformance kit**: `output.shape` also checks the result's canonical layout against
  `program.layout` and `result.validate()`; new check **`output.view`** (every kind,
  every device): the pipeline built with `out=` the middle view of a byte-poisoned
  `alloc` of `E + 2` envs equals the standalone run in the view's rows, after
  construction and a masked advance, and every byte outside the view keeps its poison;
  `check_stage` and `check_gate_generator` run it too. `diversity` reports *skipped*
  for a generator declaring `seeded=False`.
- Docs: new `docs/how-it-works/results.rst` (data model, `Layout`, schemas, `validate`,
  views and copies with their stream rules, view safety, `out=` and mixed batches,
  protocols, `check_source`, topology); the utilities pages ("What a utility reads",
  `CourseLine.centerline`); `docs/contributing/writing-a-generator.rst` "Ingesting your
  own centerlines" and "Writing an ingestion generator"; `writing-a-stage.rst` (view
  safety, topology, input names, per-env draws); `testing-a-plugin.rst` (`output.view`,
  `seeded=False`); the configuration reference (`relax="auto"`, `randomize=`,
  `ExternalParams`) and tuning guide (what `randomize=` does not do yet);
  `pipeline-preview.rst` ("Per-env draws", the latch decision, `input_info`, `out=`);
  a mixed-batch example in `docs/tutorials/cuda-graph-in-a-sim.rst`; the API reference;
  `docs/library-readiness/target-architecture.md` (as-shipped notes, roadmap PR13c).

- **Dynamic built-in knobs** (roadmap PR13c; findings A15 for the built-in knobs and
  A88). The knobs that matter for curriculum and domain randomization are DYNAMIC
  `float32` fields (per-env `[E]` arrays with declared bounds), so `randomize=` and
  caller-graph curricula reach them: `presets.track(generator="polar",
  randomize={"generator.polar_radial_jitter": (0.3, 0.85)})`,
  `gen.params["elevation"].set("z_max", per_env)`, or a kernel in the caller's graph
  writing `gen.params["generator"]["rad"]`.
  - **Generators** (the randomize key is `"generator.<field>"`): `bezier` `rad`
    `[0, 1]`, `handle_clamp_frac` `>= 0`; `hull` `hull_displacement` `[0, 1]`; `polar`
    `polar_radial_jitter` `[0, 0.85]`, `polar_angular_jitter` `[0, 0.45]`; `voronoi`
    `voronoi_radial_variation` `[0, 0.85]`, `voronoi_angular_jitter` `[0, pi]`;
    `checkpoint` `checkpoint_radius_min_frac` `[0, 1]`, `checkpoint_angle_jitter`
    `[0, 1]`, `checkpoint_turn_rate` `[0, pi]`. The polar, voronoi and checkpoint
    **gate** generators share the ring knobs (same kernels; `checkpoint_turn_rate` is
    track-only). Each generator page lists its knobs with their bounds, the direction
    that makes tracks harder, the effective range and the measured yield at both bounds
    (e.g. `rad` has no effect above `handle_clamp_frac`; `voronoi_angular_jitter` is
    legal up to `pi` but useful to about `0.5`).
  - **Elevation** (key `"elevation.<field>"`, tracks, gates and the centerline before
    the stations alike): `z_min`, `z_max` (`uniform`, `random_walk`, `noise`; unbounded),
    `z_max_step` (`random_walk`, `>= 0`), `z_noise_amplitude` (`noise`, `>= 0`).
  - **Static, deliberately** (open question 5 answered for the built-ins): every count,
    size, enum or Python branch (`max_num_points`, `polar_num_knots`, `checkpoint_count`,
    `voronoi_site_layout`, ...; `bezier` / `hull`'s `min_num_points` became a dynamic
    `int32` in PR13d, below), `half_width`, `spacing`, `edgy` (its host `atan`
    transform is not reproducible in a kernel bit for bit), the core `scale` (PR13d adds
    the per-env `size` factor), `z_base`,
    `z_noise_harmonics`, `z_control_points`, `checkpoint_steer_gain` and
    `checkpoint_lookahead_frac` (path-tracer internals), the relax / validity /
    gate-geometry knobs, the width stages' knobs (except `width.from_channel`'s `scale` /
    `lo` / `hi`, dynamic since 0.2.0.dev4) and every `repulsive` growth knob. Naming one
    in `randomize=` raises
    `PlanError` (`generator.edgy is a static field (build a new generator to change
    it; its stage's documentation says why it is static): dynamic float32 fields in
    this pipeline: [...]`); each generator page says why its static knobs are static.
  - **Bounds: one rule.** A scalar (construction, `StageParams.set`, a randomizer's
    initial range) is checked on the host in `float32` (`f32(lo) <= f32(v) <= f32(hi)`);
    an array value (an `[E]` `set`, a caller kernel, a draw from bounds written at run
    time) is clamped to the declared bounds when the kernel reads it (NaN passes; an
    unbounded side is not clamped). Each array-reading kernel loads its env's knobs once,
    clamps them with comparisons (not `wp.min` / `wp.max`, whose CUDA lowering leaves
    the sign of a zero unspecified) and runs the scalar kernel's body verbatim, so every
    configuration keeps its bytes.
  - **The five generators are Stages**: `bezier`, `hull`, `polar`, `voronoi` and
    `checkpoint` are `kind="generator"`, `output="centerline"`, `capture=STATIC` Stages
    (Stages like `repulsive`, a `BOUNDED` Stage since 0.2.0.dev4), registered as
    classes;
    `registry.get("generator", "polar").Params` is their Params class (see Deprecated for
    `.params`). Their module functions keep their frozen signatures and scalar kernels:
    `generate_centerline_warp` / `bezier_alloc_scratch`, `generate_hull_warp`,
    `generate_polar_warp`, `generate_voronoi_warp`, `generate_checkpoint_warp` and
    their `*_alloc_scratch`, `warp_zprofile.apply_z_profile` / `apply_z_profile_knots`,
    and the float helpers `sample_polar_controls`, `sample_checkpoint_rings`,
    `select_voronoi_anchors`; the Stages and the legacy functions share private cores.
  - **`sample_polar_controls_per_env`, `sample_checkpoint_rings_per_env`,
    `select_voronoi_anchors_per_env`**: the gate generators' shared sampling helpers
    with `[E]` `float32` arrays for the knobs (clamped to the declared bounds at use).
    They live in the generator modules, outside the public API, and outlive the float
    forms.
  - **`PipelineSpec.with_randomize(randomize)`** (public): the spec with each
    randomizer inserted right before its target, exactly as `presets.track` /
    `presets.gates` do (they now call it); the same mapping or `StageRef` forms, the same
    errors, plus one: a field the spec already randomizes is refused (`ValueError`
    naming that randomizer; `spec.replace` it, or write its live `lo` / `hi`). A
    pipeline built from a config gets randomizers in one call:
    `TrackGenerator(legacy.to_pipeline_spec(cfg).with_randomize({...}), rng)`, or
    `GateGenerator(legacy.to_gate_pipeline_spec(gcfg).with_randomize({...}), rng)` for a
    `GateGenConfig`. The configs still have no `randomize` field.
  - **Plan-time rules on built-in knobs** (when the pipeline is built, for every stage
    with the rule, randomized or not): `PlanError` when randomizers on `z_min` and
    `z_max` can cross (`hi(z_min) > lo(z_max)`, or one drawn range crossing the other's
    fixed value); a `UserWarning` when a `generator.rad` randomizer's `hi` exceeds every
    value `handle_clamp_frac` can take (its drawn `hi`, by a randomizer or
    `style_sampling`, or its value), since `rad` is inert above it; a `UserWarning` for
    an inert elevation window: `z_min == z_max`, neither drawn, with `z_noise_amplitude`
    (`noise`) or `z_max_step` (`random_walk`) able to be positive (configured, or the
    `hi` of its randomizer). A window drawn by randomizers does not warn; one opened
    only at run time with `set` cannot be seen at build. A run-time crossing
    (`z_min > z_max` written into the arrays) is not refused and behaves as the scalar
    kernels did (`uniform` draws in `[z_max, z_min]`, `random_walk` / `noise` give
    `z_max`). The rules are stated once in the tuning guide (`tuning-per-env`).
  - **`style_sampling` composes with the dynamic fields** (kept by PR13c; deprecated by
    PR13d, see Deprecated): a
    `rad` / `handle_clamp_frac` with a `*_range` is drawn by the style kernel straight
    into the knob's parameter array, which is unlatched and reported as drawn by the
    stage itself (`StageParams.drawn` maps it to the stage's own id, `"generator"`, so
    every `drawn` value is a key of `gen.params`; `repr` says `rad=<drawn by the
    generator's style_sampling (rad_range)>`), so
    `params["generator"]["rad"]` holds each env's style draw; a knob whose range is
    `None` is an ordinary dynamic field (`set` and `randomize=` work on it).
    `scale_range` is unchanged (static; PR13d's `size` composes with it).
  - **Built-in static views refuse host reads of dynamic fields**: in `build` the
    built-in Params' dynamic names hold a sentinel whose numeric conversions,
    comparisons (`==` / `!=` and so `in` included), hashing and arithmetic raise
    `TypeError` (`<Params>.<field> is dynamic: a
    built-in stage reads it on the device through b.param(...)`), so a knob routed
    through the host by mistake fails instead of silently using the default
    (`getattr(view, name, default)` returns the sentinel). Plugin Params keep the
    documented `AttributeError`. Identity and type checks (`x is None`,
    `isinstance(x, float)`) cannot be intercepted.
- **Conformance kit: `params.per_env`** (every kind, after `statelessness`, every
  device): every dynamic field of the stage under test that is not drawn is set per env
  (each moved from its initial value toward its farther declared bound, a field with one
  bound away from it, fields with no bound alternating down / up), and env `e` of that generation must equal, bitwise, env `e` of
  a generation with the fields set to `e`'s values as scalars; the live arrays must read
  back what was set. Skipped ("no dynamic parameters") without one. It passes for every
  built-in.
- Docs: each generator page gains "Per-env (dynamic) parameters" (bounds, `randomize=`
  key, harder-when, effective range, yield at the bounds, why each other knob is static);
  `docs/generators/overview.rst` ("Per-env shape knobs"); the tuning guide's "Per-env
  parameters: randomization and curricula" (randomize on built-in knobs, curricula
  through `gen.params`, the bounds rule, yields, per-profile elevation knobs and
  windows, masked-regeneration bookkeeping, segmented-Program timing) and its
  difficulty-axis table; `docs/tracks-25d.rst` ("Per-env altitude") and
  `docs/gates-3d.rst`; the configuration reference (a "Kind" column in the built-in
  Params tables, the bounds and rejections, `registry.get(kind, name).Params`, the
  drawable built-in fields and `PipelineSpec.with_randomize` in `randomize=`);
  `pipeline-preview.rst` ("Dynamic built-in knobs", launch counts, segment-0 latch
  timing); `writing-a-generator.rst` (built-ins are Stages, "Adding a built-in
  generator": the dynamic-knob rule and the array-reading twin pattern, the legacy
  module functions); `writing-a-stage.rst` and `testing-a-plugin.rst`
  (`params.per_env`); the API reference;
  `docs/library-readiness/target-architecture.md` (PR13c as shipped, new roadmap
  PR13d).

- **Per-env size and the remaining difficulty axes** (roadmap PR13d; the size, corner
  count and gate size axes of finding A15, and the end of A88). Each new axis is a
  dynamic field, so caller-graph curricula reach it and `randomize=` draws the `float32`
  ones; every default is byte-identical.
  - **`size`**: a per-env multiplier of the plan view (`float32`, default `1.0`, bounds
    `[0.1, 10]`), the last field of `BezierParams`, `HullParams`, `PolarParams`,
    `VoronoiParams`, `CheckpointParams`, `RepulsiveParams`, `PointGateParams` and the
    new `NormalizeExtentParams` (`track_gen.stages.params`; `normalize.extent` had no
    Params). It acts where each generator applies `scale`, right after it: the corner
    grid of `bezier` / `hull` and their gate generators (`((x * cell_size + jitter) *
    scale) * size[e]`: at `scale = 1`, `size = s` is bitwise `scale = s`), the
    normalization target of `polar`, `voronoi` (both normalizations) and `checkpoint`
    (`f32(scale * TARGET_EXTENT) * size[e]`, within one `float32` rounding of `scale =
    s`), the gate pipeline's `normalize.extent` stage for the `polar`, `voronoi` and
    `checkpoint` gate generators, and `repulsive`'s output (its `scale` sets a physics
    domain, so the copy of the grown loop into `raw.xy` multiplies its finite values:
    `raw.xy` at size `s` is `f32(raw.xy(1) * s)` bit for bit, a point with a non-finite
    component is copied unscaled (a NaN bit for bit, its payload included), and the
    growth does not depend on the size). Keys:
    `generator.size` (track generators, the `bezier` / `hull` gate generators, the
    track generator of `source="stations"` gates), `normalize.size` (the `polar`,
    `voronoi` and `checkpoint` gate generators). Bit-exact at `1.0` on CPU and CUDA: the
    product never feeds an addition, so no FMA contraction is created or removed (a CPU
    test locks the generated CUDA source of each twin). It scales the plan view only:
    `half_width`, `spacing`, `gate_radius`, `gate_width` and z (`z_min` / `z_max`,
    amplitudes, `z_valid_grade`) stay absolute, so `uniform` / `noise` grades fall as
    `~1 / size` (documented). **`external` has no `size`**: it ingests the caller's
    geometry as given (scale `centerline.xy` in your graph); `generator.size`,
    `set("size")` and a flat `size` on it raise with that hint, and its launch sequence
    is unchanged. Each generator page lists the size's yield and truncation point.
  - **The sizing rule** (the tuning guide's "Size: the sizing rule" and "Size and the
    utilities"): size changes no buffer and no launch, and a track that outgrows `n_max`
    is truncated and fails validity (PR5). `n_max >= size_hi * L_max / spacing + 1`
    and `num_points >= size_hi * L_max / spacing`, with `L_max` measured per generator
    (the longest loop of 8192 envs; at the default `n_max = 384`, truncation starts at
    about size 2.6 for `bezier`, 2.8 `hull`, 3.7 `polar`, 4.6 `voronoi`, 3.2
    `checkpoint` and 1.6 `repulsive`); the
    checkpoint and prop rings of a size curriculum need `max_checkpoints = ceil(1.5 *
    n_max * spacing / checkpoint_spacing)` (and the same for `max_props`), and the SDF /
    heightfield grids `R >= R_ref * size_hi`.
  - **A build-time truncation `UserWarning`**: a `generator.size` randomizer whose `hi`,
    or a construction value, would truncate the generator's nominal longest loop at the
    pipeline's `n_max` warns once, naming the `n_max` that fits (`generator.size drawn
    up to 2.0 truncates at n_max=384 (nominal raw perimeter ~29.7 > (n_max - 1) * spacing
    = 22.98): raise n_max to >= 496 or lower the range; truncated envs fail
    validity`). The nominal loop is exact for `repulsive` (from its params); for the
    others it is the measured longest loop of 8192 envs at the defaults (so it fires
    from the documented "truncates from" size; a larger batch can truncate a few envs
    just below it), scaled by `bezier` / `hull`'s `max_num_points`,
    `hull_displacement` (its randomizer's `hi` when drawn) and `style_sampling`'s
    `scale_range`. It is not checked when `polar`'s `polar_num_knots` /
    `polar_radial_jitter`, `voronoi`'s `voronoi_num_sites` / `voronoi_control_points` /
    `voronoi_angular_jitter`, or `checkpoint`'s `checkpoint_count` /
    `checkpoint_radius_min_frac` / `checkpoint_angle_jitter` / `checkpoint_steer_gain` /
    `checkpoint_lookahead_frac` / `checkpoint_best_of_k` differ from their defaults or
    are drawn (the other knobs keep the check), nor for plugins or a `set()`
    curriculum. Also a `UserWarning` when `style_sampling`'s `scale_range` and a
    `generator.size` randomizer are both active (two per-env size draws multiply).
  - **`TrackGenConfig.size` / `GateGenConfig.size`** (flat, default `1.0`, just before
    `generator_params`): the initial size, and the only construction-time route for
    `Course`. `GateGenConfig.size` reaches `normalize.size` for the ring gate generators
    (the normalize `StageRef` gets `params={"size": ...}` only when it is not `1.0`, so
    the default `presets.*` spec dicts and the ring gates' normalize `StageRef` are
    unchanged; a config-built spec dict gains `size`, see Behaviour changes). A plugin
    Stage whose Params declare `size` as a dynamic `float` and whose `from_legacy`
    leaves it at its default receives the flat value (as `half_width` is mapped; a
    static or `int` field named `size` is the plugin's own: the flat `size` and
    `presets.*(size=)` never write it, and a value other than `1.0` raises); a gate
    generator re-normalized by `normalize.extent` (one that declares `size` included)
    has the flat size routed to `normalize.size`, with a build-time `UserWarning` for a
    `generator.size` randomizer or construction value on it (the normalization erases
    it).
    **`presets.track(size=1.0)` /
    `presets.gates(size=1.0)`**, next to `scale=`, routed to `generator.size` or
    `normalize.size`; `size=1.0` adds nothing to the spec.
  - **`size=` on `ops.sample_grid_points` and `ops.normalize_extent`** (keyword-only,
    `[E]` `float32`, each value clamped at use to the new public
    **`ops.constants.SIZE_BOUNDS`** `(0.1, 10.0)`, the convention a plugin's own `size`
    field imports); `size=None` runs the unchanged kernels and a size-1 array gives
    their bytes. Twins: `_sample_grid_points_sized_k`,
    `_sample_grid_points_scaled_sized_k`, `_normalize_extent_sized_k` (no `count`
    variant: `normalize_extent(count=..., size=...)` raises `ValueError`).
  - **A dynamic corner count**: `bezier` / `hull`'s `min_num_points` is a dynamic
    `int32` field (latched by `_latch_all_i32_k`; the count twin `_sample_counts_dyn_k`
    clamps it to `[2, max_num_points]` at use). `max_num_points` stays static (it sizes
    the buffers). Per env the count is uniform on `[clamp(min_e, 2, max), max]` from the
    same draw and over one fixed set of corner positions, so raising it never lowers an
    env's count and the axis only narrows toward `max_num_points` (documented, with the
    curriculum recipe). `set` takes an `int` in `[2, max_num_points]`; `randomize=` does
    not draw it (`int32`). The gate generators' `min_num_points` stays static (their
    `min_gates` capacity).
  - **A dynamic gate bar width**: `gate_frames.default`'s `gate_width` (key
    `gate_frames.gate_width`, the full bar width, `>= 0`), read per env by
    `_finalize_frame_dyn_k`. **`validity.gate_default` checks bar crossings per env**
    (`_finalize_validity_bars_k`): the crossing loop runs for each env with a real gate
    of non-zero `half_size` (a NaN counts), so a run-time width (drawn, or set from
    point gates) is checked as a width built in, and point-gate envs skip the loop. Its
    valid and fail bits equal the 0.2.0.dev4 kernel's wherever its check was on, and
    the check-off kernel's for every env without bars. Yields (valid at width 0.2 /
    0.5, `E = 512`): `bezier` 0.51 / 0.04, `hull` 0.48 / 0.02, `polar` 1.00 / 0.55,
    `voronoi` 0.81 / 0.04, `checkpoint` 0.83 / 0.28.
  - **`StageParams.in_effect(field, *, out=None, stream=None)`**: the `[E]` values the
    stage READ for its current rows (the latched work copy, or the drawn array), for
    curriculum bookkeeping under masked regeneration; without `out=` the internal array
    (zero-copy, a stable pointer, read-only by contract), with `out=` one recordable copy
    into the caller's `[E]` array on the resolved stream. **`StageParams.bounds(field)`**:
    the `(lo, hi)` a built-in kernel clamps an array value to (declared, narrowed per
    instance: `min_num_points` `(2, max_num_points)`), so the value used is
    `clamp(in_effect(f), *bounds(f))`. A latched field is current once segment 0 has run,
    a drawn one once its randomizer's segment has.
  - **Hints**: the `KeyError` of `StageParams` and the randomize `PlanError`s name the
    key that holds an axis (`generator.size` on a ring gate pipeline -> `normalize.size`;
    external: scale `centerline.xy`; a gate generator's `min_num_points` is its gate
    count capacity, not per env in 0.x).
  - Docs: the generator pages (size, the corner count, the `style_sampling` migration,
    `repulsive`'s "Per-env size"), the generators overview, the tuning guide ("Size: the
    sizing rule", "Size and the utilities", the `in_effect` / `bounds` recipe replacing
    "keep your own per-env copy", the difficulty-axis table: Course size, Corner count,
    Gate bar width, Gate count), the gate tutorial ("Per-env bar width and course
    size"), `docs/gates-3d.rst`, `docs/tracks-25d.rst`, the configuration reference
    ("Course size", the Params tables, the deprecated and inert fields),
    `pipeline-preview.rst` (dynamic built-ins, `in_effect`, the gate validity and
    launch tables), `writing-a-generator.rst` ("A per-env size": the plugin convention
    and the FMA rule), `testing-a-plugin.rst`, the utilities pages (`Course`, props,
    collision), the API reference and `docs/library-readiness/target-architecture.md`.

### Behaviour changes

- **`clone()` validates**: a malformed hand-built `Track` / `GateSequence` (a field of
  the wrong length, dtype or device) now raises `ValueError` instead of being copied
  blindly. It also allocates on Warp's current stream after waiting on the producer's
  stream (the same copies as before when the generator has no default stream), and then
  waits for its copies on the host (on CUDA it now blocks until the clone is complete).
- **`clone()` inside a capture** (a Warp CPU or CUDA capture, or a torch capture) now
  raises `CaptureError` before allocating; 0.2.0.dev4's `wp.clone`-based clone could be
  recorded. Inside a graph, `copy_from` into a preallocated `alloc()` result.
- **`out=` must be a `Track` / `GateSequence` instance**: an object that merely has the
  fields (accepted by 0.2.0.dev4) raises `TypeError`.
- **Building into storage still owned by a live Program raises `PlanError`**, also when
  rebuilding the same pipeline into the same `out` (allowed by 0.2.0.dev4, e.g. to
  change static params while keeping the pointers a caller graph holds): drop the
  earlier generator / Program (`del gen`) before rebuilding into the same result.
- **The utilities' construction errors changed**: a missing member raises `TypeError`
  naming it (`CollisionChecker needs a HasBand source (...): Centerline has no 'outer',
  'inner'`), a mismatched one `ValueError` prefixed by the utility (`CollisionChecker:
  Track.inner: expected ...`), instead of `track batch layout invalid`. They check every
  member they read (a mismatched field the old one-field inference missed now raises)
  and nothing else (a field they do not read is never looked at).
- **`CourseLine.track` is a `Centerline`** and emits a `DeprecationWarning` (use
  `CourseLine.centerline`; removed at 1.0): reading its `outer`, `inner`, `normal` or
  `winding` (NaN / zero placeholders before) raises `AttributeError`, and
  `isinstance(course_line.track, Track)` is `False`.
- **`out.attributes` keys the pipeline does not export are padded, not refused, on a
  view of a larger result** (a mixed batch). On a whole result the 0.2.0.dev4
  `ValueError` stays (its tail now points to views instead of "reserved for PR13b's
  mixed batches"), and an exported key of the wrong dtype reads `out.attributes['k'] is
  float32; this pipeline exports 'k' as int32` (was `out=: the array bound to cl.k has
  dtype ...`).
- **`BatchSpec.closed`**: a non-`bool` (`1`, `np.True_`) raises `PlanError`, and the
  `closed=False` message names the closed-only stages (it still says "open tracks").
  Both are raised before any generator is bound or any `Params.validate` runs.
- **`presets.track(relax=)` / `presets.gates(relax=)` default to `"auto"`** (was
  `"xpbd"`): the same stage for every built-in, `relax.none` for `external`.
- **`Layout` validates its fields** (`TypeError` / `ValueError` for a `bool` or
  non-integer count, a value below 1, a non-`str` device, a non-`bool` `closed`), and the
  planner builds every layout from `BatchSpec.closed`.
- **The generator is the first stage that is not a randomizer** (it was `stages[0]`):
  specs without randomizers are unaffected; code that looks the generator up should use
  `spec.stages[spec.ids.index("generator")]`.
- **`registry.add` accepts `preferred={"relax": ...}` on centerline generators** (it was
  gate-generator only) and refuses a `declare()` override on other kinds;
  `registry.describe` reports `seeded` for every generator and `preferred` for
  centerline generators, and a Stage generator's `writes` / `provides` as its
  `declare()` at the default Params.
- **`FrameChecker(seq, num_envs, ...)`** cross-checks `num_envs` against the source's
  layout.
- **`registry.available("generator")` now includes `external`** (`seeded=False`): code
  that samples generators from that list (domain randomization, benchmarks) gets the
  same placeholder circles for every seed. Filter on the declaration:
  `[n for n in registry.available("generator") if registry.describe("generator",
  n).seeded]` (docs: the generators overview and "Choosing a generator").
- **Generator enumerations skip `seeded=False` generators**: the parameter explorer's
  generator list, `benchmarks/compare_generators.py` and the README asset renderer list
  only seed-dependent generators (`registry.describe("generator", n).seeded`), so
  `external` does not appear there.
- **The utilities bind their source's arrays at construction**: a member reassigned on
  the source object after a utility was built (`track.center = other`) is no longer
  read by it (0.2.0.dev4 re-read `track.<member>` at every call); write into the arrays
  instead, or build a new utility.

- **The five built-in sampling generators are Stage classes** (roadmap PR13c):
  `registry.get("generator", g)` (and `generator_registry.get(g)`, `GENERATORS[g]`)
  returns a class for `bezier`, `hull`, `polar`, `voronoi` and `checkpoint`, not a
  `GeneratorSpec` instance: use `.Params` (`.params` warns, see Deprecated); the spec's
  other attributes are gone without a deprecation: `.generate` / `.alloc_scratch` (call
  the module's `generate_*_warp` / `*_alloc_scratch`, unchanged) and `.capturable` /
  `.accepts_ctx` (read `registry.describe("generator", g).capturable` /
  `.accepts_ctx`, both `True`). `registry.describe(...)` keeps `capturable`,
  `accepts_ctx`, `builtin`, `seeded` and the origin, now reports the Stage fields for
  the five (`capture=CaptureClass.STATIC` instead of `None`, `output="centerline"`
  instead of `None`, `writes={"xy": "raw.xy", "valid": "gen.valid"}` instead of `{}`),
  and its params schema marks the dynamic fields (`x-dynamic`, `minimum` / `maximum`,
  `"x-bounds": "float32"`). `registry.add(GeneratorSpec(...,
  params=registry.get("generator", "bezier").Params))` raises `TypeError` (a
  `GeneratorSpec` params class takes static fields only). Their scratch buffers are now the Stage's (`"scratch:generator/<name>"`, e.g.
  `gen_crossers`, listed by the kit's `buffers_of`); the private `gen._scratch.gen` is a
  built stage. The PR9 reference composition (`warp_pipeline._inflate_warp_alloc` /
  `_run_pipeline` with `generator_spec=None`) looks the built-ins up in a private table
  of the legacy functions.
- **Newly rejected built-in values** (`ValueError` when the generator is constructed,
  after the legacy rules, which keep their messages and order): `rad` outside `[0, 1]`;
  `handle_clamp_frac < 0`; `hull_displacement` outside `[0, 1]`;
  **`polar_radial_jitter` outside `[0, 0.85]` and `polar_angular_jitter` outside
  `[0, 0.45]` (a break: 0.2.0.dev4 documented and applied a clamp to these ranges; set
  the bound itself, e.g. `polar_radial_jitter=0.85` for `1.0`, for exactly the old
  output; the `ValueError` says so: `polar_radial_jitter must be in [0, 0.85], got 1.0
  (0.2.0.dev4 clamped it to that range: set polar_radial_jitter=0.85 for the same
  output)`)**; `voronoi_radial_variation > 0.85` (no effect beyond it for any layout);
  `voronoi_angular_jitter > pi`; `checkpoint_angle_jitter` outside `[0, 1]`;
  `checkpoint_turn_rate` outside `[0, pi]`; a `bool` for any of the 14 dynamic built-in
  fields (`rad=True` was accepted), a numpy bool or a bool array / tensor included.
  `checkpoint_radius_min_frac` keeps its legacy `[0, 1)` rule for the configured value,
  while `set` / `randomize=` accept its declared bound `1` (every checkpoint on the
  outer radius). Every default and every value in the docs,
  examples, benchmarks and the parameter explorer's sliders lies inside. A value
  0.2.0.dev4 accepted through `float()` is still accepted: a non-bool numeric string
  (YAML's `"1e9"` for `handle_clamp_frac`), a 0-d numpy array or a torch scalar tensor
  is converted to a `float`, on every construction path.
- **Bounds of every dynamic `float32` field (plugin ones included) are compared in
  `float32`** (construction, `StageParams.set(scalar)`, a randomizer's range): looser
  than 0.2.0.dev4's float64 comparison by less than half an ulp at a bound, so a value
  read back from an array (`float(np.float32(0.85))`) is accepted. Nothing accepted
  before is refused. The params schema marks such bounds `"x-bounds": "float32"` (a
  generic JSON-schema validator compares in float64 and refuses that read-back value);
  a flat dynamic knob compares with `generator_params` in `float32` as well, so a
  `np.float32(0.4)` or `torch.tensor(0.4)` flat value equal to the default or to the
  `generator_params` value is no conflict (as under NumPy 2's comparison rules).
- **Array values of the built-in dynamic fields are clamped to the declared bounds** when
  the kernels read them (an `[E]` `set` or a caller kernel is not range-checked).
- **`style_sampling`'s ranged knobs are reported as drawn**: with `rad_range` (or
  `handle_clamp_frac_range`) set, `gen.params["generator"]["rad"]` exists and holds each
  env's draw, `params["generator"].drawn["rad"] == "generator"` (the stage itself),
  `set("rad", ...)` raises `ValueError` (`generator.rad is drawn every
  generation by the generator's style_sampling (rad_range); set rad_range=None to set
  it`), and `randomize={"generator.rad": ...}` is a `PlanError` (`generator.rad is drawn
  by the generator's style_sampling (rad_range): set rad_range=None to randomize it`).
- **New warnings and plan errors**: the `UserWarning`s and `PlanError`s of the
  randomizer rules and inert elevation windows (Added); the static-field `PlanError` for
  a built-in knob says it is static and where to read why (it said "built-in knobs are
  static in 0.x"). `legacy.to_pipeline_spec` given a `GateGenConfig` raises `TypeError`
  pointing to `legacy.to_gate_pipeline_spec` (it failed with an `AttributeError` on
  `N_max`).
- **Launch sequences (re-pinned deliberately; bytes unchanged).** One latch launch per
  latched dynamic field per generation, right after the effective seed (drawn fields
  have none), and each knob-reading kernel replaced by its array-reading twin at the same
  position (`_assemble_k` / `_assemble_style_k` -> `_assemble_dyn_k`,
  `_midpoint_displace_dyn_k`, `_polar_controls_dyn_k`, `_select_anchor_sites_dyn_k`,
  `_sample_checkpoints_dyn_k`, `_steer_dyn_k`, `_z_uniform_dyn_k`, `_z_walk_dyn_k`,
  `_z_noise_dyn_k`), same dimensions and arrays, the knob scalars now arrays
  (`_assemble_dyn_k` also takes its four bound scalars). Launches per `generate()` (CPU):
  default track (`bezier`, flat) 152 -> 154; `bezier` + `style_sampling` with all three
  ranges 153 -> 153, with one of `rad_range` / `handle_clamp_frac_range` 153 -> 154,
  with neither 153 -> 155; `hull` 151 -> 152; `polar` 143 -> 145; `voronoi` 151 -> 153;
  `checkpoint` (any `checkpoint_best_of_k`, clip or not) 146 -> 149; `bezier` +
  elevation `uniform` / `random_walk` / `noise` 156 / 156 / 154 -> 160 / 161 / 159;
  `repulsive`, `external` and the default gate pipeline (`bezier` / `hull`, 14)
  unchanged; gates `polar` / `checkpoint` 14 -> 16, `voronoi` 15 -> 17, plus 2 / 3 / 3
  with a `uniform` / `random_walk` / `noise` elevation. The re-pinned tests:
  `tests/test_pipeline_parity.py` (`test_default_bezier_launch_count`, 154; the PR9
  sequence test's four elevation cases through one shared delta helper),
  `tests/test_commit.py::test_launch_counts` (154; masked `154 + 1 + 2` with the hold at
  index 4; poison 155; a check after validity `154 + 3`),
  `tests/test_gate_commit.py::test_launch_counts` (polar 14 -> 16) and
  `tests/test_gate_pipeline_parity.py::_expected_launches` (+2 and the twin for polar /
  checkpoint / voronoi, +3 for `noise`; its golden JSON is unchanged).
- **The conformance kit runs `params.per_env`**: a plugin with a dynamic field whose
  kernel reads `param[0]` (or a host copy) instead of `param[e]`, or writes its
  parameter array, now fails it.
- **Launch sequences re-pinned again by PR13d (bytes unchanged).** The new latched
  fields add one latch launch each, in the block right after the effective seed (slot
  order, then field order; `min_num_points` first, through `_latch_all_i32_k`, `size`
  last of its generator), and the knob readers are replaced in place by their twins:
  `_sample_counts_k` -> `_sample_counts_dyn_k` (its `lo` scalar becomes the array),
  `_finalize_frame_k` -> `_finalize_frame_dyn_k` (its `gate_width` scalar becomes the
  array), `_finalize_validity_k` -> `_finalize_validity_bars_k` (its crossing-toggle
  scalar dropped, every gate pipeline), and the size twins `_sample_grid_points_sized_k`,
  `_sample_grid_points_scaled_sized_k`, `_normalize_extent_sized_k`,
  `_normalize_extent_to_sized_k` keep every scalar and append the `[E]` size array;
  `repulsive`'s final copy becomes the `_copy_sized_k` launch. Launches per `generate()`
  (CPU), 0.2.0.dev4 -> PR13c -> 0.2.0.dev5: default track (`bezier`, flat) 152 -> 154 ->
  156; `hull` 151 -> 152 -> 154; `polar` 143 -> 145 -> 146; `voronoi` 151 -> 153 -> 154;
  `checkpoint` 146 -> 149 -> 150; `bezier` + `style_sampling` with all three ranges /
  one of `rad_range`, `handle_clamp_frac_range` / `scale_range` only 153 -> 153 / 154 /
  155 -> 155 / 156 / 157; `bezier` + elevation `uniform` / `random_walk` / `noise` 156 /
  156 / 154 -> 160 / 161 / 159 -> 162 / 163 / 161; `repulsive` +1; `external`
  unchanged (20); the default gate pipeline (`bezier` / `hull`) 14 -> 14 -> 16; gates
  `polar` / `checkpoint` 14 -> 16 -> 18, `voronoi` 15 -> 17 -> 19; `presets.gates(
  source="stations")` (`bezier`, 12 gates) 27 -> 29 -> 32. A masked generation runs the
  same latches as `_latch_masked_*`, so its hold moves (default track: index 6;
  default gates: index 4). The re-pinned tests, through one rule
  (`tests/_pr13d_delta.py`, layered on `tests/_pr13c_delta.py`, which checks the latch
  block's start, its exact dtype sequence computed from the declarations and the masked
  records, per pipeline): `tests/test_pipeline_parity.py`
  (`test_default_bezier_launch_count`: 156), `tests/test_commit.py::test_launch_counts`
  (156; masked 159 with the hold at index 6; poison 157; a check after validity 159),
  `tests/test_gate_commit.py::test_launch_counts` (16; `polar` 18; masked 19 with the
  hold at index 4; poison 17; checked 19), `tests/test_gate_pipeline_parity.py`
  (`_expected_launches` on its unchanged golden; `..._emits_16_records`) and the launch
  tables of `tests/test_dynamic_knobs.py`.
- **`validity.gate_default` checks crossings per env**, and its `gate_width` and
  `check_crossing` are **inert** (both values select the per-env check; kept so specs and
  dicts load, removed in PR16). The planner no longer refuses `gate_frames.posed` with
  `check_crossing="width"` (it builds and checks crossings, the bytes of `"always"`).
  The planner still requires `gate_frames` and `validity` to start from the same
  `gate_width` when validity's is set (see below; the message's reason is updated: one
  initial bar width per pipeline).
  Every PR13c-legal spec keeps its valid and fail bits.
- **`gen.params` of every built-in gate pipeline lists `"gate_frames"`** (its
  `gate_width`), and the `polar`, `voronoi` and `checkpoint` gate pipelines list
  `"normalize"` (its `size`); `gen.params["generator"]` of `repulsive` and of the
  `bezier` / `hull` gate generators holds `size`, of `bezier` / `hull` also
  `min_num_points` (an `int32` array).
- **`min_num_points` of `bezier` / `hull` is checked as an `int32`**: `set` refuses a
  scalar outside `[2, max_num_points]` (`BezierParams.min_num_points must be in [2, 13],
  at most max_num_points (the static corner capacity), got 40`), any `float` (`9.0`
  included: `... is dynamic and must be an int, got 9.0`) and a `bool`; it takes a
  Python `int` or a NumPy integer. Construction still converts a value `int()` accepts
  (YAML's `9.0`, a NumPy integer, `"9"`), and still refuses `min_num_points=True` (its
  `>= 2` check, as before). The bounds error's
  last clause is the full reason
  (a randomizer's reads `the declared range of X (...)`). Host reads of `size`,
  `min_num_points` and `gate_width` from a built-in's static view raise the PR13c
  `TypeError`.
- **New errors.** `ValueError` for a `size` other than `1.0` that no stage can apply:
  `legacy.to_pipeline_spec` / `to_gate_pipeline_spec` when no stage of the resolved spec
  declares `size` (a plugin generator without it, a legacy `GeneratorSpec`, `external`,
  a legacy `GateGeneratorSpec`), the frozen module functions (`generate_*_warp`) given a
  config with `size != 1`, and `presets.track(generator="external", size=...)`;
  `presets.track` / `presets.gates` with both `size=` and `generator_params={"size":
  ...}`; `ops.normalize_extent(count=..., size=...)`. `PlanError` for
  `randomize={"generator.min_num_points": ...}` (an `int32` field) and for
  `generator.size` where another key holds the size (with the hint).
- **New warnings**: `style_sampling`'s `DeprecationWarning` (see Deprecated), the
  build-time truncation `UserWarning` for a size range and the composed-size-draw
  `UserWarning` (Added).
- **Spec dicts that carry a built-in Params instance gain a `size` entry**: every
  config-built spec (`legacy.to_pipeline_spec(TrackGenConfig(generator=g))` for
  `bezier`, `hull`, `polar`, `voronoi`, `checkpoint` and `repulsive`;
  `legacy.to_gate_pipeline_spec(GateGenConfig(generator=g))` for the `bezier` / `hull`
  gate generators) and any spec built with a `BezierParams`, `HullParams`,
  `PolarParams`, `VoronoiParams`, `CheckpointParams`, `RepulsiveParams` or
  `PointGateParams` instance writes `"size": 1.0` in the generator's params, and
  `min_num_points` is written as the coerced `int` (a YAML `9.0` becomes `9`). Output
  bytes are unchanged, and dicts written by 0.2.0.dev4 still load, but a dict written by
  0.2.0.dev5 from such a spec does not build on 0.2.0.dev4 (`unknown generator_params
  ... ['size']`), and `describe()` / dict diffs used for experiment tracking change. The
  `presets.*` dicts and the normalize `StageRef` are unchanged.
- **`TrackGenConfig` / `GateGenConfig` gain a public field named `size`** (the course
  size): a config subclass that declares its own `size` field (an arena size, say) must
  rename it, since the built-in Params read the flat field by name (`BezierParams.size
  must be in [0.1, 10.0], got 64.0`) and the frozen `generate_*_warp` refuse a `size`
  other than 1 (a non-numeric one with `config.size='...' is not a number; ... rename a
  config subclass field named size`).
- **`validity.gate_default`'s `gate_width` takes part in the planner's agreement check
  only when set** (to a value other than its default `0`): a hand-built spec that
  widens `gate_frames.default` alone builds (the inert copy no longer contradicts it),
  while two explicit, different widths still raise `the stages disagree on
  gate_width`.
- **`StageParams.set` names the merged range** of a field with per-instance bounds:
  `set("min_num_points", 1)` reads `must be in [2, 13], at most max_num_points ...` (the
  range `bounds()` reports), not the declared `[2, inf]`.
- **`Course` in gates mode refuses `set("gate_width", v)` with `v <= 0`** on
  `course.generator.params["gate_frames"]` (a width-0 gate can never be passed); an
  `[E]` array is still not range-checked (a value `<= 0` is used as 0).
- **More hints**: `randomize={"validity.gate_width": ...}` / `check_crossing` (inert
  fields), `generator.scale` and `generator_params={"scale": ...}` (the per-env
  multiplier is `size`), `randomize={"normalize.size": ...}` on a gate pipeline without
  `normalize` (use `generator.size`), and `generator_params={"size": ...}` on the
  `polar` / `voronoi` / `checkpoint` gate generators (use `normalize.size`, initial
  value `GateGenConfig.size`); a legacy spec without `size` is told to port to a Stage.
- **The conformance kit's `params.per_env` runs for more built-ins**: `repulsive` and
  the `bezier` / `hull` gate generators (on `size`), and `bezier` / `hull` on their
  `int32` `min_num_points`; `external` still skips it. Third-party plugins are
  unaffected.

### Changed (outputs)

None. Every default pipeline (each `TrackGenConfig` / `GateGenConfig`, each preset at its
defaults, `relax="auto"` included), every fingerprint, the PR10a, PR11 and PR13a parity
suites and goldens are byte-identical. `clone()` copies the same bytes on the same stream
when no producer stream is involved, and `CourseLine.refresh()` launches the same two
kernels into the same buffers. The dynamic built-in knobs (PR13c) keep the bytes of every
configuration 0.2.0.dev5 still accepts, `style_sampling` included, but change the launch
sequences of the generators and elevations that have them (latches and array-reading
kernels: see Behaviour changes). One unreachable-in-practice exception: under
`style_sampling` with a knob's range `None` and that knob set to `-0.0` (`rad=-0.0`),
a coordinate that is exactly zero may now carry the sign `-0.0` (the collapsed style draw
used to give `+0.0`). The PR13d axes change no byte of any configuration that PR13c
accepted (fingerprints, the PR10a / PR11 / PR13a parity suites and goldens, the
`style_sampling` bytes): new outputs come only through the new dynamic fields (`size`,
the corner count, the bar width at run time), and the per-env crossing check takes the
decisions of the 0.2.0.dev4 kernel wherever a bar width was built in.

### Fixed

- Library-readiness findings closed by roadmap PR13b: MEDIUM A22 "Track and
  GateSequence have fixed schemas" (rest: `layout`, `validate`, the centerline / band
  split through `Centerline` and the protocols; per-env attributes stay open); MEDIUM A44
  "utilities infer batch layout from one field" (construction-time member checks); MEDIUM
  A36 "Track used as a universal representation" (protocols, `Centerline`,
  `CourseLine.centerline`); MEDIUM A42 "no public centerline -> Track ingestion"
  (`generator="external"`, `load_centerlines`); MEDIUM A43 "mixed batches / caller-owned
  outputs" and A120 "the only mixed-batch path is half-capturable" (`out=` views, one
  caller graph); A121 "seed namespace not partitioned across sub-batches" (seed offsets
  on views); the `Track` part of LOW A99 "public helpers allocate per call"
  (`copy_from` into an `alloc` result; `curvature` / `speed_profile` `out=` stay with
  PR14).
- Closed by roadmap PR13c: MEDIUM A15 "all track knobs are host scalars" for the
  built-in knobs (the generators' shape knobs and the elevation ranges are dynamic, drawn
  by `randomize=` and writable from a caller graph; what stays static sizes buffers or
  launches, or is transformed on the host); LOW A88 "style sampling is bezier-only",
  closed as *generalized by randomize* (every built-in sampling generator, `bezier`,
  `hull`, `polar`, `voronoi` and `checkpoint`, has drawable knobs;
  the duplicated scalar kernels serve only the frozen legacy functions, removed at 1.0).
- Closed by roadmap PR13d: the size, corner-count and gate-size axes of MEDIUM A15 (a
  per-env `size` on every built-in generator but `external`, a dynamic `int32`
  `min_num_points` for `bezier` / `hull`, `gate_frames.default`'s `gate_width` with a
  per-env crossing check); LOW A88 completed (`scale_range` has a replacement,
  `generator.size`, and `style_sampling` is deprecated).

### Deprecated

- **`CourseLine.track`**, an alias of `CourseLine.centerline` that emits a
  `DeprecationWarning` (`CourseLine.track is a Centerline since 0.2.0.dev5 and is
  deprecated; use CourseLine.centerline (removed at 1.0)`). `Course`, the tests and
  `viz/plot_gate_courses.py` read `.centerline`.
- **`registry.get("generator", g).params`** for `g` in `bezier`, `hull`, `polar`,
  `voronoi`, `checkpoint` (the 0.2.0.dev4 `GeneratorSpec` attribute): a class attribute
  that returns `.Params` and emits a `DeprecationWarning` (`registry.get('generator',
  'polar').params is deprecated: use .Params (removed in 1.0)`), attributed to the
  calling code. `repulsive` and the gate generators never had it as Stages.
- **`bezier`'s `style_sampling`** and its `rad_range`, `scale_range` and
  `handle_clamp_frac_range` (roadmap PR13d; superseded since PR13c for the two knob
  ranges). Building a `bezier` generator with `style_sampling=True` emits one
  `DeprecationWarning` per build (not per generation), attributed to the first frame
  outside `track_gen`: `bezier's style_sampling is deprecated and will be removed in 1.0:
  draw per-env knobs with randomize= instead (scale_range=(a, b) -> {'generator.size':
  (a / scale, b / scale)}, rad_range -> {'generator.rad': ...}, handle_clamp_frac_range
  -> {'generator.handle_clamp_frac': ...}; set style_sampling=False and use
  PipelineSpec.with_randomize for a config-built pipeline). Course (until it accepts a
  generator, PR15): keep style_sampling, or draw per-env sizes yourself and call
  course.generator.params['generator'].set('size', arr) before regenerate (size
  Course's max_checkpoints for the largest size: ceil(1.5 * N_max * spacing /
  checkpoint_spacing), see CourseConfig.max_checkpoints)`.
  `TrackGenConfig(...)`, `PipelineSpec.from_dict` (loading; building warns) and the
  frozen legacy entry points do not warn. It keeps working, with its bytes, until its
  removal in 1.0 (roadmap PR16), after `Course` accepts a generator (PR15). A
  `DeprecationWarning` attributed to a module other than `__main__`
  (a gym env, a training package) is hidden by Python's default filters: run your tests
  with `-W default::DeprecationWarning`.

  ```python
  # before (deprecated)
  TrackGenConfig(style_sampling=True, scale_range=(0.8, 1.2), rad_range=(0.1, 0.35))
  # after: the same uniform distributions, other draws
  presets.track(num_envs=E, device=d,
                randomize={"generator.size": (0.8, 1.2), "generator.rad": (0.1, 0.35)})
  # from a config: style_sampling=False, explicitly (no double draw)
  cfg = TrackGenConfig(style_sampling=False, ...)
  TrackGenerator(legacy.to_pipeline_spec(cfg).with_randomize({"generator.size": (0.8, 1.2)}), rng)
  # Course (until PR15): keep style_sampling, or
  course.generator.params["generator"].set("size", my_sizes)   # [E] float32, then regenerate
  ```

  Exact at equal values (each env's style draws written into `size` / `rad` /
  `handle_clamp_frac` give the style run's bytes), but the randomizers draw on another
  random stream, so per-seed tracks change: fixed evaluation sets built with
  `style_sampling` need re-baselining. `scale_range=(a, b)` with `scale != 1` maps to
  `(a / scale, b / scale)`, which must lie in `[0.1, 10]`. A `Course` user who switches
  to `set("size", arr)` sizes `max_checkpoints` for the largest size.

### Deferred

- **Per-env size and the rest of the built-in difficulty axes moved from PR13c to a new
  roadmap PR13d** (a deviation from the PR13c scope, which listed "size" knobs), which
  ships in this release (Added). Its own deviations from its roadmap entry: the gate
  bar width is crossing-checked per env by the validity kernel instead of a static
  `> 0` toggle (a run-time width would otherwise go unchecked), so the default gate
  pipeline gains two latches (its generator's `size` and `gate_width`: 14 -> 16, not 15);
  `external` gets no `size` (it would move ingested world coordinates about the data's
  origin); `StageParams.in_effect` gains `out=` and a `bounds()` companion.
- From PR13d, not done: `int32` and categorical draws (a public sampler API); **a dynamic
  corner-count upper bound** (a curriculum gap: the corner axis only narrows toward
  `max_num_points`); dynamic `polar` / `voronoi` / `checkpoint` counts and the gate count
  (they size buffers); a `size=` on further public ops (`ops.closed_catmull_rom`); a
  fused per-dtype latch launch.
- **Roadmap PR14** gains: `Course` / `CheckpointSampler` / `PropSampler` capacities from
  the size-independent bound `ceil(1.5 * n_max * spacing / checkpoint_spacing)` instead
  of a readback of the construction batch (the default `Course`'s buffer shapes change
  there, at checkpoint spacing 0.5: from a count read back from the construction batch,
  e.g. 20 for 32 `bezier` envs at size 1 or 13 for a 1-env default `Course`, to the
  fixed 70), and a
  construction-time SDF-resolution check in the collision adapter.
- **Roadmap PR16** gains: `edgy` dynamic (its in-kernel transform is not bit-exact
  against the host's, free under the golden re-baseline); the removal of the scalar
  kernels and legacy module functions, of the deprecated `.params` alias, of
  `style_sampling` (only after this warning release and after PR15, so `Course` users
  have a replacement) and of `validity.gate_default`'s inert `gate_width` /
  `check_crossing`.
- Unslotted: the `repulsive` growth knobs; `gate_radius` and the relax / validity
  tolerances; categorical draws (layouts);
  `checkpoint_steer_gain` / `checkpoint_lookahead_frac` (only with a measured difficulty
  effect).
- **`closed` in the kernels** (a deviation from the PR13 scope, which plumbed it through
  the relax, elevation and utility kernels): open topology is refused on the host before
  anything runs, so a kernel argument would be dead code. The fallback, if ever needed:
  a trailing `closed: int` routed through `ops.funcs` helpers that evaluate today's
  integer expressions at `closed=1` (bit-identical by construction). Open topology
  itself is scheduled after 1.0.
- `build(inputs=)` (binding caller arrays as a Program's inputs); `int32` draws, other
  distributions and a public sampler API (the randomizer's private param-writer hook is
  ready; a `params_from=` dataflow binding is the post-0.x direction); `declare()` for
  gate generators; `Track.capabilities()`; a result's back-reference to its Program
  (`result._owner`); `GateSequence` ingestion and `track_from_centerlines`; a `Course`
  over a mixed batch and a `MixedBatch` helper (PR15); dropping `FrameChecker`'s
  `num_envs` (PR14's adapter); `out=` for `localize.curvature` / `speed_profile`
  (PR14).

### Testing

- `tests/test_dynamic_knobs.py` (CPU): the declarations (each Params class's exact
  dynamic set with bounds and `float32` dtype, the schema, the Stage classes and the
  deprecated `.params`); each array-reading kernel against its scalar kernel (defaults,
  non-default values, the bounds, `-0.0`, NaN, per-env arrays row by row, out-of-range
  arrays equal to the scalar kernel at the bound, `_assemble_dyn_k` with unbounded
  bounds against `_assemble_style_k`); each generator with per-env arrays against
  `TrackGenerator(TrackGenConfig(knob=v_e))` row `e` (8 fingerprint seeds) with
  non-vacuity; each Stage's raw output against its legacy `generate_*_warp`; the gate
  generators and elevations (tracks, gates, centerline stations) per env; `randomize=`
  on built-in knobs (env `e` equal to an eager twin set to the drawn values, masked
  advances, the static-field and bounds errors, `PipelineSpec.with_randomize` equal to
  the presets' insertion, a second randomizer on a field refused, the `GateGenConfig`
  route); the cross-field rules and inert-window warnings (configured and randomized
  knobs, a rad draw above a drawn clamp, windows opened by randomizers, gates); the launch
  deltas for every row of the table above (plain and masked, one helper shared with the
  parity tests); bounds and types (each new rejection, the polar migration message,
  `float32` acceptance at a bound, bools including numpy / torch ones, `"1e9"` / 0-d
  numpy / torch coercion, `np.float32` / torch flat knobs against `generator_params`);
  the static-view sentinels for every
  built-in; `style_sampling` (bytes for every range combination, drawn reporting, `set`
  / randomize refusals, `set` on a `None`-range knob); a lock on the generated CUDA
  source of each scalar / twin kernel pair (same arithmetic operations, in order) and
  one on their Python source (the twin's body, once its knob loads, comparison clamps
  and env index are dropped and its locals renamed, is the scalar kernel's body
  statement for statement: operands, operand order, constants and indices; six
  mutants that keep the operation sequence fail it); the
  one-generation lag of a write recorded after segment 0 of a segmented Program; the
  legacy surfaces (`_run_pipeline` per generator, frozen signatures, a `GeneratorSpec`
  with `BezierParams` refused); the kit's `params.per_env` (passes for the built-ins and
  the dynamic toys, skipped without dynamic fields, fails a toy reading `m[0]` or
  writing its parameter array).
- CUDA (`tests/test_dynamic_knobs_cuda.py`, collects on CPU): each twin against its
  scalar kernel on the device; every generator and elevation (non-default values)
  against the legacy scalar path; gates end to end against the frozen PR11 reference;
  `presets.track` / `presets.gates(randomize=...)` on built-in knobs recorded in a
  caller graph, every replay equal to an eager twin set to the replay's draws, masked
  advances; a curriculum kernel writing `params["generator"]["rad"]` and
  `params["elevation"]["z_max"]` (one value out of bounds) in the caller's graph; the
  kit with `params.per_env` on `cuda:0`.
- Existing tests updated where PR13c required it: the launch re-pins above, through one
  delta helper (`tests/_pr13c_delta.py`, also used by the CUDA PR9-sequence test in
  `tests/test_program_capture_cuda.py`); the kit's check lists and per-unit expectations
  (`tests/test_conformance.py`: `params.per_env` runs for the five track and three ring
  gate generators, is skipped elsewhere); `tests/test_randomize.py` (`generator.rad` now
  builds; the static refusal uses `generator.edgy` / `relax.relax_iters` and the new
  message); `tests/test_generator_params.py` (crossers through the Stage scratch, the
  flat hull case's dynamic `hull_displacement`); `tests/test_generator_registry.py`
  (the Stage classes; decoy / override specs built from the frozen functions);
  `tests/test_generate_hull.py` / `tests/test_generate_voronoi.py` (they call the frozen
  `*_alloc_scratch` / `generate_*_warp`); `tests/test_plugin_stage.py` (a legacy
  `GeneratorSpec` described through a toy; `bezier` described as a Stage);
  `tests/test_caller_capture_cpu.py` (the call-counting override of `bezier` is a Stage
  subclass); `.params` -> `.Params` in `tests/test_plugin_params.py` and
  `tests/test_pipeline_planner.py`; the CUDA toys' legacy specs built from the frozen
  functions (`tests/test_caller_capture_cuda.py`, `tests/test_bounded_host_cuda.py`);
  `tests/test_no_host_sync.py` (the private cores through the Stage probes, the frozen
  legacy entry points through the private legacy specs);
  `tests/test_generator_docs.py` (the reference tables' "Kind" column);
  `tests/tools/capture_pr7_fingerprints.py` (crossers from either scratch form). The
  byte guards pass unchanged: `tests/test_generator_fingerprints.py` and its goldens,
  the boundary bytes of `tests/test_pipeline_parity.py` and the taps and frozen
  reference of `tests/test_gate_pipeline_parity.py` (only their launch-sequence
  expectations are re-pinned, above), `tests/goldens/pr11_gate_stages.json`,
  `tests/goldens/pr13a_pr11_relaxed_stations.json` and its test,
  `tests/test_style_sampling.py` and the `tests/test_legacy_plugin_abi.py` pins.
- `tests/test_difficulty_axes.py` (CPU, roadmap PR13d): the declarations (`size` on
  exactly the eight Params classes, last, `[0.1, 10]`; `min_num_points` dynamic `int32`
  on `bezier` / `hull` only; `gate_width` dynamic on `gate_frames`, static on
  `validity`), the flat `size` routes and a default spec dict without it; the public ops'
  `size=` (argument checks, size-1 bytes, clamping, `count` + `size` refused); each
  twin against its original (the count twin in and out of range, the sized grid and
  normalize twins at size 1 and per env against `scale * s`, the frame twin, the
  validity twin against the PR11 kernel with the toggle on and off on real outputs of
  the five gate generators and on posed-like half sizes `{0, +, -, NaN}`,
  `_copy_sized_k` on discriminating payloads: signalling and quiet NaNs, subnormals,
  `±inf`, with a negative control); a lock on the generated CUDA source of each pair
  (equal reduced operation sequences and `mul -> add/sub` edges) and on
  `_copy_sized_k`'s structure; per env equal to per-env scalar runs (size on every
  generator, `bezier` with `scale_range`, `repulsive`, the corner count, size on the
  gate generators and `normalize.size`, `gate_width` with the valid and fail bits of
  pipelines built with each width); `repulsive`'s exact output scaling; size against
  scale; `randomize=` on the new axes against eager twins (plain and masked), the plan
  rules, the truncation and composed-draw warnings; the launch deltas of every pipeline
  through `tests/_pr13d_delta.py` (plain and masked); `in_effect` / `bounds` (latched and
  drawn fields, masked advances, `out=` eagerly and in a caller capture, out-of-range
  writes, segmented Programs); the `style_sampling` warning (once per construction, at
  the caller's line) and the migration's exactness; bounds and types; the static-view
  sentinels; capacity (`repulsive` at size 2 fails closed, a truncating
  `relaxed_stations` source, a `Course` sized with the `n_max` bound under a size
  curriculum); the legacy surfaces and their `ValueError`s; the kit on `repulsive` and
  the `bezier` / `hull` gates; presets `size=`; the hints; posed gates with
  `check_crossing="width"`.
- CUDA (`tests/test_difficulty_axes_cuda.py`, collects on CPU): each twin against its
  original on the device, on the inputs a real generation handed it (the validity twin
  against the PR11 kernel with the toggle on and off, on real gates and posed-like half
  sizes); `_copy_sized_k` against its device contract (finite values bitwise at size 1
  and `f32(v * s)` otherwise, a non-finite point unscaled) and NaN payloads kept
  bitwise (signalling and non-default quiet NaNs; a hard assertion); per env against per-env scalar
  runs (every axis, out-of-range arrays clamped, `repulsive`'s exact output scaling);
  the per-env crossing check against pipelines built with each width;
  `presets.track(randomize={"generator.size": ...})` for every generator and
  `presets.gates(randomize=...)` on `generator.size`, `normalize.size` and
  `gate_frames.gate_width`, recorded with `generate(stream=s, advance=True)` and
  `in_effect(out=...)` in one caller graph, every replay equal to eager twins (crossing
  bits included); `in_effect` under masked regeneration in a caller graph; a curriculum
  writing `size`, `min_num_points` and `gate_width` (values out of bounds included) in a
  Warp capture and a registered `torch.cuda.graph`, every replay equal to the eager twin
  at `clamp(in_effect, *bounds)`; the kit on `cuda:0`.
- Existing tests updated where PR13d required it: the launch re-pins above
  (`tests/test_pipeline_parity.py`, `tests/test_commit.py`, `tests/test_gate_commit.py`,
  `tests/test_gate_pipeline_parity.py`, `tests/test_dynamic_knobs.py`, through
  `tests/_pr13d_delta.py` on `tests/_pr13c_delta.py`); the dynamic-field and
  static-field tables (`tests/test_dynamic_knobs.py`, `tests/test_generator_params.py`);
  the drawable-field lists (`tests/test_randomize.py`: `generator.size` is drawable); the
  kit's expectations (`tests/test_conformance.py`: `params.per_env` now also runs for
  `repulsive` and the `bezier` / `hull` gate generators, so for all five gate
  generators); `tests/test_gate_presets.py` (posed gates with
  `check_crossing="width"` now build); `tests/test_no_host_sync.py` (the frozen
  `generate_repulsive_warp`, the ops' `size=`); `tests/test_gate_stages_cuda.py` (the
  frame twin's name); and a filter for the `style_sampling` deprecation where a test
  still exercises it (`tests/test_style_sampling.py`, `tests/test_pipeline_planner.py`,
  around the capture in `tests/tools/capture_pr7_fingerprints.py`, ...).
  The byte guards pass with their content unchanged:
  `tests/test_generator_fingerprints.py` and its goldens, the PR9 boundary bytes, `tests/test_gate_pipeline_parity.py` (taps
  and frozen reference; only its launch expectations follow the rule) and
  `tests/goldens/pr11_gate_stages.json`, `tests/goldens/pr13a_pr11_relaxed_stations.json`
  and its test, the bytes of `tests/test_style_sampling.py` and the
  `tests/test_legacy_plugin_abi.py` pins.
- `tests/test_results.py` (CPU): `Layout` identity across the three import paths and its
  field rules; every facade / `build` result's layout; schema completeness; `validate`
  parametrized over every field (length, dtype, contiguity), attributes, `need`, a stored
  layout contradicting `count`, the messages; **a hand-built `Track` with one mismatched
  field rejected by `validate()` and by every utility at construction for each field it
  reads**, and accepted by a utility that does not read it; protocols and `check_source`
  (truth table over `Track`, `GateSequence`, `Centerline`, minimal objects without
  `layout`, a `__getattr__` proxy, an object with an unrelated `validate`), utilities on
  `Centerline` / minimal objects byte-equal to the `Track`'s, `CourseLine.centerline` and
  the `.track` warning; `view` (offsets, layouts, attributes, `None` fields, views of
  views, bounds, utilities on views == on clones); `copy_from` (rows, no allocation or
  sync, every error, self-copy, one copy per field); `clone` and `alloc`; `out=` mixed
  batches (bezier + polar halves == standalone generators through plain, advanced,
  masked and poisoned generations; gates; a subset of producer configs through
  `build(out=)` into byte-poisoned parents; default and int seeds on halves == one batch;
  overlap refusals and release, no entry left by a failed warm-up); `out.attributes`
  foreign keys padded on views (and a carried unexported family), refused on a whole
  result; the topology refusals; `check_source` argument checks, tuples of protocols,
  `HasGeneration` alone, an unrelated `layout` attribute, 2-D members; utilities
  binding the checked arrays; `alloc` dtype names and plugin keys; `validate(need=str)`;
  empty `view` ranges.
- `tests/test_randomize.py` (CPU): preset placement and round trip, every planner error,
  draws in range / distinct / deterministic, the Track equal to an eager twin given the
  drawn values, `advance` and masked redraws, the drawn field's `params` behaviour,
  curriculum bounds (a scalar bound past the field's range refused, an array one
  clamped), the launch trace (+2), segments with a HOST generator, gate and
  `width.from_channel` targets, the plugin static-field message, and
  `import track_gen.stages` not loading `track_gen.pipeline`.
- `tests/test_external.py` (CPU): registration and `describe`, `declare()` routing (the
  new toy `acme.declared` and `external` through `presets.track`, `presets.gates` with
  both station sources and `TrackGenConfig`), analytic circles and ellipses, padding
  exactness byte for byte against exact-count runs (tracks and stations gates), the
  kernel order (copies, then validity on the copies), `with_z` / `with_half_width` /
  attributes, device validation and poison, the host loader (forms, `envs=`, messages,
  sizing, declared inputs required, value kinds), `input_info`, the spec's JSON round
  trip by equality, `check_generator("external")` with `diversity` skipped, and the
  hot-path driver.
- CUDA (`tests/test_results_cuda.py`, collects on CPU): two generators writing the halves
  of one `Track` recorded in one caller graph (Warp and torch captures) equal to eager
  twins on every replay, masked and poisoned variants, a gate mixed batch, `copy_from`
  recorded, `clone` / `alloc` refused under capture, the stream rules of `clone`,
  `copy_from` and `alloc` (a fresh `alloc` / `clone` snapshot filled on the producer's
  stream right away, the current stream held busy), randomize and external recorded in a caller graph (an
  in-graph writer of `centerline.xy`, `load_centerlines` ordered after an in-flight
  generation), CUDA layouts given as `"cuda"`, a utility on a view inside a graph, the
  tutorial's mixed-batch step graph, and the kit on `external`.
- Existing tests updated where a decision required it: the exported-key dtype message
  (`tests/test_channels.py`; its unexported key on a whole result keeps the 0.2.0.dev4
  refusal), seed-dependence enumerations filtered on
  `describe(...).seeded`, the kit's check lists (`output.view`), the catalogue kinds,
  `CourseLine.track` -> `.centerline`, the public API list. The frozen guards
  (`tests/test_generator_fingerprints.py`, `tests/test_pipeline_parity.py`,
  `tests/test_gate_pipeline_parity.py` and `tests/goldens/pr11_gate_stages.json`,
  `tests/goldens/pr13a_pr11_relaxed_stations.json` and its test, the
  `tests/test_legacy_plugin_abi.py` pins) are unchanged and pass.

## 0.2.0.dev4 (2026-10-23, tag `v0.2.0.dev4`)

### Added

- **The gate pipeline as stages** (roadmap PR11). `GateGenerator` now runs a `Program`
  built from a `PipelineSpec`, like `TrackGenerator` since PR10a: same public API and
  semantics (`generate` / `enqueue`, masked regeneration and the `generation` counter,
  caller capture, `seeds`, `inputs`, `params`, `graph`), same `GateSequence` fields, and
  byte-identical outputs for every built-in gate generator (the ten gate fingerprint
  variants are unchanged). The default stage list is `generator` (anchors) -> `order`
  -> `normalize` (`polar`, `checkpoint`, `voronoi`) -> `gate_relax` -> `gate_arclen`
  (non-flat profiles) -> `elevation` -> `lift` -> `tangents` -> `gate_frames` ->
  `validity`, then the commit. `docs/how-it-works/pipeline-preview.rst` gains "The gate
  pipeline".
  - **`BatchSpec.max_gates`** (`None` default: a track pipeline; an int `>= 2`: a gate
    pipeline whose `Program.result` is a `GateSequence` with that many slots per env).
    `to_dict` writes it and `from_dict` reads it when present (format unchanged).
    `build(spec)` accepts both kinds and binds the generator from the first stage's kind.
  - **`legacy.to_gate_pipeline_spec(config, *, registry=None)`**: the default gate spec
    for a `GateGenConfig`, the one `GateGenerator(config)` builds.
  - **`presets.gates(*, num_envs, device, generator="bezier", generator_params=None,
    source="auto", gates=None, max_gates=None, min_gates=4, order=None, stations=None,
    gate_radius=0.025, gate_solve_iters=8, gate_width=0.0, gate_align="yaw_only",
    elevation="flat", z_base=0.0, z_valid_grade=0.0, checks=("gate_default",),
    scale=1.0, on_invalid="keep", n_max=384, num_points=256, half_width=0.1,
    spacing=None, relax="xpbd", registry=None)`**. Unlike
    `presets.track` it reads the registry (the generator's `output`, `provides` and
    orderings decide the stage list). `presets.gates(num_envs=E, device=d)` builds
    byte-identical to `legacy.to_gate_pipeline_spec(GateGenConfig(num_envs=E,
    device=d))`. `order=None` is the generator's default order (its preferred order
    when it accepts it, else `"ccw"` when it accepts it, else the first order it
    accepts: the rule of `GateGenConfig.gate_ordering=None` and of the conformance kit
    too). `z_base` is `elevation.flat`'s altitude
    (with another elevation or a `z` provider a nonzero value raises, as
    `presets.track(z_base=)` does). `validity.gate_default` in `checks` gets
    `min_gates`, `gate_radius`, `gate_width`, `z_valid_grade` (and
    `check_crossing="always"` for posed gates) unless its params set them, and params
    setting another `gate_radius` / `gate_width` raise `ValueError` at preset time.
    `gates=` fills the station count of a `stations` selection that does not set one
    and must agree with one that does. The centerline arguments (`n_max`, `num_points`,
    `half_width`, `spacing`, `relax`) raise `ValueError` when set for a source that does
    not read them (an anchors source reads none, `"stations"` only `n_max` and
    `num_points`). A `validity.gate_default` `gate_width` for a generator that provides
    `half_size` raises the provider error (it must stay 0).
  - **`GateGenerator(config_or_spec, rng=None, *, seeds=None, stream=None)`** accepts a
    gate `PipelineSpec`; `rng` is optional (env `e` gets seed `e`), `seeds=` takes any
    `SeedState` form (`rng` and `seeds` together raise `ValueError`). New
    `gen.program`, `gen.num_envs`, `gen.device`, `gen.max_gates`; `gen.config` returns
    what was passed. A track spec raises `ValueError` (use `TrackGenerator`), and
    `TrackGenerator` refuses a gate spec the same way, before binding anything.
  - Two point domains: anchors (`per="anchors"`, `[E*A]`, `A` = the generator's
    `capacity()["gates"]`) and gate slots (`per="gates"`, `[E*G]`, `G = max_gates`).
    Each stage gets its own `Layout`, derived from its ports: `b.layout.stride` is `G`
    on gate slots and `N_max` for a centerline sub-pipeline. New
    **`StageBuilder.stride_of(port)`** and **`StageBuilder.per_of(port)`** (the stride
    and `per` of the channel a port is wired to); `temp_like` works on `anchors.*` and
    `gates.*` ports.
  - Channels: `anchors.xy`, `anchors.count`, `anchors.z`, `anchors.tangent`,
    `anchors.half_size`, `anchors.roll`; `gates.xy`, `gates.count`, `gates.cum`,
    `gates.perim`, `gates.z`, `gates.position`, `gates.tangent`, `gates.forward`,
    `gates.left`, `gates.right`, `gates.orientation`, `gates.half_size`, `gates.roll`,
    `gates.valid`; `valid.fail` is shared with tracks. The final version of each
    exported channel is the `GateSequence` array; `Program.tap` reads the others
    (`tap("anchors.xy")`, `tap("gates.xy", after="order")`, ...).
  - Built-in gate stages, in TrackGen's private catalogue: `normalize.extent`,
    `carry.{z,tangent,half_size,roll}`, `gate_relax.{spheres,none}`,
    `gate_arclen.default`, `lift.gates`, `tangents.central`,
    `gate_frames.{default,posed}`; and registered: `validity.gate_default`,
    `order.{raw,ccw,random_pairs}`, `stations.{raw,relaxed}`. Each launches the
    0.2.0.dev3 kernels with the same arguments; `_relax_gate_spheres_k`,
    `_finalize_frame_k` and `_finalize_validity_k` gained trailing array arguments, and
    `_finalize_validity_k`'s grade-check wrap is also bounded by `max_gates` (the same
    arithmetic for every count <= max_gates; see Behaviour changes).
- **Gate generators are `output="anchors"` Stages.** A `Stage` with
  `kind="gate_generator"` and `output="anchors"` writes `anchors.xy` and
  `anchors.count`, and does not order, normalize or lift anything. The five built-ins
  (`bezier`, `hull`, `polar`, `checkpoint`, `voronoi`) are ported (same modules, so
  `describe().origin` is unchanged) and no longer call `finish_ordered_gates`.
  - **`capacity(params, layout) -> {"gates": A, "min_gates": k}`** replaces the
    overloaded `GateGeneratorSpec.max_gates` callable: `A` is the most anchors per env
    (their row stride), `k` the fewest. The planner checks `A <= BatchSpec.max_gates`
    and the validity stage's `min_gates <= A` with the `GateGenConfig` messages.
  - **`provides`**, a subset of `{"z", "tangent", "half_size", "roll"}` naming exactly
    the `anchors.*` attributes written (`half_size` with `roll`). The preset drops the
    stages a generator provides (`z`: `gate_arclen` and `elevation`; `tangent`:
    `tangents`) and inserts `carry.<attr>` stages that copy the values to the gate
    slots by anchor index; `half_size` + `roll` select `gate_frames.posed` and
    `validity.gate_default(check_crossing="always")`. A provided `z` reaches
    `GateSequence.position.z` and a provided `half_size` `GateSequence.half_size` bit
    for bit; a provided tangent is normalized. The planner refuses a provided attribute
    that no stage reads, and a `carry` stage after an order other than `order.raw`;
    the presets and `GateGenConfig` refuse an explicit elevation (or `z_base`) for a
    `z` provider and a `gate_width` for a `half_size` provider.
  - **`Stage.orderings`** (a whitelist, or `None`), **`Stage.excluded_orderings`** (with
    `orderings=None`: every registered order but these) and **`Stage.preferred`**
    (`{"order": ..., "normalize": ...}`), reported by `SpecInfo`, plus the resolved
    **`SpecInfo.supported_orderings`** (sorted; `()` for other kinds), the public way to
    list a gate generator's orders (`viz/param_explorer.py` uses it). The built-ins exclude
    instead of whitelisting (`bezier`/`hull` exclude `raw`; `polar`/`checkpoint`/
    `voronoi` exclude `random_pairs`), so a third-party order runs with all five; with
    only the built-in orders installed the supported lists are 0.2.0.dev3's. A
    generator that provides attributes must declare `orderings=frozenset({"raw"})`.
  - Registration rules for anchors Stages (each a `ValueError`/`TypeError`, quarantine
    under `warn`), replacing PR10b's "arrives with roadmap PR11" errors: `output` must be
    `"anchors"` (and a `kind="generator"` Stage with `output="anchors"` is told to
    declare `kind="gate_generator"`), `reads` empty, `writes` within the anchors channels
    and including `anchors.xy` / `anchors.count`, `provides` equal to the written
    attributes, `orderings` / `excluded_orderings` / `preferred` well formed and not
    both a whitelist and exclusions, `preferred` only on gate generators.
- **The ordering registry**: the open kind `order`, with the built-ins `order.raw`,
  `order.ccw` and `order.random_pairs` (0.2.0.dev3's kernels and salt). An order stage
  writes `gates.xy` and `gates.count` (both required). `GateGenConfig.gate_ordering`
  and `presets.gates(order=...)` name any registered order the generator accepts;
  unknown names raise with the registry's list (`GateGenConfig.gate_ordering: unknown
  order 'spiral'; available: ['ccw', 'random_pairs', 'raw']`), an unsupported one
  `gate generator 'polar' does not support gate_ordering 'random_pairs'; supported:
  ['ccw', 'raw'] (its preferred order is 'ccw')` (the hint only when the generator
  declares a preferred order). The default order is one rule
  (`stages.gates._default_order`): the generator's `preferred["order"]` when it accepts
  it, else `"ccw"` when it accepts it, else the first order it accepts.
  `presets.gates(order=None)`, `GateGenConfig.gate_ordering=None` (the new default, see
  Behaviour changes) and the conformance kit apply it.
- **`stations`: gates from any track generator.** `presets.gates(generator=<a track
  generator>, gates=G)` inserts `stations.raw` (equal arc-length anchors on the raw
  loop `raw.xy`), or with `source="relaxed_stations"` the first three stages of
  `presets.track` and `stations.relaxed` (on the relaxed `cl.xy`), then the shared
  gate stages. `source="auto"` picks the gate generator when the name is one (the
  built-in names are both kinds), else stations. Works for every built-in track
  generator, `repulsive` included, and for third-party ones (Stage or
  `GeneratorSpec`). An env whose generator flagged it invalid, whose loop was
  truncated, or whose length is not positive gets no anchors (fail bit `count`).
  `stations` and `gate_arclen` measure plan-view arc length (0.2.0.dev3's); PR13a adds a
  3D metric to the stations (`spacing_metric`, below; see Deferred for `gate_arclen`).
- **Elevation on gates through port rewiring.** The planner rewires every `elevation`
  slot of a gate pipeline (`cl.xy -> gates.xy`, `cl.count -> gates.count`,
  `cl.arclen2 -> gates.cum`, `cl.length2 -> gates.perim`, `cl.z -> gates.z`; explicit
  `StageRef.ports` override), so the built-in profiles and plugin elevation stages
  (PR10b's `acme.sinusoid_hills` included) run on gates unchanged, selected by
  `GateGenConfig.z_profile`, `presets.gates(elevation=...)` or `spec.replace(...)` (on
  a spec whose elevation is already non-flat; a flat gate spec has no `gate_arclen`
  slot, so also `insert_before("elevation", StageRef("gate_arclen", "default"))`, as the
  `PlanError` says). No
  monkeypatch, name tuples or fall-through to noise. `elevation.flat` gains
  `z_base` (gate altitudes; on a track a non-zero value raises `PlanError` before
  anything is allocated, and `ValueError` from `presets.track`: `lift.flat`'s `z_base`
  sets a track's flat altitude) and a `count` read port. On gates, `uniform` /
  `random_walk` draw one altitude per gate (the gates are the knots:
  `z_control_points` is not read). A plugin's `Params.from_legacy(config)` receives the
  `GateGenConfig` through `GateGenConfig.z_profile`; one that reads a
  `TrackGenConfig`-only field raises `ValueError` naming the field.
- **Gate fail bits.** `validity.gate_default` keeps the fused `_finalize_validity_k`
  (`GateSequence.valid` byte-identical) and writes one bit per check into
  `valid.fail`: 0 `count`, 1 `finite`, 2 `tangent`, 3 `spacing`, 4 `crossing`,
  5 `grade` (6-15 reserved, 16-30 plugin checks via `StageBuilder.fail_bit()`).
  **`track_gen.pipeline.GATE_FAIL_BITS`** is that table. New
  **`GateSequence.fail_mask`** (`[E]` int32; allocated by `GateGenerator` and
  `warp_gate.alloc_gate_sequence`, `None` on a `GateSequence` built by hand) and
  **`GateSequence.fail_bits`**; `valid == (fail_mask == 0)`; `clone()` clones
  `fail_mask`. `validity.gate_default` takes `check_crossing="width"` (default, as
  before: the crossing check runs when `gate_width > 0`) or `"always"` (posed gates,
  whose widths come from the generator). `warp_gate.finalize_gate_sequence` gains
  `fail=None` (`None`: the bits go to `gates.fail_mask`, so the pre-PR11 call with a
  persistent `fallbacks` still allocates nothing).
- **The commit for gates.** PR10b's hold / restore commit runs over the
  `GateSequence`: masked regeneration of gates is exact by construction (unmasked envs
  keep their bytes), and `BatchSpec.on_invalid="poison"` (`presets.gates(on_invalid=)`)
  leaves an invalid env with `count = 0`, every per-gate field NaN and `valid = 0`,
  keeping `fail_mask`. Cost: none for an unmasked default generation, one launch for a
  derived `valid` or poison, two for a masked generation; the held copy is 80 B per
  gate slot and 12 B per env.
- **Conformance kit.** `check_gate_generator` runs anchors Stages and legacy specs alike;
  its raw output is `tap("anchors.xy")`, `output.count` also checks every env's
  `anchors.count` against `capacity()`, and the default ordering is the generator's
  preferred one when it accepts it. `check_stage` accepts the kinds `"order"` and
  `"stations"` (run in a `presets.gates` stations pipeline) and runs a validity stage
  that reads gate channels in the default gate pipeline. **`check_stage(...,
  pipeline="gates")`** runs an `elevation` stage on gates (`presets.gates(elevation=)`,
  ports rewired), so a plugin elevation is checked in both pipelines. `CHECKS`, the
  pytest plugin's `track_gen_conformance(kind=...)` and `generator_names(kind)` list the
  new kinds.
- **`StageRef` (hence `PipelineSpec`) pickles**: `StageRef.__reduce__` rebuilds the ref
  from plain dicts (the frozen `MappingProxyType` fields did not pickle; pre-existing);
  a frozen mapping nested in the params (another ref's params) comes back as a dict.
- Docs: `docs/contributing/writing-a-stage.rst` gains "Writing a gate generator
  (anchors)" (a helix that provides its altitudes and tangents) and "Writing an
  ordering" (a reversed order); the configuration reference documents `gate_ordering`
  / `z_profile` as registered names and the capacity / `max_gates` / `min_gates`
  relation; the generator guide, the conformance-kit guide, the gate tutorial,
  `docs/gates-3d.rst` and `docs/relaxation/gates.rst` describe the gate Program.
- **Iterative stages: the `BOUNDED` capture class** (roadmap PR12).
  **`LaunchContext.bounded_loop(n_iters, body, *, active=None, first_check=0,
  check_every=1)`** runs `body(ctx, it)` for `it in range(n_iters)`, `it` a static Python
  int. Iterations `[0, first_check)` always run; the rest form windows of `check_every`
  iterations (the last one may be shorter), each preceded by a check of `active[0]`, a
  `[1]` int32 flag the body writes on device on the iteration before the window; a
  window whose check reads 0 may be skipped, with every later one. The lowering is chosen
  once per call from the capture state, never from data: (a) a host poll that stops the
  loop (`cpu`, eager, no CPU graph capture open on the device); (b) one `wp.capture_if`
  conditional node per window, through `skip_unless`, inside a Warp-registered capture
  of the context's stream with conditional-node support (CUDA 12.4+: a producer's own
  graph, `wp.ScopedCapture`, `registered(conditional=True)`); (c) every window
  unconditionally everywhere else (eager `cuda` calls, plain `torch.cuda.graph`, older
  drivers, CPU `wp.ScopedCapture("cpu")`, construction and `prepare()`). Under the
  documented contract (once `active[0]` reads 0, every remaining iteration leaves
  everything read after the loop bitwise unchanged: an absorbing freeze) the three are
  bitwise identical, and `n_iters` is a bound. Arguments are checked on the host before
  any launch (`TypeError` for a non-int or a bool; `ValueError` for a negative count or
  `check_every < 1`; `active` as `skip_unless` checks its flag). `active=None` is a plain
  unroll, allowed in any stage.
  - `capture = CaptureClass.BOUNDED` registers (stages of every open kind and both
    generator kinds), plans and binds; it was rejected with "arrives with roadmap PR12".
    A Stage that calls `bounded_loop(active=...)` while declaring `STATIC` raises
    `ValueError` on its first run, at construction (`... declare capture =
    CaptureClass.BOUNDED`); a legacy `GeneratorSpec` cannot declare a class and is not
    checked. The built-in catalogue stages stay `STATIC`.
  - The host poll is the library's one sanctioned eager readback on a capturable path:
    the `track_gen.testing` host-sync guard lets it through only when the loop's window
    driver makes it, and records it in the new `HotPathGuard.sanctioned` (not in
    `allowed`), as it does a `HOST` stage's `HostOps` operations (see the conformance-kit
    entry). Any other sync is still a violation: a plugin's own, inside a loop body or a
    `HOST` stage, and a direct call of the poll function included.
- **`ops.all_set(ctx, mask, flag_out)`**: `flag_out[0] = 1` if every `mask[e] != 0` (an
  empty mask: 1), else 0. Two launches (a fill, then an integer `atomic_min`),
  deterministic, with `any_set`'s validation and messages.
- **`repulsive` is a `BOUNDED` Stage.** `registry.get("generator", "repulsive")` returns
  a `Stage` class (`kind="generator"`, `output="centerline"`, `Params = RepulsiveParams`,
  `capacity() -> {"raw": repulsive_stages[-1]}`), and `describe(...).capture` is
  `BOUNDED` (`capturable` stays `True`; `origin` is unchanged). Its growth loop is a
  `ctx.bounded_loop` with `active = any env not frozen`, written right after each stall
  check, `check_every = repulsive_stall_window` (`W`) and `first_check = W * ceil((s +
  1) / W)`, `s` the iteration of the last stage transition: 128 at the defaults, 9
  windows in a 269-iteration bound. Same kernels, order and scalars as before; the new
  launches (a reset of `active` in the prologue and a refresh after each stall check, 19
  at the defaults) write only the flag. `generate_repulsive_warp(seeds_wp, config,
  out_centerline, out_valid_wp, scratch, ctx=None)` keeps its signature and stays
  importable (the Stage calls it; it derives the windows from the scratch's stage
  schedule and the generate-time `repulsive_stall_window`), `repulsive_alloc_scratch`
  gains an `alloc=` hook (the Stage allocates through `b.scratch`) and
  `RepulsiveScratch` an `active` flag.
- **HOST stages reach the host through `HostOps`.** **`LaunchContext.host()`** returns
  the `track_gen.exec.HostOps` of the `HOST` stage running on the context:
  `read_i32(arr, index=0)`, `readback(src, *, via)` (a NumPy view of `via` shaped like
  `src`), `upload(dst, data, *, via)` (blocking: `via` is reusable on return) and
  `sync()`. Every copy runs on the call's stream through a host buffer allocated at build
  with **`StageBuilder.host_buffer(dtype, shape)`** (pinned on CUDA; `HOST` stages only,
  `ValueError` otherwise), and every wait is a stream synchronize in relaxed capture
  mode: nothing allocates per call. On `cuda` a `via` that is not pinned raises
  `ValueError`. `ctx.host()` raises `ValueError` inside a stage that does not declare
  `HOST` (naming the stage and its declared class, the error `bounded_loop(active=...)`
  in a `STATIC` stage and `host_buffer` give too), `CaptureError` outside every pipeline
  stage, and `NotCapturableError` when the context's stream is capturing or a CPU graph
  capture is open on the device. The Program runs every stage's `enqueue` inside a
  per-slot context (private) that carries this and the `bounded_loop` declaration check;
  it emits nothing. Only a `HOST` slot gets a build-time `[1]` host buffer (`read_i32`);
  the kit's private forced CUDA poll allocates its pinned buffer on first use.
- **`track_gen.exec.unconditional()`**: a context manager (thread- and task-local) under
  which `LaunchContext.skip_unless` records no conditional node, so every
  `bounded_loop` window (and a masked `Course.regenerate`) is recorded unconditionally
  into the captures made in the block, with the same results. Use it for a graph you
  will nest as a CUDA child graph, or for a capture whose stream has a side stream that
  joined it and is not joined back yet (each `wp.capture_if` ends and resumes the
  capture, which CUDA refuses with an unjoined fork). A producer's own graph captured in
  the block is unrolled too; `unconditional(False)` restores the default inside it.
- **Segmented execution of HOST stages.** The planner groups consecutive stages into
  segments of equal host-ness (`HOST` vs. `STATIC` / `BOUNDED`). New
  **`Program.segments`** (a tuple of **`track_gen.pipeline.Segment`**: `index`,
  `capture_class`, `stage_ids`, `host`), **`Program.host_stage_ids`** and
  **`Program.enqueue_segment(i, ctx, *, advance=False, mask=None)`** (segment 0 opens
  with the seed prologue, the last segment closes with the commit and the `generation`
  write; a `HOST` segment refuses a capturing `ctx`, a capturable one can be recorded
  into a caller's capture once the Program is prepared). `Program.capture_class` is the
  highest of its stages' (`STATIC` < `BOUNDED` < `HOST`), and `Program.capturable` means
  "has no `HOST` stage". Outside a capture on `cuda`, `run()` (hence `generate()`)
  records each capturable segment into a graph of its own on the first call and then
  runs the segments in order on the call's stream: `HOST` segments eagerly, the others
  with `wp.capture_launch` (keyed per `(advance, mask)` like a plain self-graph, within
  `MAX_KEYED_GRAPHS`). `graph` stays `None` for such a Program. Under a caller's capture,
  `run`, `enqueue` and `enqueue_segment` of a `HOST` segment raise
  **`NotCapturableError`** naming the `HOST` stages before any launch: the message lists
  them and the new **`NotCapturableError.stage_ids`** holds them. Gate pipelines are
  segmented the same way. **`Program.segment_graphs(key=None)`** (and
  **`GraphRunner.segment_graphs_for(key=None)`**) return the per-segment self-graphs
  (`None` at a `HOST` segment); `GraphRunner.graph_for(key)` stays the whole-body graph
  (`None` for a segmented key). Every segment of one generation takes the same
  `advance` / `mask`, in order, on one stream (a recorded segment keeps the arguments it
  was recorded with); that is documented, not checked (a replayed segment runs no
  Python). With a legacy-ABI generator (`legacy_stream`), `enqueue` and
  `enqueue_segment(0, ...)` on a stream other than Warp's current one run on the
  current stream ordered with events (the runner's legacy path), and raise
  `LegacyPluginStreamError` before any launch when that stream is capturing.
- **`PipelineSpec(batch, stages, require_caller_capturable=False)`**: `True` makes
  `build` refuse every `HOST` stage, host-side, before allocating anything, with a
  `PlanError` naming them (`PipelineSpec.require_caller_capturable=True, but stage(s)
  'generator' (generator.acme.blob) is HOST (...): use STATIC / BOUNDED stages or drop
  require_caller_capturable`; a legacy spec is named `'generator' (generator 'acme.old',
  GeneratorSpec capturable=False)`). A non-bool raises `TypeError`. `to_dict` writes the flag only when
  it is set (existing dicts are unchanged) and `from_dict` reads it; the edit operations
  and the resolved `Program.spec` keep it.
- **Conformance kit.** New checks for every kind, after `statelessness`:
  `bounded.absorbing` (each generator or stage that runs a `bounded_loop` with `active`,
  as the probe sees it, whatever it declares: with `active` and the outputs poisoned, an
  eager generation with the host poll equals one with every window unrolled, raw output,
  result and the stage's channels, for both batches; "not applicable" without such a
  loop, "not exercised" when the kit's seeds skip no window), `bounded.lowering` (CUDA:
  an eager generation with the forced host poll, and replays of a Warp capture with
  conditional windows and of one with every window unrolled, into poisoned outputs and
  before and after a reseed, all equal the eager batch; the conditional part skipped
  with the reason without conditional-node support) and
  `host.require_caller_capturable` (a pipeline with `HOST` stages rebuilt with the flag
  raises `PlanError` naming every `host_stage_ids`). The `hot_path.*` calls run with
  every window unrolled (the superset of what any lowering launches), plus one call with
  the default lowering whose sanctioned polls `hot_path.host_sync` reports in
  `details["sanctioned"]`. A pipeline with `HOST` stages waives `hot_path.topology` and
  `hot_path.buffers` only: `hot_path.host_sync` and `hot_path.allocation` still run, with
  the `HostOps` operations sanctioned (`HotPathGuard.sanctioned`,
  `details["host_ops"]`), so a `HOST` stage that calls `.numpy()`,
  `wp.synchronize*()` or allocates per call itself fails them; a legacy
  `capturable=False` spec (no `ctx.host()`) still waives all four. For a `HOST`
  generator, `capture.caller` checks the refusal
  (`NotCapturableError` naming the `HOST` stages before any launch, the caller graph
  holding a marker and nothing of the generation, seeds and epochs untouched) and
  `capture.self_graph` the segmented self-graph (a graph exactly for each capturable
  segment, replays equal the eager batch). `check_generator("repulsive")` exercises
  `bounded.absorbing`.
- Docs: `docs/how-it-works/cuda-graph.rst` gains "Iterative stages (BOUNDED)" (the
  lowerings, where each applies, the absorbing-freeze contract, the sanctioned poll) and
  "HOST stages and segments" (segmented execution, refusals, `HostOps`, concurrent
  captures, the lock rule); `docs/how-it-works/pipeline-preview.rst` "Capture classes
  and segments" (`Program.segments`, `enqueue_segment`, refusal order,
  `require_caller_capturable`); `docs/contributing/writing-a-stage.rst` "Writing an
  iterative (BOUNDED) stage" (a relax stage that freezes converged envs) and "Writing a
  host-driven (HOST) stage" (a NumPy generator through `HostOps`); the repulsive page
  "Early exit and capture" (schedule, windows, node counts); the generator and kit
  guides, the tutorials and the API reference follow.
- **Per-point channels carried through the resamplers** (roadmap PR13, part a). A
  centerline generator may write per-point attributes, `raw.<package>.<name>` channels
  (`float32`, `int32`, `vec2f` or `vec3f`), declared at the write with an interpolation
  policy: `b.write(port, dtype, per="raw", interp=..., export=...)`. The
  constant-spacing resample carries every raw channel to its `cl.*` image, the uniform
  resample re-interpolates every carried `cl.*` channel onto the final points, and the
  stations carry them on to the anchors. Policies: `"linear"` (floats and vectors; the
  resamplers' own expression, so a carried copy of the source xy reproduces the points
  byte for byte), `"nearest"` (ties to the segment start), **`"previous"`** (a
  zero-order hold, for per-segment labels) and `"angle"` (`float32` radians, the short
  way round, output in `[-pi, pi)`); `int32` takes `nearest` / `previous`. Padding is
  NaN (`-1` for `int32`). The carry sets are static (decided from the declared writes
  and reads before anything is allocated), the carriers are ordinary writers
  (`Program.channels`, `tap(after=...)`), and a pipeline that carries nothing launches
  what 0.2.0.dev3 launched (the default pipelines are byte-identical, launch for
  launch). Cost: two launches per raw attribute and generation, plus the resample tables
  (12 B per point per carrying resampler).
  - **Attributes through relaxation are bead-attached** (open question 7 of the target
    architecture): a relax stage moves points without re-indexing them, so a carried
    channel keeps its bead index through relax (no launch, no new version) and the
    uniform resample re-interpolates it with the relaxed loop's tables. The half-width
    XPBD separated a point with is the one the band gets there, and labels stay with
    their points. Documented as the relax contract ("point `i` out is point `i` in,
    moved"); the kit does not check it.
  - Stage-written `cl.*` plugin channels are not carried: reading one across the
    uniform resample (written before it, read after it) raises `PlanError` naming the
    writer and the fix (write it after `uniform`, or carry it from the generator).
  - **`ops.ResampleTables`** (`i0`, `i1`, `frac`, `n_src`; `ResampleTables.alloc(n_out,
    n_src, device)` fills the sentinels `-1` / NaN), **`ops.resample_channels(ctx,
    tables, pairs, interp)`** (one launch per pair, host-checked: lengths, dtypes,
    policies, no aliasing) and a `tables=` keyword on `ops.resample_constant_spacing` /
    `ops.resample_uniform` (a lookup variant that also writes the tables, the sentinel
    on every padding slot; the points are byte-equal to the plain lookup's).
- **`StageBuilder.write(port, dtype=None, *, per=None, interp=None, export=False)`** is
  implemented (it raised `NotImplementedError`), and **`StageBuilder.dtype_of(port)`**
  joins `per_of` / `stride_of`. `export=True` (or the key itself; renaming is not
  supported) makes a per-point plugin channel a result attribute; `interp=` is required
  on a generator's `raw.<package>.<name>` writes and refused elsewhere. Plugin channels
  generalise to four domain prefixes, `raw.`, `cl.`, `anchors.`, `gates.` +
  `<package>.<name>`, the prefix fixing the domain (and `per`, for `raw` / `anchors`).
- **The width registry and per-point half-width.** `cl.half_width` is the band's
  half-width per point and its only source: the width stage writes it before `relax`;
  `relax.xpbd` separates the beads with it, the uniform resample carries it to the final
  points, `offset.default` builds the band from it, `validity.default` checks thickness
  and the floor against it. New **`width.from_channel`** (`scale`, `lo`, `hi`, dynamic;
  `min(max(scale * src, lo), max(hi, lo))`: `lo <= hi` is checked at build, a run-time
  `hi` below `lo` clamps to `lo`; NaN kept) turns any per-point `float32` channel into
  the width, wired through a required read port. **Constant width stays bit-exact**:
  `width.constant` declares its value constant, and a consumer that reads a constant
  width runs the unchanged 0.2.0.dev3 scalar kernels; any other width selects per-point
  kernels at build (`track_gen/_src/warp_width.py`: the XPBD solve with per-bead radii
  `r_i = w_i (1 + margin)`, pair targets `r_i + r_j`, the band offset, the per-pair /
  per-point thickness test). `width.constant` launches nothing (its buffer is filled
  when allocated), and the uniform resample's fill of the final `cl.half_width` is
  0.2.0.dev3's width launch at its old trace position.
- **The `w_floor` check is real** (fail bit 4): an env fails when a real final point's
  half-width is not above `w_floor` (NaN included). It compared a constant fill with
  `w_floor` before (finding A128); at constant width it still gives that answer.
- **Generators may provide their altitude and half-width.** A centerline generator
  writes `raw.z` / `raw.half_width` (`float32`, carried `linear`) and declares
  `provides={"z"}` / `{"half_width"}` (exactly what it writes). The carries write
  `cl.z` / `cl.half_width`; `presets.track`, `presets.gates` and
  `legacy.to_pipeline_spec` drop the elevation (and pick `lift.elevated`) or the width
  stage, and refuse a non-default selection of what the generator provides. The planner's
  second-writer error names the provider for a hand-built spec.
- **`Track.half_width`** (`[E * N_max]` float32: the band's half-width on the real
  points, NaN padding; the constant at constant width), written by `offset.default`
  (channel `track.half_width`), and **`Track.attributes`** / **`GateSequence.attributes`**
  (read-only mappings from key to the final, pointer-stable `[E * stride]` array of each
  exported per-point plugin channel, NaN / `-1` padding). Appended after `fail_bits`, so
  positional construction keeps working; `clone()` copies them. The commit holds and
  restores them under a masked regeneration and poisons them; `out=` binds the keys the
  caller supplies (a key the Program does not export raises `ValueError`).
  **`Program.attributes`** maps each exported key to a
  **`track_gen.pipeline.AttributeInfo`** (`key`, `channel`, `dtype`, `interp`,
  `padding`, `domain`). Gate attributes are best effort: a family with no index path to
  the gates (a non-`raw` order, a plugin stations stage, a channel a stage writes on the
  centerline such as an elevation run with `elevation_on="centerline"`) is left out, and
  asking for it through `out.attributes` raises.
- **3D station spacing.** **`StationsParams.spacing_metric`** (`"xy"`, the default and
  PR11's bytes, or `"xyz"`: equal 3D arc length, accumulated in float64) and
  **`presets.gates(spacing_metric=, elevation_on=, width=)`**. The altitude comes from
  a generator that provides `z`, or with `source="relaxed_stations"` from an elevation
  run on the relaxed centerline before the stations (`elevation_on="centerline"`, the
  default for `"xyz"`); the stations carry it to `anchors.z`, `order.raw` keeps the
  index, `carry.z` copies it, and the gate-level `gate_arclen` / elevation are dropped.
  `elevation_on="gates"` (the default for `"xy"`) keeps the PR11 stage lists, byte for
  byte. Impossible combinations raise `ValueError` from the preset, and an `"xyz"`
  stations stage on a loop without altitude raises `PlanError`. `gate_arclen` never
  gets a `spacing_metric`: its arc table feeds the gate elevation that produces z, so a
  3D metric there would be circular (it stays plan-view; see Deferred for the PR11
  entry).
- **Gate plugin channels.** Gate-domain stages read and write `gates.<package>.<name>`
  channels and export per-gate ones to `GateSequence.attributes`. Gate generators may
  write `anchors.<package>.<name>` attributes (with `orderings=frozenset({"raw"})`),
  stations stages too; the new catalogue stage **`carry.attributes`**, inserted by
  `presets.gates` after an `order.raw`, copies every one to `gates.<package>.<name>` by
  index. The planner rewires an elevation stage's plugin channels on gates by prefix
  (`cl.<package>.<name> -> gates.<package>.<name>`), and an `anchors.*` attribute
  written by a gate generator or a plugin stations stage must be read (`carry.z` /
  `carry.attributes`), else `PlanError`. The PR11 `PlanError` "gate plugin channels
  arrive with roadmap PR13" is gone; a stage on the wrong domain's plugin channel gets a
  targeted one.
- **Required read ports.** A `reads` value may be a domain, `"cl.*"` or `"gates.*"`: the
  user wires it with `StageRef.ports` to a concrete channel, and an unwired one raises
  `PlanError` with the form to write; `registry.describe(...).reads` shows it as is
  (`{"src": "cl.*", ...}`). `width.from_channel`'s `src` is one.
- **`presets.track(registry=None)`**: the preset reads the registry to see what the
  generator provides and whether the width stage has a `half_width` field (unknown names
  are left to `build`). `half_width=` fills `width.constant`'s value, or the
  `half_width` field of a plugin width stage, and keeps checking an explicit value
  against it; a `half_width` other than the default 0.1 that nothing reads (a width
  stage without the field or a half-width provider, and an explicit `spacing`) raises.
- **Spec format `track_gen.pipeline/2`.** `to_dict()` writes it; `from_dict()` reads
  `/1` dicts (0.2.0.dev3, and v2 before PR13a, gate specs included) and migrates them on
  the data (the three `half_width` keys checked and removed, the width stage moved
  before `relax`, a `width.constant` inserted before a `relax.xpbd` that had none), with
  one `DeprecationWarning`. With `width.constant` the migrated spec builds the bytes of
  the `/1` spec it came from; a spec with another width
  stage migrates but its outputs change (see "Changed (outputs)"; the warning names
  the stage), and a dev3 width stage after `frames` with no width before `relax.xpbd`
  raises `ValueError` (make it a rewrite of `cl.half_width`, or move it before relax).
- **Conformance kit.** `check_stage(..., ports=)` wires required ports; new checks
  **`width.values`** (a width stage's own `cl.half_width` version is finite and `>= 0`
  on every real point), **`attributes.written`** (every slot of every raw / anchors
  attribute written) and **`attributes.carried`** (exported attributes padded, finite,
  and within what their policy can produce from the env's raw values); determinism,
  batch independence, statelessness and the caller-capture checks cover
  `Track.half_width` and every attribute (`testing.capture.result_arrays` lists them).
- Docs: `docs/how-it-works/pipeline-preview.rst` gains "Carried channels", "Required
  read ports", "Width", "Generator-provided altitude and width", "Specs written for
  0.2.0.dev3", "Per-point attributes in the result", "3D station spacing" and "Gate
  plugin channels"; `docs/contributing/writing-a-generator.rst` "Per-point attributes"
  and "Providing altitude and width"; `docs/contributing/writing-a-stage.rst` "Writing a
  width stage" (writer, rewrite, from a channel, geometry-dependent widths), "Reading
  cl.half_width" and the relax bead-identity contract; the configuration reference "Road
  width: half_width, w_floor and the width stage"; `docs/tracks-25d.rst` "Altitude from
  the generator" and `docs/gates-3d.rst` "Gates on a track centerline: 3D spacing"; the
  kit guide, the API reference, the inflation and relaxation-constraint pages and
  `docs/library-readiness/target-architecture.md` (as-shipped notes, open question 7
  answered, the PR13a / PR13b split and the later list) follow.

### Behaviour changes

- **`GateGenConfig.gate_ordering` and `GateGenConfig.z_profile` are resolved through the
  registry** at `GateGenerator` (hence `Course`) construction, as `TrackGenConfig`'s
  `relax_solver` / `z_profile` are since PR10b. Config construction only requires
  non-empty strings: `GateGenConfig(gate_ordering="spiral")` and
  `GateGenConfig(z_profile="bogus")` now construct, and `GateGenerator` raises
  `ValueError` (`GateGenConfig.z_profile: unknown elevation 'bogus'; available:
  ['flat', 'noise', 'random_walk', 'uniform']`). The `z_*` range rules still run at
  config construction.
- **`GateGenConfig.gate_ordering` defaults to `None`**, "the generator's default order"
  (was `"ccw"`): it is resolved at `GateGenerator` (hence `Course`) construction, before
  any generator code reads the config, with the rule of `presets.gates(order=None)`
  (`legacy.to_gate_pipeline_spec` applies the same rule), and the name is written into
  the generator's run config, so a legacy `GateGeneratorSpec` (its `Params.from_legacy`,
  `max_gates`, `alloc_scratch`, `generate`), an elevation's `from_legacy` and
  `GateGenerator._config` never see `None`; `GateGenerator.config` reports the config as
  passed. The five built-ins resolve to `"ccw"` (same spec, same bytes), and a
  generator that accepts only `"raw"` (one that provides attributes, a legacy `toy.ring`)
  now builds without an explicit `gate_ordering`
  (`GateGenerator(GateGenConfig(generator="acme.helix"))`, and a `CourseConfig` with
  it). An explicit name keeps its checks and messages; `GateGenConfig(gate_ordering=None)`
  is accepted where it raised.
- **`gate_width` is documented as the full bar width** (finding A85): `half_size =
  0.5 * gate_width`, `left` / `right` at `position ± half_size * left_axis`. The code
  always did this; the configuration reference, `docs/relaxation/gates.rst` and the
  `GateGenConfig` docstring called it a half-width.
- **Launch sequence.** `bezier` / `hull` gate generations launch exactly what they did
  (14 launches for a default `bezier` / `ccw` generation on CPU). The normalizing
  generators (`polar`, `checkpoint`, `voronoi`) launch as many (14), with the count
  copy before the normalization instead of after it: the normalization is a versioned
  rewrite of `gates.xy`, one out-of-place launch (`_normalize_extent_to_k`, the same
  arithmetic, dimensions and scalars as 0.2.0.dev3's in-place kernel). The bytes are
  identical. A `gate_relax.spheres` stage refuses params that would not move anything
  (`gate_solve_iters == 0` or a float32 `2 * gate_radius` of 0); the presets and
  `GateGenConfig` select `gate_relax.none` for them, as the old path skipped the
  launch.
- **Gate working buffers are zero-initialised** at construction (0.2.0.dev3 used
  `wp.empty` for the generator scratch). No buffer is read before it is written.
- **The legacy deprecation fires at bind time, not at registration**, so loading a
  plugin that registers a `GateGeneratorSpec` (under `-W error` too) stays silent.
- **Parameter agreement.** PR10b's interim rule (every stage with a `half_width` field
  agrees) extends to `gate_width` and `gate_radius`: in a gate pipeline,
  `gate_frames` / `validity.gate_default` agree on `gate_width` and `gate_relax.spheres`
  / `validity.gate_default` on `gate_radius` (`PlanError` otherwise). The presets pass
  one value to every stage.
- `GateGenerator._scratch` (the positional 6-tuple) is gone: read intermediates with
  `gen.program.tap`. `gen.frame_fallbacks` stays.
- **The built-in gate generators are Stage classes**: `registry.get("gate_generator",
  "bezier")` (and `gate_generator_registry.get` / `GATE_GENERATORS`) returns a
  `kind="gate_generator"` Stage, not a `GateGeneratorSpec`: it has no
  `supported_orderings`, `max_gates`, `generate` or `alloc_scratch`, and
  `dataclasses.replace(...)` no longer applies. Read the orders from
  `registry.describe("gate_generator", name).supported_orderings` and the capacity from
  `Stage.capacity(params, layout)`; wrap a built-in in a Stage of your own to change its
  flags.
- **`elevation.flat` on tracks** now declares a `cl.count` read and carries
  `ElevationFlatParams(z_base=0.0)` (the resolved `Program.spec` of a default track
  shows it where it showed `None`). Track bytes are unchanged; a hand-built track spec
  that places `elevation.flat` before any `cl.count` writer now fails to plan.
- **The grade check's closing chord stays in the env's row**: in `_finalize_validity_k`
  the wrap is bounded by `max_gates` as well as by `count`, so a count above
  `max_gates` (an anchors or order plugin over-reporting it; it fails the `count` bit)
  no longer reads the next env's gates (or past the buffer) when `z_valid_grade > 0`.
  The same arithmetic for every `count <= max_gates`: bytes unchanged.
- **Planner hints** for common gate-pipeline mistakes: a track check (`validity.default`)
  in a gate pipeline names `validity.gate_default`; a centerline generator without a
  `stations` stage (or a track spec given `max_gates`) says to insert one; the
  `min_gates` error of a stations source names the stations stage.
- **`capturable=False` and `HOST` pipelines replay their capturable part from graphs.**
  On `cuda` outside a capture, a legacy `GeneratorSpec(capturable=False)` (or
  `GateGeneratorSpec`) is a `HOST` generator segment followed by a captured tail, and a
  `HOST` Stage anywhere in the pipeline runs eagerly between the graph launches of the
  segments around it; before, one `HOST` stage made the whole Program eager. Outputs are
  unchanged (the same launches in the same order; only where they are recorded changes),
  `graph` stays `None`, and a caller's capture still raises `NotCapturableError` before
  any launch. `Course` does not segment: over a `HOST` generator it still runs eagerly.
- **`repulsive` stops early where it can.** An eager generation on `cpu` stops its
  growth loop at the first stall window whose check finds every env frozen (with the
  defaults no earlier than iteration 144 of 269), so the launch count of an eager CPU
  generation now depends on the data. Its self-graph, and a Warp capture with
  conditional nodes, skip those windows on device. Eager `cuda` calls (`enqueue()`,
  `policy="eager"`) run the full bound with no host sync: `TrackGenerator.enqueue`,
  `GateGenerator.enqueue` and `Program.enqueue` keep "never host-syncs" (a `HOST` stage's
  own `ctx.host()` waits on the call's stream aside). Construction and `prepare()` run
  every window (the warm-up loads every kernel a window launches). The outputs are
  byte-identical to 0.2.0.dev3 in every case (fingerprints unchanged). Its graph now
  holds conditional nodes in Warp-registered captures (CUDA 12.4+), so, like a masked
  `Course` regeneration graph, it cannot be used as a CUDA child graph, and recording it
  into a capture whose stream has an unjoined side-stream fork fails (each window's
  `wp.capture_if` ends and resumes the capture); record inside
  `track_gen.exec.unconditional()` to keep 0.2.0.dev3's plain kernel nodes.
- **A Warp registration of torch's current, capturing stream gets no conditional node
  unless opted in.** When `wp.capture_begin(stream=..., external=True)` (or
  `wp.ScopedCapture(external=True)`) registers the `torch.cuda.graph` capture running in
  the calling thread, `skip_unless` (a masked `Course.regenerate`, every `BOUNDED`
  window) now records unconditionally, as `interop.torch.registered()` does by default:
  a conditional node renews the capture id torch's caching allocator tracks. Warp does
  not record who began a capture, so the same applies to your own `wp.ScopedCapture` of
  torch's current stream (same bytes, no device-side skip): opt in with
  `track_gen.exec.conditional()` (new) when torch is not the capturer, or with
  `registered(conditional=True)` under `torch.cuda.graph`. The library's own graphs (a
  producer's `graph`, legacy producers included) are marked and unaffected, and Warp
  captures of other streams are unchanged.
- **`registry.get("generator", "repulsive")` is a Stage class**, not a `GeneratorSpec`:
  it has no `capturable`, `generate` or `alloc_scratch` attribute and
  `dataclasses.replace(...)` no longer applies; read `registry.describe("generator",
  "repulsive")` (`capture`, `capturable`). A repulsive `TrackGenerator`'s private
  `gen._scratch.gen` is the built stage, whose buffers are its `scratch` mapping
  (`"scratch:generator/<name>"`).
- **The process-wide capture lock** (`runtime._CAPTURE_LOCK`). Every eager run of legacy
  (`accepts_ctx=False`) code still holds it, since that code launches on Warp's shared
  current stream. A legacy `capturable=False` producer now holds it only while its
  `HOST` segment runs, not while its captured tail replays, and a `HOST` Stage (which
  emits through `ctx`) never takes it, apart from the existing rule for calls on Warp's
  current stream while a capturable legacy producer has not captured its self-graph yet,
  which a segmented call re-evaluates before each part (graph launch or `HOST` part).
- **`LaunchContext.skip_unless` returns a `bool`**: `True` when it recorded the body
  under a conditional node, `False` when it ran it unconditionally (it returned `None`).
- **`NotCapturableError(name="", detail="", *, stage_ids=())`**: with stage ids, the
  message names the `HOST` stages and points at `require_caller_capturable`. Both forms
  now say "cannot be recorded into a graph capture" (was "a CUDA graph"): the error is
  also raised for a CPU `wp.ScopedCapture("cpu")`.
- **`HOST` work refuses a CPU graph capture.** Inside `wp.ScopedCapture("cpu")` (Warp's
  CPU capture records launches without running them), a producer with a `HOST` stage
  raises `NotCapturableError` before any launch, and `ctx.host()` raises it too; before,
  a `capturable=False` producer ran and recorded host reads of stale data.
  `GraphRunner.check_record` applies the same refusal, so `Course.generate(seeds=...)`
  raises before its reseed launches. Constructing a producer (`build`, `TrackGenerator`,
  `GateGenerator`, `Course`, `prepare()`) inside one raises `CaptureError` before
  allocating, as inside a CUDA capture (it recorded the warm-up instead of running it). A
  `bounded_loop` recorded there unrolls (never polls). `LaunchContext.capturing` stays
  `False` on `cpu`.
- **Refusal order under a caller's capture.** A Program with `HOST` stages raises
  `NotCapturableError` before the "never prepared" `CaptureError`, so one built with
  `prepare=False` reports its `HOST` stages; the argument checks (`mask`, `advance`,
  segment index) still come first.
- **The width stage runs before `relax`** (preview pipeline API). The default track
  order is `generator, resample, width, relax, uniform, frames, offset, elevation,
  validity, lift` (`presets.track`, `legacy.to_pipeline_spec`, `presets.gates(source=
  "relaxed_stations")`, which gains a `width` slot). The default pipelines' outputs and
  launch sequences are unchanged.
- **`XpbdParams`, `OffsetParams` and `ValidityParams` lose `half_width`**: the stages
  read `cl.half_width`. A `half_width` key in their params raises `ValueError` before
  coercion (`stage 'relax' (relax.xpbd): half_width is no longer a relax.xpbd param
  (0.2.0.dev4): the half-width is cl.half_width, written by the width stage
  (presets.track(half_width=...) / width.constant)`). `TrackGenConfig.half_width` and
  `presets.track(half_width=)` are unchanged for their users. The planner's `half_width`
  agreement rule stays for the stages that keep the field (`width.constant`, plugins).
- **Hand-built specs in the 0.2.0.dev3 order fail loudly**: `relax.xpbd` reads
  `cl.half_width` before any writer, and the read error says that the width stage runs
  before relax since 0.2.0.dev4 and how to fix it (a `relaxed_stations` gate spec built
  before PR13a (PR11), which had no width stage, gets the same hint). An old-order spec without
  relaxation (`relax.none, uniform, width, ...`) still plans and gives the same bytes,
  with one launch fewer (the uniform resample has no `cl.half_width` version to fill).
  Saved `/1` dicts migrate (see Added).
- **Plugin width stages run before relax, on the constant-spacing beads**, and their
  output is carried to the final points by the uniform resample (they ran after it). A
  width that depends on the final geometry is a rewrite of `cl.half_width` inserted
  after `frames` (documented); a writer of `cl.half_width` after `offset` raises
  `PlanError`. The preset fills `half_width=` into a plugin width stage's `half_width`
  field.
- **`presets.track` reads the registry** (it was pure data): only to see what the
  generator provides and which width stages have a `half_width` field.
- **`w_floor` (fail bit 4) is checked per final point** (see Added and Changed
  (outputs)).
- **`offset`'s kernel gained a trailing optional output** (`_offset_assign_k`:
  `track.half_width`, stored only when the array is non-empty), and
  `warp_pipeline.offset(..., hw_out=None)` / `InflateScratch(..., hw_out=None)` a
  trailing keyword: with `None` nothing is stored or allocated, so the legacy entry
  points keep their "allocates nothing" contract. The 0.2.0.dev2 composition
  (`_inflate_warp_alloc`) allocates it and binds it as its `Track.half_width`. The commit
  holds, restores and poisons `half_width` with the other point fields (same launch
  counts; the held copy grows by 4 B per point).

### Changed (outputs)

Deliberate output changes, for pipelines whose width is anything but `width.constant`
(any plugin stage writing `cl.half_width`, `width.from_channel`, a generator that
provides `raw.half_width`). No built-in default pipeline uses one: every
`TrackGenConfig`, every default and PR11 preset and every fingerprint is byte-identical.

1. **XPBD separates with the per-bead half-widths** (pair target `r_i + r_j`, bend floor
   `r_i`, `r_i = w_i (1 + margin)`) instead of a single `half_width`.
2. **The band follows the per-point width**: `outer` / `inner` at `c_i ± w_i n_i`.
3. **The thickness check (fail bit 3) is per pair and per point**: every non-neighbouring
   pair at least `(1 - relax_tol)(w_i + w_j)` apart, every curvature radius at least
   `(1 - relax_tol) w_i`.
4. **The floor check (fail bit 4) reads the width carried to the final points**: a plugin
   width now runs on the pre-relax beads and is re-interpolated, so a pinch on a single
   bead that falls between two final samples no longer fails (the minimum over the
   final points is at least the minimum over the beads).
5. **A plugin width that writes a constant value runs the per-point kernels** and does
   not reproduce `width.constant`'s bytes (an in-kernel `float32` target differs from
   the host's double-precision one for some `(half_width, margin)` pairs, by up to
   `6e-3` at `half_width=0.07, margin=0.2`).
6. **Plugin width stages see the unrelaxed constant-spacing loop** (they saw the final
   points in 0.2.0.dev3).

### Fixed

- Library-readiness findings closed by roadmap PR12: MEDIUM "capturable is
  all-or-nothing" (segmented `HOST` execution); LOW "capturable=False holds the global
  lock for seconds" (its captured tail no longer holds it, `HOST` Stages never take it);
  LOW "capture_while port not a drop-in" and LOW "capture_while fails inside non-Warp
  captures" (sidestepped: `bounded_loop` puts each window under `skip_unless`, a
  `capture_if` in Warp-registered captures and an unrolled window everywhere else); LOW
  "stale capture docs" (`capturable` is no longer described as all-or-nothing, and the
  repulsive docs describe its early exit). With the absorbing freeze already on `v2`,
  the repulsive part of CRITICAL "repulsive cannot be graph-captured" is closed.
- Library-readiness findings closed by roadmap PR13a: HIGH A8 "constant half-width and
  a 2D-centerline-only generator output baked through every stage; no per-point
  attribute channel" (carried `raw.*` channels, the width registry and per-point
  consumers, provided altitudes and widths); LOW A128 "the `w_floor` check is a
  tautology" (a per-point check on the final points). MEDIUM A22 "Track and
  GateSequence have fixed schemas" is closed for per-point attributes
  (`Track.half_width`, `Track.attributes`, `GateSequence.attributes`); per-env
  attributes, `layout` and the centerline / band split stay open (PR13b and later).

### Deprecated

- **`GateGeneratorSpec`** (the legacy gate generator ABI: `generate(seeds, config, out,
  scratch[, ctx])` into a staging view, `alloc_scratch`, `max_gates`,
  `supported_orderings`). It keeps working through an adapter stage that writes
  `gates.xy` / `gates.count` and orders its own gates (no order or normalize stage
  follows it); a `capturable=False` spec makes the Program eager, an
  `accepts_ctx=False` one runs in `ctx.scope()`, as before. Binding one
  (`GateGenerator`, `Course`, `build`, `legacy.to_gate_pipeline_spec`, the conformance
  kit) emits a `DeprecationWarning` once per registry and name, attributed to the
  caller's frame: `GateGeneratorSpec 'toy.ring' (...) is deprecated: write a Stage with
  kind='gate_generator' and output='anchors' (docs/contributing/writing-a-generator.rst);
  legacy gate generators run through an adapter until 1.0`. Registration and
  entry-point loading stay silent. `presets.gates` with a legacy generator refuses
  `order=` (choose its ordering with `GateGenConfig(gate_ordering=...)`).
- **The `warp_gate` helpers of 0.1 gate plugins**: `alloc_order_scratch`,
  `order_points` (the closed-set ordering dispatch), `finish_ordered_gates` and the
  `_GateStaging` name (moved into the legacy adapter, `track_gen.stages.gates`) are
  deprecated and removed in 1.0. They keep their 0.2.0.dev3 signatures and bodies and
  emit a `DeprecationWarning`, so a `GateGeneratorSpec` written to the 0.1 ABI (which
  calls them from `alloc_scratch` / `generate`) still runs; the library itself no longer
  calls them (ordering and normalization are pipeline stages). Their signatures are
  pinned in `tests/test_legacy_plugin_abi.py`.

### Removed

- Private `track_gen._src.warp_gate` names: `_run_gate_pipeline` and `_gate_warp_alloc`
  (the positional 6-tuple scratch). From the built-in gate modules: `generate_*_gates`,
  `*_gate_alloc_scratch`, `PointGateScratch` and its siblings, `_max_point_gates`. None
  was public or called by a legacy plugin; the tests keep a frozen copy of the
  0.2.0.dev3 gate path (`tests/_legacy_gate_pr11.py`) as the byte-identity reference.

### Deferred

- **`spacing_metric`** (`"xy"` / `"xyz"`) moved from roadmap PR11 to PR13, because 3D
  arc length needs an altitude on the centerline before the stations are placed (the
  `raw.z` / `cl.z` carry-through). PR13a ships it on the `stations` stages (see Added).
  **`gate_arclen` never gets it**, a decision rather than a deferral: its arc table feeds
  the gate elevation that produces `gates.z`, so a 3D metric there would be circular
  (and whenever an altitude exists upstream, `gate_arclen` and the gate elevation are
  dropped). It stays plan-view.
- `Recipe` composite generators; attributes other than `z` provided by `stations`
  (`tangent`, `half_size`, `roll`); `roll` on `gate_frames.default`; `preferred` hints for track generators; split gate checks; a
  `GateGenConfig.on_invalid` field; `Course` built from a `PipelineSpec`.
- **`Course` with a gate generator that provides `half_size`** (posed gates):
  `CourseConfig` requires `gen.gate_width > 0` while such a generator refuses any
  `gate_width`, so the combination cannot be built yet (pinned by a test).
- From roadmap PR12: a segmented `Course` (over a `HOST` generator it runs eagerly);
  self-graphs on `cpu` (Warp's CPU graph capture); a public knob for the `bounded_loop`
  lowering; `capture_while`; a non-blocking `HostOps.upload`; refusing direct host syncs
  in `HOST` stages at run time (a `.numpy()` or `wp.synchronize()` in a `HOST` stage still
  runs in 0.x, but the conformance kit fails it in `hot_path.host_sync`, and a per-call
  allocation in `hot_path.allocation`; only a legacy `capturable=False` spec is waived).
- From roadmap PR13 to **PR13b** (the result-object half of the split): `Track.layout`,
  `validate()`, `view()` and `copy_from()`; a `clone()` that iterates the fields (PR13a
  adds `half_width` and `attributes` to the explicit one); `out=` on the facades and
  `out.attributes` with keys from another pipeline (mixed batches; PR13a raises); the
  capability protocols (`HasCenterline`, `HasBand`, `HasGates`) and `CourseLine ->
  Centerline`; utilities calling `validate()`; `randomize.uniform` and the parameter
  latch's removal; the `generator/external` ingestion stage; the `closed` flag plumbing.
- From roadmap PR13a, later: a public `StageBuilder.carried()` (plugin `stations` stages
  carry no centerline channels: they compute what they write, and a `z` provider with
  one raises `ValueError`); carried channels written by a stage rather than the
  generator (an `annotate` kind, `interp=` on stage writes; a stage-written `cl.*`
  channel lives on the points that stage saw); an order that publishes its permutation
  (`gates.src_idx`), so attributes reach the gates through any order (they need
  `order.raw` now: best effort); `ops.arc_resample(tables=)`; a public constant
  declaration for width stages (only `width.constant` selects the scalar kernels); an
  arc-fraction attribute policy (opt-in, against bead attachment); renaming on
  `export=`; per-env attributes; `GateGenConfig.spacing_metric`; an elevation on the raw
  loop (`"xyz"` with `source="stations"` needs a `z` provider); a dynamic
  `width.constant` (the spacing and capacity derive from it); attribute-aware utilities
  (PR14).

### Testing

- `tests/test_gate_pipeline_parity.py`: every stage boundary of the gate Program
  (`anchors.*`, `gates.xy` after order / normalize / relax, `gates.cum`, `gates.perim`,
  `gates.z`, lifted positions, tangents, the final fields, the frame fallbacks) equals a
  frozen copy of the 0.2.0.dev3 gate path (`tests/_legacy_gate_pr11.py`, pinned to
  golden digests captured from the live path before it was deleted:
  `tests/goldens/pr11_gate_stages.json`, tool `tests/tools/capture_pr11_gate_stages.py`)
  for the five generators x their orderings x four profiles x `gate_solve_iters` {0, 8}
  x `gate_align` x `gate_width` {0, 0.04}, plus radius-0, scale and `min_gates` cases
  and the legacy `toy.ring`; masked regeneration; launch sequences equal 0.2.0.dev3's
  (with the documented normalize reordering). The ten gate fingerprints and
  `tests/test_pipeline_parity.py` are unchanged.
- `tests/test_gate_stage_plugins.py` with the toys of `tests/_gate_toys.py`
  (`acme.helix` providing `z` and `tangent`, `acme.helix_posed` adding `half_size` and
  `roll`, `acme.scatter`, the order `acme.reverse`, the track generator
  `acme.superformula`, the gate check `acme.max_height`, and PR10b's
  `acme.sinusoid_hills` on gates): provides-driven stage lists and bitwise carried
  values; third-party orders through the config and the preset; stations and relaxed
  stations spacing; elevation rewiring three ways; the plugin fail bit; every toy
  through `check_gate_generator` / `check_stage`; registration rules and quarantine;
  the bind-time deprecation (silent registration, one warning attributed to the test
  module).
- `tests/test_gate_presets.py` (preset vs. config invariant, selection forms and
  errors, sources, order compatibility and the default ordering (`None` resolving to
  `"ccw"` for every built-in, `"raw"` for a provider),
  capacity texts, `BatchSpec.max_gates` validation and round trip, the domain, plugin
  channel, carry, provided-attribute, agreement and posed rules, per-slot layouts,
  `build` / `TrackGenerator` / `GateGenerator` dispatch) and `tests/test_gate_commit.py`
  (`GATE_FAIL_BITS`, each bit on hand-built inputs, exact masked regeneration, poison,
  `CheckpointSet.from_gates` / `ProgressTracker` on a poisoned env, launch counts).
- Review follow-ups: the 0.1-ABI `warp_gate` helpers pinned and byte-checked against the
  frozen 0.2.0.dev3 code, and a 0.1-style plugin (`alloc_order_scratch` +
  `finish_ordered_gates`) through `GateGenerator` equal to the frozen pipeline
  (`tests/test_legacy_plugin_abi.py`); the bind-time deprecation under an error filter;
  elevation stages through `check_stage(pipeline="gates")`; a track `from_legacy` on
  gates; the default-order rule for a provider without a preferred order; the preset's
  validity / `z_base` / centerline-argument / station-count rules; the planner hints;
  `SpecInfo.supported_orderings`; spec pickling (nested frozen params included); a
  legacy `GeneratorSpec` through `stations` and `relaxed_stations`; the default
  ordering (`None`) seen by a legacy spec and a `Course` with a provider; the posed
  `validity.gate_default` `gate_width` message; a count above `max_gates` with the grade
  check (count bit only, no cross-row read); `finalize_gate_sequence` with a persistent
  `fallbacks` and no `fail=` allocating nothing (and its no-host-sync driver).
- CUDA (`tests/test_gate_stages_cuda.py`, collects on CPU): stations gates, the helix
  toys and the five built-ins recorded in a caller's graph equal to eager twins; masked
  gate regeneration under capture leaves unmasked envs untouched.
- Roadmap PR12 toys (`tests/_bounded_toys.py`, public plugin surface only): the BOUNDED
  relax smoothers `toy.converging_smooth` (an absorbing per-env convergence freeze) and
  `toy.leaky_smooth` (the same schedule, not absorbing), the HOST generator
  `toy.host_circle` and the HOST gate generator `toy.host_ring` (NumPy host RNG through
  `ctx.host()`), plus deliberately broken variants (an undeclared loop, a body that
  reads `active` back, a direct call of the poll, an allocation or a readback in the last
  window only, a HOST relax, a HOST relax that syncs, reads back and allocates outside
  `ctx.host()`).
- `tests/test_bounded_loop.py` (CPU): `bounded_loop` semantics (the static `it`, the
  prefix, short last windows, `first_check` at or past `n_iters`, `check_every = 1`, the
  stop at the first 0, validation), the lowering policy per capture state (fake CUDA
  device: eager windows inline with no poll and no sync, one `capture_if` per window in a
  registered capture, the forced seams, CPU graph captures never polling, `prepare()`
  unrolling), the STATIC declaration refusal, the loop probe, `ops.all_set`, the
  sanctioned poll (a plugin's sync in a loop body and a direct poll call are caught), the
  kit's unrolled hot path catching a last-window allocation or readback, `HostOps` and
  `StageBuilder.host_buffer` (the declared-class `ValueError` of `ctx.host()`, a pinned
  `via` on CUDA, a host buffer for HOST slots only and the lazy forced-poll buffer).
- `tests/test_host_segments.py` (CPU): planner segments (HOST generator, HOST stage in
  the middle, BOUNDED + STATIC, legacy `capturable=False`, a HOST gate generator),
  `require_caller_capturable` (every HOST id named before allocating, round trip),
  eager determinism and `enqueue_segment` equivalence, exact masked regeneration, the
  refusals before any launch (emulated and real CPU captures, ahead of the unprepared
  error; `Course.generate(seeds=...)` before its reseed; construction inside a CPU
  capture), a recordable non-HOST segment, the runner's segmented mode and the
  capture-lock rule on a fake CUDA device (re-evaluated per part when a legacy producer
  becomes pending mid-call), `Program.enqueue`'s legacy-stream routing, and the kit on
  the toys (`toy.leaky_smooth` fails exactly `batch_independence` and
  `bounded.absorbing`; the HOST toys pass `hot_path.host_sync` / `hot_path.allocation`
  with their `HostOps` sanctioned; `toy.sneaky_host` fails both).
- `tests/test_inplace_regen_cpu.py`: `unconditional()` and a hand-made registration of
  torch's capture keep `skip_unless` unconditional; `registered(conditional=True)`'s
  explicit marker re-allows it.
- `tests/test_repulsive_bounded.py` (CPU): the Stage registration, the default schedule
  (`first_check = 128`, 9 windows) and a property check of the window alignment, the
  host poll equal to the unrolled loop (small and default configs), the legacy entry
  equal to the Stage, also with a mismatched stall window, env `e` alone vs. inside a
  larger batch, and a CPU graph capture replaying the eager batch.
- CUDA (`tests/test_bounded_host_cuda.py`, collects on CPU): the three lowerings bitwise
  equal for repulsive and the converging toy (and different for the leaky one), CUDA eager
  calls never stream-synchronizing, repulsive caller capture and self-graph against
  eager twins, the self-graph node counts (fewer than 5k unrolled, conditional windows),
  batch independence in every lowering, segmented HOST execution (tail graph launches,
  masked regeneration, the refusal before any launch, a caller recording the tail),
  legacy `capturable=False` segmentation and the capture lock, the HOST gate generator,
  bounded windows nested in conditional bodies (and in a masked `Course` regeneration),
  the kit on CUDA, and HOST segments running while another thread captures.
- Existing tests updated for the new surface (the repulsive Stage, `describe()`, the
  BOUNDED acceptance, the kit's check list, the no-host-sync drivers). The frozen guards
  (`tests/test_generator_fingerprints.py` including `track/repulsive/*`,
  `tests/test_pipeline_parity.py`, `tests/test_gate_pipeline_parity.py` and its golden,
  the `tests/test_legacy_plugin_abi.py` pins) are unchanged and pass.
- Roadmap PR13a toys (`tests/_channel_toys.py`, public plugin surface only, installed by
  a fake `acme-channels` distribution): the generators `acme.ribbon` (exported `angle`,
  `nearest`, `previous`, `linear` `float32` and `vec2f` attributes, plus an unexported
  width source), `acme.ribbon_plain` (the same loop without them), `acme.ribbon_metres`
  (a `linear` attribute in metres), `acme.helix_loop` (provides `z`) and `acme.pinch`
  (provides `half_width`); the width stages `acme.wave`, `acme.wave_after` (a rewrite),
  `acme.rescale` and `acme.curv_width` (also writes a `cl.*` plugin channel); the gate
  generator `acme.tagged` (an `anchors.*` attribute), the gate check
  `acme.gate_clearance` (an exported `gates.*` channel and a fail bit), the plugin
  stations stage `acme.st`, the track check `acme.curv_check` and the elevation
  `acme.slope` (a plugin channel rewired on gates); and broken ones
  (`acme.ribbon_partial`, an attribute half written; `acme.negative`, a negative width).
- `tests/test_width.py` (CPU): the stage order, `width.constant` launching nothing, the
  kernel selection by width kind, the per-point kernels equal to the scalar ones at
  constant width (offset, validity, band, one XPBD sweep and the full solve with
  `float32`-consistent scalars) and the scalar path pinned where the two differ,
  `Track.half_width` (constant, variable, `|outer - center|`, `clone()`), the floor and
  thickness bits on variable widths, XPBD respecting a per-point width,
  `width.from_channel` (values, NaN, wiring and dtype errors), providers, placement and
  stale-read errors, the preset's `half_width` rules, the 0.2.0.dev3 messages, the `/1`
  migration, and the legacy no-allocation entry points.
- `tests/test_channels.py` (CPU): carry exactness (a carried copy of the points equals
  them byte for byte; attributes never change the geometry), analytic attribute values
  for every policy, the interpolation kernels on hand-built tables (ties, holds, angle
  range, sentinels, host checks), bead attachment, the policy / dtype / export errors,
  `Track.attributes` (keys, dtypes, stable pointers, masked regeneration, poison,
  `out=`, `Program.attributes`, launch counts), provided altitudes.
- `tests/test_stations_xyz.py` (CPU): equal 3D spacing on a helix loop, the centerline
  elevation of `relaxed_stations` + `"xyz"`, `relaxed_stations` + `noise` (`"xy"`)
  byte-identical to PR11 (golden `tests/goldens/pr13a_pr11_relaxed_stations.json`, tool
  `tests/tools/capture_pr13a_relaxed_stations.py`), the preset's composition errors and
  the planner's, the stations' sentinels on skipped envs, geometry identity with and
  without attributes, gate attributes (through the stations, left out after a permuting
  order, by index from a gate generator, a gate check's own export, masked and
  poisoned), the unread-attribute and domain errors, elevation plugin channels rewired on
  gates, and a plugin stations stage that carries nothing and says so.
- CUDA (`tests/test_channels_cuda.py`, collects on CPU): caller capture of attribute,
  variable-width, poison and 3D-stations pipelines equal to eager twins (before and
  after a reseed, with width and attribute params written in the graph), masked
  regeneration recorded with in-graph masks, the per-point kernels, the carries and the
  table lookups against the scalar and plain ones (geometry identity, the stations' xy
  branch against PR11, the XPBD variant against the scalar solve), the build-time
  constant fill ordered before a side-stream build, exact padding on dirty device
  memory, and the kit on the toys.
- The conformance kit over the new built-ins and toys (`width.from_channel` with
  `ports=`, the attribute generators, the gate generator), and broken toys that fail
  `attributes.written` and `width.values`. The frozen guards (fingerprints,
  `tests/test_pipeline_parity.py`, `tests/test_gate_pipeline_parity.py` and its golden,
  the `tests/test_legacy_plugin_abi.py` pins) are unchanged and pass; the tests that pin
  the stage order, the removed `half_width` fields and the format string are updated.

## 0.2.0.dev3 (2026-10-22, tag `v0.2.0.dev3`)

### Packaging

- **`warp-lang` floor raised to 1.18** (`warp-lang>=1.18,<2`); Warp 1.14 is no longer
  supported. Warp 1.18's PyPI wheels need an NVIDIA driver R580+ and an sm_75+ GPU for the
  `cuda` device. Compatibility code for older Warp is removed (the conformance kit's
  recorded-launch fallback), the CI matrix tests {1.18.0, latest}, and the stale Warp 1.14
  fingerprints are dropped.

### Added

- **`track_gen.pipeline` (preview): the track pipeline as a list of stages** (roadmap
  PR10a). The pipeline `TrackGenerator` runs is now described by data: a `PipelineSpec`
  lists the stages in order, `build()` plans and allocates it once, and a `Program` runs
  it. Outputs are byte-identical to 0.2.0.dev2. The package is a **preview**: names may
  change before 1.0 (roadmap PR10b, below, opens four more stage kinds to plugins). The
  new page `docs/how-it-works/pipeline-preview.rst` describes it, and
  `docs/reference/api.rst` documents it. `track_gen.pipeline` is outside
  `track_gen.__all__`; `import track_gen` does not import it.
  - `StageRef(kind, name, params=None, id=None, ports={}, capacity={})`, frozen: one stage
    slot. `params` is a Params instance, a mapping (coerced and validated at build time)
    or `None`; `id` (no `"/"`) defaults to `kind` (`slot_id` is `id or kind`); `ports`
    rewires a local port to another channel of the same dtype and size; `capacity` is
    the generator's `{"raw": num_points}`, the default raw point count that a Stage
    generator's `Stage.capacity()` overrides (any other key, or a `capacity` on a
    non-generator stage, is a `PlanError`).
  - `BatchSpec(num_envs, device, n_max, closed=True, scale=1.0)`.
  - `PipelineSpec(batch, stages)`: `ids`, and the pure data operations `replace(id, ref)`
    (keeps the slot's id when `ref.id` is `None`), `insert_after`, `insert_before` and
    `remove`, which raise `ValueError` for an unknown id and leave everything else to
    `build`. `to_dict()` / `from_dict()` (format `"track_gen.pipeline/1"`, JSON-safe, no
    registry needed to read it back; a missing key raises a `ValueError` naming it);
    building the round-tripped spec gives byte-identical outputs.
  - `legacy.to_pipeline_spec(config, *, registry=None)`: the default spec for a
    `TrackGenConfig`, with the generator's params resolved through `registry`. Its ten
    stages are `generator` (`generator.<config.generator>`), `resample`
    (`resample.constant_spacing`), `relax` (`relax.xpbd`, or `relax.none` when
    `relax_enable=False`), `uniform` (`resample.uniform`), `width` (`width.constant`),
    `frames` (`frames.default`), `offset` (`offset.default`), `elevation`
    (`elevation.<z_profile>`), `validity` (`validity.default`) and `lift` (`lift.flat` or
    `lift.elevated`). Each built-in stage calls the 0.2.0.dev2 wrappers and kernels with
    the same arguments, and its params keep the flat `TrackGenConfig` field names.
  - `build(spec, *, registry=None, seeds=None, out=None, stream=None, prepare=True)`, the
    planner. Before it allocates or launches anything it checks the batch (`num_envs >= 1`,
    `n_max >= 3`, `closed=True`), unique slot ids with exactly one generator stage, first,
    with the id `generator`; resolves the generator through `registry` as
    `TrackGenerator` does and every other stage from TrackGen's built-in catalogue
    (since PR10b, the `relax`, `width`, `elevation` and `validity` stages through
    `registry` too); coerces and validates each stage's params; checks that every stage
    that has a `half_width` field agrees on it; checks that every channel read has an
    earlier writer, that a second writer of a channel also reads it, that every `Track`
    field has a writer, and that `lift.flat` does not
    follow an elevation other than `elevation.flat`; and checks the `seeds` type. It
    then builds the `SeedState`, allocates every channel version once
    (zero-initialised), binds the final version of each exported channel to the `Track`
    (or to the `out=` Track, checked field by field), namespaces stage scratch as
    `scratch:<id>/<name>` and calls each stage's `build` once, checking its write ports
    right after. A broken rule raises `PlanError` (a `ValueError`) naming the stage, its
    `kind.name`, the port and the channel; params keep their own `ValueError` /
    `TypeError`, an unknown generator the registry's `ValueError`, and a bad `seeds` a
    `TypeError`; a build inside a capture raises `CaptureError` before allocating.
  - `Program`, a `GraphRunner` launchable with the PR3/PR4 rules: `run(*, stream=None,
    sync=False, advance=False, mask=None, policy="auto")`, `enqueue(ctx, *,
    advance=False, mask=None)` (seed prologue, parameter latch, every stage in order,
    `generation` write, with PR10b's commit before it; no allocation, no host sync),
    `prepare()`, `result` (the same
    `Track` forever), `seeds`, `seed`, `inputs`, `params` (keyed by stage id), `graph`,
    `capturable` and `capture_class`. A Program built with `prepare=False` is prepared
    by its first `run()` outside a capture; recording it unprepared raises
    `CaptureError`.
  - Channels v1: `seed`, `raw.xy`, `gen.valid`, `cl.xy`, `cl.count`, `cl.truncated`,
    `cl.half_width`, `cl.tangent2`, `cl.normal2`, `cl.kappa`, `cl.arclen2`,
    `cl.length2`, `cl.winding`, `band.outer2`, `band.inner2`, `cl.z`, `track.count`,
    `track.valid`, `track.arclen`, `track.length` and
    `track.outer/center/inner/tangent/normal`. `Track.arclen` / `Track.length` are the
    3D `track.arclen` / `track.length` of `lift.elevated` when it runs, else the
    plan-view `cl.arclen2` / `cl.length2` of `frames`. A stage that reads and writes one
    channel (`relax.xpbd`, `resample.uniform`) gets a new version of it, and must write
    its whole output there: the pre-relax centerline survives the relax stage.
- **`StageBuilder.read(port)` and `StageBuilder.temp_like(port)`**, with `Stage.reads` /
  `Stage.writes` documented as `{local port: channel}` maps and an in-place rewrite
  declared by naming one channel in both. A PR8 generator `Stage` runs unchanged: a
  generator still declares no reads, so `read()` raises `ValueError` there, and
  `temp_like()` returns private scratch shaped like a write port's channel.
  `StageBuilder.write(port, dtype=None, *, per=None)` takes `dtype` / `per` from the
  channel when omitted (a catalogue channel, or a plugin channel an earlier stage
  wrote); explicit values must still match, and a plugin channel's first writer passes
  `dtype` (`per` still defaults to `"env"`).
- **Plugin pipeline stages: `relax`, `width`, `elevation` and `validity`** (roadmap
  PR10b). A plugin registers a `Stage` subclass of one of these kinds with
  `reg.add(MyStage)`, through the same entry point, registry, quarantine and
  `describe()` as a generator, and users select it by name. The new guide
  `docs/contributing/writing-a-stage.rst` walks through a relax stage (Taubin smoothing),
  a width stage, an elevation stage (sinusoidal hills) and a validity check (maximum
  curvature).
  - The built-ins are registered under these kinds by TrackGen's own `register`:
    `relax:xpbd`, `relax:none`, `width:constant`, `elevation:flat`,
    `elevation:uniform`, `elevation:random_walk`, `elevation:noise` and
    `validity:default`. `resample`, `frames`, `offset` and `lift` stay in the private
    catalogue (not open to plugins).
  - Registration rules for the four kinds: no `output` and no `provides`; `reads` and
    `writes` name catalogue channels or namespaced plugin channels `cl.<package>.<name>`;
    one write port per channel; a rewrite reads and writes a channel through ONE port;
    each kind writes only its own catalogue channel (`relax` -> `cl.xy`, `width` ->
    `cl.half_width`, `elevation` -> `cl.z`, `validity` -> `valid.fail`) plus plugin
    channels; a validity stage writes `valid.fail`. A Stage of another kind is refused
    as not open to plugins, and a non-Stage spec under one of the four kinds raises
    `TypeError`. The planner repeats the port and write rules after `StageRef.ports`
    rewiring, as `PlanError`. `SpecInfo.reads` reports a stage's read ports.
  - The planner resolves these kinds through the registry (`build(spec,
    registry=...)`) and keeps the resolved class: an unknown name raises `PlanError`
    with the registry's list (`stage 'relax' (relax.nope): unknown relax 'nope';
    available: ['none', 'xpbd']`), a quarantined plugin its `PluginLoadError`. A plugin
    channel's dtype and size are fixed by its first writer. The Program's capture class
    is the highest of its stages': a `HOST` stage makes it eager; `BOUNDED` is a
    `PlanError` (PR12).
- **`track_gen.pipeline.presets.track(*, num_envs, device, n_max=384, scale=1.0,
  generator="bezier", generator_params=None, num_points=256, half_width=0.1,
  spacing=None, relax="xpbd", width="constant", elevation="flat", checks=("default",),
  on_invalid="keep", z_base=0.0)`**, returning the `PipelineSpec` of a track pipeline. Each stage is
  a name, a `(name, params)` pair, a `StageRef` (of the matching kind) or `None`
  (`relax=None` is `relax.none`, `elevation=None` is `elevation.flat`; `width=None` is a
  `ValueError`); further checks are appended after `validity` under their names (a
  duplicate id is a `ValueError`). `half_width` is filled into the built-ins that bake
  it, `spacing` defaults to `0.6 * half_width`, and `lift` is `lift.flat` exactly when
  the elevation is `elevation.flat`. It is pure data (no registry access), and
  `presets.track(num_envs=E, device=d)` builds byte-identical to
  `legacy.to_pipeline_spec(TrackGenConfig(num_envs=E, device=d))`. Combine it with the
  edit operations, e.g. `.insert_after("validity", StageRef("validity",
  "acme.max_curvature", id="kmax"))`. `checks` is a sequence whose first selection is
  `"default"` (a single selection raises `TypeError`, another first check
  `ValueError`), and params that set a `half_width` other than the preset's raise
  `ValueError` naming `half_width=`.
- **`TrackGenerator(config_or_spec, rng=None, *, seeds=None, stream=None)`** accepts a
  `PipelineSpec` as well as a `TrackGenConfig`. Seeds come from `rng` or from `seeds`
  (any `SeedState` form; both raise `ValueError`; neither gives env `e` the seed `e`).
  `gen.config` returns what was passed, so `TrackGenerator(gen.config, rng)` rebuilds
  an identical generator. `gen.program` (preview) is the `Program`; the new
  `gen.num_envs`, `gen.device` and `gen.n_max` read the batch whichever form was passed
  (`gen.config` is a `TrackGenConfig` or a `PipelineSpec`). Any other argument raises
  `TypeError`. `Course` still takes a `TrackGenConfig`.
- **Fail bits.** The new `[E]` int32 channel `valid.fail` is a per-env bitmask, bit `i`
  set when check `i` failed:

  | bits | owner |
  |---|---|
  | 0 generator, 1 turning, 2 finite, 3 thickness, 4 w_floor, 5 border, 6 grade, 7 truncation | `validity.default` |
  | 8-15 | reserved for future built-in checks |
  | 16-30 | plugin checks, in pipeline order |
  | 31 | never used |

  `validity.default` keeps the single fused `_validity_k` launch and writes bits 0-7
  next to its 0/1 flag (an empty loop gets bit 1, plus 0 and 7 when they apply);
  `track_gen.pipeline.BUILTIN_FAIL_BITS` is that name -> bit table (the border and
  grade bits are named even when their check is off). A plugin validity stage rewrites
  `valid.fail` (`fail_out = fail_in | own`) with a bit from the new
  **`StageBuilder.fail_bit(name=None)`**: 16, 17, ... in pipeline order, named after the
  stage id by default; a 16th plugin bit, a name taken twice, a validity stage that owns
  no bit, a plugin reserving built-in bits (built-in-ness is the registry's record of
  where the class was registered, not its module name) or a pipeline whose `valid.fail`
  another stage than `validity.default` starts (that would drop the built-in checks, so
  truncation would fail open) raise `PlanError`, and `fail_bit()` from another kind
  raises `ValueError`. `Program.fail_bits` and the new
  **`Track.fail_mask`** (`[E]` int32, `None` on a `Track` built elsewhere) and
  **`Track.fail_bits`** (name -> bit) expose it; `Track.valid == (fail_mask == 0)`.
  `Track.clone()` clones `fail_mask`.
- **The commit, `on_invalid` and exact masked regeneration.** The commit is the
  Program's epilogue, after the stages and before the `generation` write. The `Track`
  arrays stay the final channel versions (no copy). Under
  `generate(advance=True, mask=m)` the stages still recompute and write every env, but
  a kernel before them holds the `Track` rows of every env with `m[e] == 0` and the
  commit restores them bit for bit after them: an unmasked env ends the generation with
  exactly the bytes it had, `generation` included, whatever its recomputation gave, so
  masked regeneration no longer relies on bitwise determinism (an all-zero mask stays an
  exact no-op). This is a hold / restore, not a sole-writer commit: within the generation
  the unmasked rows are overwritten by the recomputation and then restored, both on the
  generation's stream. Work ordered after the generation on that stream (or after an
  event recorded on it) never observes them; a reader on another stream without such a
  dependency can see an unmasked env's recomputed track mid-generation. The commit also
  derives `Track.valid = (fail_mask == 0)` when a check runs after `validity.default`.
  The new **`BatchSpec.on_invalid`**
  (`presets.track(on_invalid=...)`, `Program.on_invalid`) is `"keep"` (default,
  0.2.0.dev2 behaviour) or `"poison"`: an env with a fail bit set then leaves with
  `count = 0`, NaN `outer` / `center` / `inner` / `tangent` / `normal` / `arclen`,
  `length = 0`, `winding = 0` and `valid = 0`, keeping its `fail_mask`, `truncated` and
  `generation`; valid envs are byte-identical to `"keep"`, and a masked generation
  poisons masked envs only. `BatchSpec.to_dict()` writes `on_invalid`, and `from_dict()`
  defaults it to `"keep"` (format unchanged).
- **`Program.tap(channel, *, after=None)`**: the working buffer of a channel, its final
  version or the version right after stage `after`; read-only by contract and stable for
  the Program's lifetime. An intermediate version holds every env's recomputation of
  the last generation; the final version of a channel the `Track` exports is the
  `Track`'s own array, so it holds the committed values, except `track.valid` after a
  plugin check (then `validity.default`'s own flag: read `Track.valid` or
  `tap("valid.fail") == 0`). `Program.channels` maps each
  channel to its writers' ids. An unknown channel or stage,
  or a stage before the channel's first writer, raises `KeyError`.
  `benchmarks/compare_generators.py` and `viz/render_readme_assets.py` read the
  intermediate centerlines through it instead of `gen._scratch`.
- **`track_gen.testing.check_stage(kind, name, params=None, *, upstream=None,
  devices=None, num_envs=8, seeds=(11, 1009), capture_modes=..., skip=(),
  modules=())`**, the conformance kit for the four stage kinds: it runs the stage in its
  slot of `presets.track(**upstream)` through `TrackGenerator` and reports the generator
  kit's checks on the whole `Track` (without `diversity` and `output.closed`), plus
  `output.accepted` (relax, width, elevation), `stage.isolation` (re-running only the
  stage, into write buffers filled with a poison byte, changes none of its inputs and
  reproduces its outputs, padding included) and, for validity,
  `validity.bits` (the stage owns a bit and touches no other: re-run alone on a
  synthetic `valid.fail` holding foreign bits, it must pass them through for every env).
  `CHECKS` lists the four
  kinds, and the pytest plugin's `track_gen_conformance(name, kind=...)` and
  `generator_names(kind)` accept them. Every built-in stage passes.

### Behaviour changes

- **The `sdf` box query takes the exact minimum of the interpolated field over the box
  outline.** It used to take the minimum of 5 bilinear samples (the centre and the 4
  corners), so it missed an edge crossing a box side while all 4 corners stayed on the
  road, typically the inner edge at a tight bend: a car box whose side reaches 10 mm into
  an inner bend read +16 mm. Each side is now split at the grid lines it crosses, and each
  piece, a quadratic, is minimised in closed form. The centre and corners stay candidates,
  so `distance` never increases beyond float rounding (at most 1e-7 m on the racing set).
  `nearest` and `normal` come from the point where the minimum is reached. **`oob` and `distance` change near the edges, so policies trained
  on the old `sdf` query (rewards or terminations from `oob` / `distance`) should be
  retrained.** On the racing near-edge set (100 worlds × 1024 car boxes, R=256), flag
  agreement with the exact backend goes from 99.728% to 99.949%: misses drop from 259 to
  32, and the 20 false positives are unchanged. The query costs more: ~22–24 µs per
  CUDA-graph replay for 4096 worlds × 1 car box at R=256 on an RTX 5090 (7.8 µs before),
  and ~0.041 ms for the default 65,536-box benchmark (~0.018 ms before). It stays
  allocation-free and capturable. The `segments` backend is unchanged.
- **`sdf`: a box whose corners are not finite reads `oob=1`**, with NaN `distance`,
  `nearest` and `normal` and `boundary=-1`, as for a degenerate env; the grid is not read.
  This covers NaN or inf half-extents, yaw or position, and corners that overflow float32.
  Before, Warp's `clamp` sent a NaN grid coordinate to 0, so such a box reported grid texel
  (0, 0): that texel's phi as `distance`, its boundary id, and a zero normal.
- **`TrackGenerator` runs a `Program`.** Construction builds the default spec (the one
  `legacy.to_pipeline_spec(config)` returns), and `generate()`, `enqueue()` and `Course`
  run the Program. Outputs are byte-identical (the CPU fingerprint guard is unchanged),
  the launch sequence is the same (kernel, dimension and scalars, in order) and so is the
  launch count (152 for a default `bezier` generation at `E=2`, also after PR10b), and
  `Track` array addresses stay stable. The `relax_solver` and `smooth_finish` errors are
  raised while building the spec, with the same messages, in the same order (PR10b
  rewords the `relax_solver` one and adds `z_profile`'s, see below). `gen.params` still
  lists only `"generator"` for the default pipeline.
- **`gen._scratch` is a view** of the Program's buffers: the same `_Scratch` field names
  (`gen_centerline`, `cs_center`, `count`, `relax.relaxed`, `inflate.ctr2`, ...) alias the
  channel versions and stage scratch. A slot whose stage is absent is `None`
  (`_scratch.relax` with `relax_enable=False`).
- **The 0.2.0.dev2 composition stays, unused by the facade.** `warp_pipeline._run_pipeline`,
  `inflate_warp`, `_inflate_warp_alloc`, `_Scratch`, `RelaxScratch` and `InflateScratch`
  keep their code and the pinned legacy ABI; the tests use them as the byte-identity
  reference. They are removed once the parity tests carry a frozen copy.
- **Memory.** The re-uniformising resample has scratch of its own (`+E*(2*N_max+1)`
  float32), elevated profiles keep the plan-view arc-length tables apart from the 3D ones
  (`+E*(N_max+1)` float32), and `relax_enable=False` no longer allocates the relax
  scratch.
- **Channel buffers and stage scratch are zero-initialised** at construction (0.2.0.dev2
  used `wp.empty` for most; the `Track` is allocated as before). No buffer is read before
  it is written, so outputs do not change.
- **`relax_solver` and `z_profile` resolve through the registry** (roadmap PR10b).
  `TrackGenConfig.relax_solver` names any registered relax stage (`"xpbd"` default) and
  `TrackGenConfig.z_profile` any registered elevation stage (`"flat"` default); a plugin
  stage gets its `Params.from_legacy(config)` and a non-flat elevation lifts with
  `lift.elevated`. The names are resolved at `TrackGenerator` construction, in the order
  `relax_solver`, `smooth_finish`, `z_profile`, with new messages:
  `TrackGenConfig.relax_solver: unknown relax 'bogus'; available: ['none', 'xpbd']`, an
  oracle-only `"energy"` / `"tp_sobolev"` named as such, and
  `TrackGenConfig.z_profile: unknown elevation 'bogus'; available: ['flat', 'noise',
  'random_walk', 'uniform']` (plus installed plugins); a name served by a quarantined
  plugin raises its `PluginLoadError`. **`TrackGenConfig` construction no longer checks
  `z_profile` against a fixed set**: `TrackGenConfig(z_profile="bogus")` constructs and
  `TrackGenerator` raises; the config only requires a non-empty string, and the `z_*`
  range rules stay. `GateGenConfig.z_profile` keeps its closed set (PR11).
- **The in-place model is a versioned rewrite, with no aliasing.** A stage rewrites a
  channel by declaring it in `reads` and `writes` through one port; `b.read(port)` is
  the previous version and `b.write(port, ...)` a new buffer every later stage reads.
  The stage writes its whole output there and never writes a read buffer. This is
  PR10a's behaviour, now the documented model (it keeps the built-ins byte-identical,
  costs one buffer per rewrite, and makes `tap(after=...)` meaningful). There is no
  aliasing form (`inplace=`) in this release.
- **What the commit costs.** An unmasked generation of a pipeline whose last check is
  `validity.default`, with `on_invalid="keep"` (every default pipeline), launches
  exactly what it launched before (152 launches for a default `bezier` generation at
  `E=2`; the parity test is unchanged). A check after `validity.default`, or
  `on_invalid="poison"`, adds one launch (`E` threads to derive `valid`, `E * N_max` for
  the poison commit); a masked generation adds two (the hold and the commit, `E * N_max`
  threads each, however few envs are masked). The held copy of the `Track` adds 64 B per
  point and 24 B per env (about 100 MB at `E=4096`, `N_max=384`), allocated once at
  construction for every Program, masked use or not. Outputs of the default pipelines
  are byte-identical (the fingerprint guard and the parity test are unchanged), and
  pointers stay stable.
- **`Track.valid == (fail_mask == 0)`.** `Track.valid` is still `validity.default`'s
  `track.valid` when it is the last check (equal by construction); after a plugin check
  the commit derives it from `valid.fail`. `build(..., out=track)` accepts a `track`
  whose `fail_mask` is `None` (the bitmask then lives in a buffer of the Program).
- **`_validity_k` gains a trailing `fail` output** (its other arguments, its name and its
  0/1 output are unchanged); `warp_pipeline.validity_inplace(...)` gains an optional
  `fail_wp=None` (allocated by the wrapper when `None`), `InflateScratch` an optional
  `fail` slot that `_inflate_warp_alloc` allocates, and `inflate_warp` passes it, so the
  0.2.0.dev2 composition stays allocation-free. `gen._scratch.inflate.fail` aliases the
  Program's `valid.fail`.
- **`gen.params` lists every stage with a dynamic field**, keyed by stage id, after
  `"generator"`: still `["generator"]` for the default pipeline (the built-in stages'
  fields are all static), `["generator", "relax"]` with a relax stage that has one.
- **`TrackGenerator(config)` accepts `rng=None`** (env `e` gets seed `e`), and a
  `seeds=` keyword (any `SeedState` form); previously `rng=None` raised `ValueError`.
- **Registration messages.** A generator `Stage` with read ports is rejected because a
  generator reads no channel. (PR10a's message for a `Stage` of another kind, "arrives
  with roadmap PR10b", is gone: PR10b accepts the four pipeline kinds, see Added, and
  names the kinds that stay closed.)

### Deferred

- The gate pipeline as stages, `presets.gates`, the ordering registry and the `stations`
  adapter; `GateGenConfig.z_profile` keeps its closed set of names (PR11).
- `BOUNDED` stages and HOST segmentation (PR12): a `HOST` stage makes the whole Program
  eager in this release.
- Channel carry-through (`raw.*` -> `cl.*`, `write(interp=, export=)`), `cl.half_width`
  consumers (offset, XPBD, thickness), a width stage before relax, `Track.half_width` /
  `Track.attributes`, dropping the per-stage `half_width` fields, `randomize.uniform`
  and removing the PR8 parameter latch (PR13).
- Split built-in checks (`validity.turning`, ...), an aliasing in-place write form,
  opening `resample` / `frames` / `offset` / `lift` to plugins, the commit as a
  registered kind, a `TrackGenConfig.on_invalid` field, `Course(PipelineSpec)`, and
  fusing the lift into the commit (which would drop most of the commit's memory).
- Deleting `_run_pipeline`, `inflate_warp`, `_Scratch`, `_StageGenerator` and the
  `_scratch` view (the parity tests and the fingerprint tool still use them).
- `on_invalid="poison"` as the default (1.0).

### Testing

- PR10b: `tests/test_fail_bits.py` (the `BUILTIN_FAIL_BITS` table and the default
  pipeline's names; `valid == (fail_mask == 0)` for every parity case of
  `test_pipeline_parity`; each built-in check setting its own bit on hand-built inputs
  through `validity_inplace(fail_wp=...)`, an empty loop failing `turning`, and
  `validity_inplace` still running without `fail_wp`; plugin bits assigned in pipeline
  order; the `fail_bit` errors: name clashes, a 16th plugin bit, a validity stage without
  a bit, a plugin reserving built-in bits, a call from another kind).
  `tests/test_commit.py` (the `Track` arrays are the final channel versions; a derived
  `valid`; a masked regeneration never changing an unmasked env's bytes although a stage's
  output is perturbed on every call; an all-zero mask writing nothing; masked envs equal
  to a fresh generator at the same epoch; `"poison"` equal to `"keep"` with the invalid
  envs poisoned, for a failed check and for truncation, and touching masked envs only
  under a mask; `TrackLocalizer`, `CollisionChecker`, `CheckpointSampler` and
  `PropSampler` taking their degenerate paths on a poisoned env; `out=` with and without
  `fail_mask`; `on_invalid` validation; the launch counts: 152 plain, two more masked,
  one more with poison or a check after `validity.default`).
  `tests/test_pipeline_presets.py` (the default preset byte-identical to the config's
  spec; the stage list; `half_width` reaching every stage that bakes it; every selection
  form and error; check ids and generator forms; edit operations; the JSON round trip
  with plugin stages building byte-identical; `TrackGenerator(spec)` equal to
  `build(spec)` and to the config; the `seeds` forms and argument errors; `gen.params` holding the
  stages with dynamic fields; a plugin `z_profile` lifting with `lift.elevated`).
  `tests/test_stage_plugins.py`, with the toy plugins of `tests/_stage_toys.py` (relax
  `acme.taubin`, elevation `acme.sinusoid_hills`, validity `acme.max_curvature`)
  installed through a fake entry point: discovery and `describe()`; `scoped()` restoring
  the stage kinds; each toy byte-equal through the presets and the flat shortcut; the
  rewrite leaving the version it read intact; live dynamic parameters; the hills
  altitude against a host evaluation; the curvature bit set exactly where `cl.kappa`
  exceeds the threshold; two checks getting bits in pipeline order; the three toys in
  one Program; each toy passing `check_stage`, with params and `upstream`; the broken
  stages of `tests/_broken_stage_toys.py` (writing a read port, stateful, skipping the
  padding, OR-ing a built-in bit, dropping earlier bits, writing failing envs only,
  allocating) failing the named checks; a plugin registering an invalid
  stage quarantined; unknown names failing at build. Existing files: the open and closed
  kinds (`test_plugin_stage.py`), open-kind tests moved to a `"banking"` kind
  (`test_plugin_registry.py`, `test_conformance.py`), a non-Stage spec refused under a
  pipeline kind, the planner's toy stages registered through `registry.scoped()`
  (`test_pipeline_planner.py`), the `z_profile` check moved to `TrackGenerator`
  (`test_types.py`), and the no-host-sync drivers `program[toys,poison]` (plain and
  masked) and `track_run[z=acme.sinusoid_hills]` with the new modules scanned.
- PR10b, CUDA (`tests/test_stages_cuda.py`): each toy under `check_stage` on `cuda:0`
  (caller capture, self-graph; the relax toy on a thinner band so `output.accepted`
  runs); each toy selected through the presets and recorded into a caller's capture;
  a dynamic stage parameter written between replays of one recording; one Program with
  the three toys and `on_invalid="poison"` recorded into a caller's capture, with
  parameters written inside the graph; the masked commit in a caller's graph (the whole
  `Track`, `generation` included, is poisoned before each replay: unmasked envs must
  still hold the poison, masked envs equal an eager twin), also when the relax output
  is perturbed differently on every replay. The existing in-place regeneration, Program
  capture, stream capture and CUDA fingerprint tests run unchanged.
- `tests/test_collision_sdf_outline.py`: the `sdf` outline query. Side crossings with all 4
  corners on the road: the analytic annulus (inner radius 1 m, R=41 and 256; cpu and cuda),
  a long box spanning 20 cells, and the tightest bend of racing track 1000000 against an
  exact reference. Random boxes on a coarse grid, some past the grid edge: `distance`
  equals the minimum of a dense float64 replica of the interpolated field on the outline,
  never exceeds the 5-sample minimum, and its argmin, `nearest` and `normal` agree. Boxes
  partly or fully outside the grid, far away (1e30) or swallowing the track; non-finite
  half-extents or yaw; degenerate envs; determinism; `oob == (distance < 0)` on the racing
  near-edge set. Slow: 100 racing worlds × 16 boxes (agreement ≥ 99.95%, measured 100%), and
  100 × 1024 boxes (99.949%), where every disagreement is checked to be the grid's, with no
  new false positive against the 5-sample query.
- `benchmarks/benchmark_collision.py --racing`: racing near-edge mode (bezier tracks at
  `scale=17`, `half_width=0.5`, `N_max=512`, seeds 1000000 + env; a 0.46 × 0.34 m car box near
  the edges; R=256): CUDA-graph time per query for both backends and the `sdf` flag
  agreement, misses and false positives. Both modes now report the graph replay as GPU time
  from events (the default mode used wall-clock time).
- `tests/test_pipeline_planner.py`: the default stage list for the flat and elevated
  profiles and without relax; the `relax_solver` / `smooth_finish` errors; the flat
  params validating with the `TrackGenConfig` messages; `StageRef` as frozen data; the
  edit operations; the `to_dict` / `from_dict` round trip (JSON, byte-identical build),
  a foreign format and missing keys; a standalone `build()` equal to `TrackGenerator`;
  `out=` binding and mismatch; every `seeds` form, other seed types refused before any
  allocation and the `SeedState` checks before the `Track`; `run()` preparing a
  `prepare=False` Program and refusing to record it unprepared; every planner error
  (unknown kind or name, duplicate ids, the generator slot, a read without an earlier
  writer, a `Track` field without a writer, `lift.flat` after a non-flat elevation,
  `capacity` keys, the `half_width` disagreement, port rewiring, batch and params
  errors), each raised before any allocation; `Channel.per` validation; an elevated
  lift after a replaced elevation; `build` inside a capture; toy stages injected into the private catalogue
  (a toy elevation stage end to end, a live dynamic parameter of a stage, the
  `StageBuilder` rules, the catalogue rules, input and salt collisions, salts keyed by
  stage id); channel versioning with and without relax and the elevated lift's own arc
  tables; the `_scratch` view aliases; `gen.params` staying `{"generator"}`; no
  allocation in `Program.enqueue`; a Stage generator through the Program;
  `temp_like()` in generator mode; and `import track_gen` not importing the pipeline.
- `tests/test_pipeline_parity.py`: for every built-in generator (and the bezier, hull
  and voronoi fallbacks, `style_sampling`, `relax_enable=False`, `z_base`, every
  `z_profile`, `validity_border_check`, `relax_accel="none"`, `relax_sep_cache_slots=0`,
  `relax_band=2`, `N_max` truncation, the toy Stage generators and the toy legacy-ABI
  generators), after a plain and a masked generation, every channel version the Program
  writes equals, byte for byte on CPU, the buffer of the 0.2.0.dev2 `_run_pipeline` it
  reproduces, and every `Track` field matches; the launch sequences (op, kernel, dim,
  scalars) are equal; a default `bezier` generation at `E=2` is still 152 launches.
  PR10b leaves the file unchanged.
- The no-host-sync and launch-stream drivers cover the Program (`program[standalone]`)
  and the 0.2.0.dev2 composition on a generator's `_scratch` view
  (`wrappers[legacy_pipeline]`); the pipeline and stage modules are scanned.
- `tests/test_caller_capture_cpu.py`'s fake facade patches
  `track_gen.pipeline.legacy._to_pipeline_spec` and `track_gen.stages.track._alloc_track`
  instead of `_run_pipeline` / `_inflate_warp_alloc`; its assertions are unchanged.
- CUDA: `tests/test_program_capture_cuda.py` builds a standalone Program (default, noise
  elevation without relax, random-walk elevation) and checks that it is caller
  capturable (Warp `ScopedCapture`, a registered `torch.cuda.graph` and a plain one:
  poisoned replays equal an eager twin, before and after a reseed), that its self-graph
  replays the eager batch (also for the masked-advance key), and that on CUDA too it is
  the 0.2.0.dev2 composition byte for byte with the same launch sequence. The existing
  caller-capture, stream-capture and in-place regeneration CUDA tests cover
  `TrackGenerator` unchanged.

## 0.2.0.dev2 (2026-10-22, tag `v0.2.0.dev2`)

### Added

- **`track_gen.plugin`: one public registry, with entry points** (roadmap PR6). A
  pip-installable package can add a track or gate generator without importing
  `track_gen._src` and without editing TrackGen. It declares an entry point,
  `[project.entry-points."track_gen.plugins"] acme = "acme_tracks:register"`, whose
  `register(reg)` calls `reg.add(GeneratorSpec(name="acme.wobbly", ...))`;
  `TrackGenConfig(generator="acme.wobbly")` then selects it. The workflow and the
  generator contract are in `docs/contributing/writing-a-generator.rst`.
  - `track_gen.plugin.registry`, the process-wide `Registry`:
    `add(spec, *, kind=None, replace=False)` (the kind is inferred from a
    `GeneratorSpec` or `GateGeneratorSpec`, any other spec needs `kind=`: the set of
    kinds is open), `get(kind, name)`, `available(kind=None)` (sorted; every
    `"kind:name"` when `kind` is `None`), `describe(kind, name)` (a `SpecInfo`: kind,
    name, origin module (the module defining the spec's callables; a `functools.partial`
    or a decorated function counts as the function it wraps), distribution
    `(name, version)` of the registering entry point
    or `None` for a built-in and for a spec added at run time, `capturable`,
    `accepts_ctx`, `builtin` and `entry_point`), `unregister(kind, name)`,
    `scoped(*, fresh=False)` (snapshots every kind and restores it on exit, for tests;
    `fresh=True` starts from an empty, unloaded registry so discovery re-runs) and
    `quarantined` (a read-only mapping of plugin entry-point name to error text; keys
    are qualified as `"<distribution>:<entry point>"` when two quarantined plugins share
    an entry-point name). Adding the identical spec object again is a no-op.
  - `GeneratorSpec` and `GateGeneratorSpec` re-exported for plugin authors,
    `API_VERSION` (`1`), `ENTRY_POINT_GROUP` (`"track_gen.plugins"`), `POLICY_ENV`
    (`"TRACK_GEN_PLUGINS"`), and `DuplicateSpecError` (a `ValueError`),
    `PluginOverrideWarning` (a `UserWarning`) and `PluginLoadError` (a `RuntimeError`).
  - Loading is lazy (on the first registry call: `get`, `available`, `describe`, `add`,
    `unregister`, `quarantined`, `scoped()` or an old `_src` registry shim), runs once
    per process under a lock, and is idempotent. The built-ins register first, without
    an entry point, so a source checkout works. Then every entry point of the
    `track_gen.plugins` group, sorted by distribution name and entry-point name, is
    loaded and called with the registry. A distribution with unreadable metadata never
    breaks discovery: its entry point loads with an unknown distribution.
  - Load policy, from the `TRACK_GEN_PLUGINS` environment variable (case-insensitive).
    `warn` (default) quarantines a plugin that fails to import or register, takes a
    name that is already registered without `replace=True`, unregisters a name it did
    not register, or declares a newer plugin API than `API_VERSION` (`track_gen_api =
    <int>` in the module its entry point names or on the register callable; an `int`
    `API_VERSION` there is read as a fallback), with one `RuntimeWarning`, and the other
    plugins still load. `strict` raises `PluginLoadError` instead. `off` skips
    third-party discovery. Any other value emits a `RuntimeWarning` and falls back to
    `warn`. An unknown-name error lists the available names and the quarantined plugins
    with their errors; asking for a name a quarantined plugin tried to register, or one
    namespaced under its entry-point name, entry-point module or distribution name,
    raises `PluginLoadError` chained from its original error.
  - Third-party names are dotted (`pkg.name`). A name without a dot still registers,
    with a one-time `UserWarning`. Built-in names are unchanged.
- **`track_gen.ops`: public device building blocks for generators** (roadmap PR7). The
  primitives the built-in generators are made of are now public and documented, so a
  plugin generator no longer needs `track_gen._src` helpers. Every op takes the call's
  `LaunchContext` first and launches only through it. It never allocates, reads back or
  host-synchronizes: working buffers are arguments, allocated once in `alloc_scratch`. It
  takes no config object (sizes, salts and bounds are explicit arguments), uses the flat
  per-env layout, and validates its arguments on the host before any launch: a wrong
  type, shape, dtype or device raises `ValueError` naming the argument (except that
  `fill` raises `TypeError` for an unsupported dtype and `any_set` for an argument that
  is not an `int32` `wp.array`). The ops modules import only Warp: `import track_gen.ops`
  loads no generator module and no torch.
  `docs/contributing/writing-a-generator.rst` gains a "Building blocks" section with a
  generator assembled from ops only, and `docs/reference/api.rst` documents every op.
  - One set of conventions: `src` is the input point array and `stride` its points per
    env, `num` an exact output count per env and `out_stride` an output row capacity;
    `E` is `src.shape[0] // stride` (the samplers take it from `seeds`) and every other
    per-env array must be exactly `[E]`; `count` and the env mask `active` are always
    keyword-only, and a `count[e]` above the stride is clamped to it. `ctx.device` is
    canonicalised, so a context whose device is `"cuda"` accepts arrays on `cuda:0`.
  - Resampling: `arc_resample(ctx, src, stride, num, scratch, out, count_out, *,
    active=None)` is NaN-aware. It drops NaN points, gives a row with fewer than two real
    points NaN and count 0, and with `active` resamples only the flagged envs, leaving
    the other rows bit for bit. Also `resample_uniform(ctx, src, stride, scratch, out, *,
    count)` (NaN after `count[e]`) and `resample_constant_spacing(ctx, src, stride,
    spacing, out_stride, scratch, out, count_out, truncated_out)` (per-env count and
    truncation flag, both on device). Their working buffers are an `ArcScratch`:
    `ArcScratch.alloc(E, stride, device, *, compact=True)`, or wrap arrays you already
    own. `out` must not alias the scratch, nor `src` for the two count-based resamples.
  - Geometry: `self_intersections(ctx, src, stride, out, *, count)`,
    `turning_number(ctx, src, stride, out, *, count)`,
    `thickness(ctx, src, stride, band, out, *, count)`,
    `normalize_extent(ctx, src, stride, target_extent, *, count=None)` and
    `angle_sort(ctx, src, stride, keys, out, *, count, out_stride=None)` (the order the
    built-ins emit for `gate_ordering="ccw"`, NaN after `count[e]`).
  - Sampling: `sample_counts(ctx, seeds, salt, offset, lo, hi, out)`,
    `sample_grid_points(ctx, seeds, salt, offset, num, num_cells, cell_size, scale, used,
    out)` (a float or a per-env `[E]` scale; distinct cells are best effort and the
    jitter is an absolute `[-0.5, 0.5)`) and the host helper
    `grid_cells(min_point_distance)` (the built-in grid; `ValueError` outside
    `(0, 0.5]`). Streams are keyed as the built-ins key theirs,
    `wp.rand_init(seeds[e] * salt + offset)` in int32 arithmetic, with the salt and offset
    as arguments; the generator guide lists the pairs the built-ins use.
  - Curves: `closed_catmull_rom(ctx, src, stride, per_seg, out, *, count=None,
    endpoint=False)` and `closed_polyline(ctx, src, stride, per_seg, out, *, count=None,
    endpoint=False, active=None)`, the straight polyline in the same layout (with
    `active`, the fallback for flagged envs).
  - `track_gen.ops.funcs`: the `@wp.func` helpers for plugin kernels, `ccw`, `cross2`,
    `nan0`, `safe_normalize2`, `catmull_rom`, `segments_cross` and `hash32`. They perform
    the built-in kernels' operations in the same order: bitwise equal on CPU; on CUDA,
    FMA contraction can differ when the surrounding kernel differs.
  - `check_array(name, arr, shape, dtype, device)`, the library's array validator (the
    facades' `runtime._check_arr` is now this function; it now canonicalises `device`);
    `track_gen.ops.constants.TARGET_EXTENT` (`1.44`), the longest bounding-box side per
    unit `scale` that `polar`, `voronoi` and `checkpoint` normalise to; and `ops.fill` now
    also fills `quatf` arrays (from four components) and raises `ValueError` for an array
    that is not a `wp.array` or not on `ctx.device`.
- **Per-generator `Params`, `Stage` generators and `describe()` schemas** (roadmap PR8).
  A generator declares its own parameters, so a plugin no longer subclasses
  `TrackGenConfig` to add knobs: users pass
  `TrackGenConfig(generator="acme.x", generator_params=AcmeParams(m=7.0))`, or a mapping
  of its fields. `docs/contributing/writing-a-generator.rst` gains a Stage quick start
  and sections on Params, the Stage contract and its builder, runtime parameters and
  `describe()`; `docs/configuration/reference.rst` documents `generator_params`, every
  built-in Params class and the deprecated fields.
  - `track_gen.plugin.Params`, the base of a stage's parameters: a frozen, keyword-only
    dataclass whose fields all have defaults. A field is `static(default, *, lo=None,
    hi=None, choices=None, doc="", ui=None)`, a host value fixed at construction (a plain
    field counts as static), or `dynamic(default, *, lo=None, hi=None, doc="", ui=None)`,
    an `[E]` device array per field (`float` -> `wp.float32`, `int` / `bool` ->
    `wp.int32`) that the stage reads by pointer. `lo` / `hi` are inclusive.
    `Params.validate(layout)` checks the object's own fields and runs at generator
    construction, for the selected generator only; `Params.from_legacy(cfg)` gives the
    instance used when `generator_params` is `None` (`cls()` by default; a plugin
    migrating off a config subclass overrides it). `NoParams` is the class of a stage
    without parameters. `registry.add` checks the class: decorated as a frozen
    dataclass itself, every field with a default inside its own bounds and choices, no
    field named `validate`, `from_legacy`, `to_dict`, `from_dict`, `replace`, `schema`,
    `fields` or `describe` or starting with `_` (reserved for later Params helpers), and
    dynamic fields annotated `float`, `int` or `bool`. A static field annotated `int`,
    `float` or `bool` must hold a value of that type (`ValueError` otherwise, checked
    with its bounds); the built-ins' flat Params keep the legacy configs' leniency.
  - `track_gen.plugin.Stage`, the generator subset of the stage protocol. A `Stage`
    subclass with `kind = "generator"` and `output = "centerline"` is registered as a
    class (`registry.add(MyStage)`) and selected by its `name`. It declares a `Params`
    class, `writes` (local port -> channel: `"raw.xy"`, required, and `"gen.valid"`),
    a `capture` class, `topology` (`"closed"` or `"any"`) and `api_version`, and
    implements `build(b, params) -> enqueue(ctx)`, plus optionally `capacity(params,
    layout)` (`{"raw": n}` raw points per env, any integral `n >= 3`, default
    `num_points`; another key raises `ValueError`) and `capture_class(params)`. These
    three receive the params with their static fields only: reading a dynamic field
    raises `AttributeError` (read it on device through `b.param`), so no host scalar can
    bake a dynamic value into the launches. `registry.add` checks the declaration
    (`reads` and `provides` must be empty until PR10 / PR11; a kind other than
    `"generator"` is rejected until PR10, and `capture=BOUNDED` until PR12).
  - `track_gen.plugin.StageBuilder`, handed to `build` and created by the orchestrator
    only: `E`, `layout`, `device`, `stage_id` (`"generator"`), `scale`, `seed` (the
    `[E]` effective seed of each generation), `raw`; `write(port, dtype, *, per)` (the
    pipeline-owned output buffer, read without a copy), `scratch(name, dtype, shape)`
    (zeroed once), `param(field)` (a dynamic field's working array, read-only for the
    stage), `input(name, dtype, default=0)` (a per-env device input, read-only likewise)
    and `salt(stream)`
    (`hash32_host(fnv1a_32(f"{stage_id}/{stream}".encode()))`, frozen for plugin API
    version 1). `gen.valid` is set to 1 once when a stage never writes it.
  - `track_gen.plugin.CaptureClass` (`STATIC`; `BOUNDED`, reserved: a stage declaring
    or returning it is rejected until `ctx.bounded_loop` arrives in PR12; `HOST`, eager
    on every call and refused by a caller's capture with `NotCapturableError`) and
    `track_gen.plugin.Layout`
    (`num_envs`, `stride`, `device`, `closed`), handed to `validate`, `capacity` and the
    builder.
  - `TrackGenConfig.generator_params` and `GateGenConfig.generator_params` (the last
    field of each): `None` (the default: the generator's `from_legacy(config)`; for the
    built-ins, the flat fields), a Params instance, or a mapping of its fields. A
    mapping or an instance is the generator's complete parameter set: left-out keys take
    the class defaults, unknown keys raise `ValueError` listing the valid fields, and a
    flat field the selected built-in reads must stay at its default (the Params class
    default, which equals the base config's) or equal the given value. A mapping's
    values are converted by each field's annotation (arrays to tuples, also inside
    `T | None`; enum values to members), so a JSON, YAML or Hydra mapping shaped like the
    schema gives an equal, hashable instance. Resolved and validated when the generator
    is constructed.
  - `TrackGenerator.params` and `GateGenerator.params`: a `ParamTable`, the read-only
    mapping `{"generator": StageParams}` (keyed by stage id, as later multi-stage
    programs will be; their constructors are private). `StageParams.params` is the
    resolved Params instance as of construction (initial dynamic values; so is
    `gen.config`); each dynamic field maps to its live `[E]` array, with a stable pointer;
    `StageParams.set(field, value, *, stream=None)` (and `ParamTable.set(stage_id, field,
    value, *, stream=None)`) writes a host scalar (checked against `lo` / `hi`, one fill)
    or an `[E]` array (one copy, no host sync) in place on the resolved stream: recorded
    when that stream is capturing, never allocating. Each generation latches the live
    arrays into the stage's working copies for the envs it regenerates (every env, or
    `mask[e] != 0`), so a write takes effect at the next generation, including between
    replays of a caller's CUDA graph without re-capture, and a masked regeneration still
    leaves the other envs bit for bit. `TrackGenerator.inputs` also lists every
    `StageBuilder.input`, latched the same way.
  - `SpecInfo.params_schema`: the JSON schema (draft 2020-12) of the spec's Params
    class, per field `type`, `default`, `minimum` / `maximum` (from `lo` / `hi`), `enum`
    (choices), `description` (doc), `x-static` or `x-dynamic` (with `x-dtype`) and `x-ui`;
    `{}` for a spec without one (a legacy spec without `params=`, or a Stage with
    `NoParams`). Always `json.dumps`-able. For a Stage, `describe()` reports
    `capturable` from `capture_class` at the Params defaults (`False` for `HOST`),
    `accepts_ctx=True`, and the new `SpecInfo` fields `capture`, `output`, `writes` and
    `provides` (`None`, `None`, `{}` and empty for a spec that is not a Stage).
  - `GeneratorSpec.params` and `GateGeneratorSpec.params` (a new last field, default
    `None`): a Params class with static fields only for a callable-based generator, read
    as `config.generator_params` by `generate`, `alloc_scratch` and `max_gates`.
  - A Params class for every built-in: track `bezier`, `hull`, `polar`, `voronoi`,
    `checkpoint`, `repulsive`, and gate `bezier` / `hull`, `polar`, `voronoi`,
    `checkpoint`. Their fields are the flat config fields each generator reads, with the
    same names and defaults, all static. The orchestrator writes the resolved values onto
    its private config copy, so the generators read the same values as before:
    `polar_num_knots=9`, `generator_params={"polar_num_knots": 9}` and
    `generator_params=PolarParams(polar_num_knots=9)` give byte-identical outputs.
    The classes have no public import path: pass a mapping, or use
    `registry.get(kind, name).params` and `describe(kind, name).params_schema`.
  - `viz/param_explorer.py` builds controls for plugin generators from their
    `params_schema` (the built-ins keep their curated layout), and
    `benchmarks/compare_generators.py` passes each generator its own `generator_params`,
    so a plugin with a Params class runs from a plain `TrackGenConfig`.
  - `API_VERSION` stays `1`: all of this is additive. Gate generators stay
    `GateGeneratorSpec`s: `registry.add` rejects a Stage with `kind="gate_generator"` or
    `output="anchors"` until roadmap PR11.
- **`track_gen.testing`: a conformance kit for generator plugins** (roadmap PR9). A
  plugin author runs one call in their own CI,
  `check_generator("acme.x", params={...}).raise_for_failures()`, and learns whether the
  generator honours the contract; TrackGen's own suite runs the same checks over every
  built-in. The new page `docs/contributing/testing-a-plugin.rst` covers the kit, the
  writing-a-generator guide links to it, and `docs/reference/api.rst` documents it.
  `track_gen.testing` is outside `track_gen.__all__`; `import track_gen` does not import
  it, and importing it imports neither torch nor pytest.
  - `check_generator(name, params=None, *, devices=None, num_envs=8, seeds=(11, 1009),
    config=None, capture_modes=CAPTURE_MODES, skip=(), modules=())` and
    `check_gate_generator(...)`
    (the same arguments, `config` holding `GateGenConfig` fields) run a registered
    generator on `cpu`, plus `cuda:0` when Warp sees a CUDA device, and return a
    `Report`: one `CheckResult(name, device, passed, message, skipped, details)` per
    check and device, `ok`, `failures`, `failed_names`, `result(name, device=None)`,
    `raise_for_failures()` (one `AssertionError` listing every failure) and a table as
    `str()`. `CHECKS[kind]` lists the stable check names in run order: `construct`;
    `output.shape`, `output.finite` (raw and final output finite for the envs marked
    valid; the raw output is poisoned before the generation the checks read, so an
    unwritten point or `out_valid` flag is reported as such), `output.padding` (NaN
    past `count[e]`), `output.count`, and `output.closed` (the raw loop closes, its
    closing edge no longer than 3x the median step or 1.5x the longest step, and at
    least one env is valid) or, for gates, `output.accepted`; `determinism` (two fresh
    generators, bitwise), `diversity` (over the envs valid in both batches, with the
    smallest raw-output differences in `details`), `batch_independence` (env `e`
    alone, inside a larger batch and under a permutation of the other envs' seeds),
    `statelessness` (repeat calls, reseed and reseed back); on CPU
    `hot_path.host_sync`, `hot_path.allocation`, `hot_path.stream`, `hot_path.capture`,
    `hot_path.topology` and `hot_path.buffers` (every pointer owned by the generator,
    and none shared with a second generator built alike); on CUDA `capture.caller` and
    `capture.self_graph`. A `capturable=False` generator has the host-sync, allocation,
    topology, buffers and caller-capture checks reported as skipped. Each failure
    message names the env, field or call site (`module.function:line`) and the rule
    broken. `modules=` adds packages for `hot_path.stream` to probe, besides
    `track_gen` and the generator's own top-level package.
  - `track_gen.testing.contract`: `hot_path_guard(*, host_sync, allocation, stream,
    capture, modules=("track_gen",), allow=(), strict=True)` and the single-purpose
    `no_host_sync()`, `no_allocation()` and `stream_explicit(modules)`, context managers
    returning a `HotPathGuard`. They record (and, when strict, raise `HotPathViolation`
    on) `wp.synchronize*`, `wp.array.numpy()` / `list()`, `np.asarray` of a CPU array
    and a `wp.copy` between two devices; every Warp array allocation (seen at Warp's
    allocation chokepoint) and every `wp.HashGrid` (and `HashGrid.reserve`),
    `wp.Mesh`, `wp.Volume` and `wp.Bvh`; a `wp.launch` / `wp.launch_tiled` /
    `wp.Launch.launch` / `wp.copy` from the probed modules (or from a Warp utility they
    called) without `stream=`, `wp.array.assign()` / `zero_()` / `fill_()` and
    `wp.ScopedStream` from them; and nested `ScopedCapture` / `capture_*` calls. Each
    breach is a `Violation` with its category (`CATEGORIES`), Warp API and call site (a
    `Site`). The patches are installed while a guard is active, wherever a loaded
    module binds a Warp entry point (`from warp import launch` included; classes keep
    their identity, only their `__init__` is wrapped), and dispatch on a context
    variable, so other threads are not seen; a guard exited in another context than
    it was entered in still restores Warp.
  - `track_gen.testing.trace`: `trace_launches()` records every `wp.launch` /
    `wp.launch_tiled` / `wp.Launch.launch` / `wp.copy` / `wp.array.assign` as a
    `LaunchRecord` (kernel, dim, stream handle, array pointers, scalar arguments, with
    a `wp.struct` argument flattened into its fields, call site) in a `LaunchTrace`,
    with `diff()` / `same_topology()` and `foreign_pointers()`; `buffers_of(obj)`
    collects the `wp.array`s reachable from a producer (`wp.struct` fields included,
    registry specs excluded: they are shared by every generator built from them) into
    a `BufferSet`; `assert_same_topology` and `assert_pointers_within`.
  - `track_gen.testing.capture` (CUDA): `assert_caller_capturable(producer, *,
    modes=("warp", "external", "torch"), record=None, twin=None, reseed=None)` records
    `producer.generate(stream=s)` into a caller's capture on a side stream with the
    mempool disabled and `wp.capture_launch` forbidden, poisons the outputs with finite,
    dtype-aware sentinels, replays, and compares with an eager twin before and after a
    reseed. The torch modes import torch inside the mode, run on the producer's device,
    and are skipped without torch. Its
    helpers (`side_stream`, `mempool_disabled`, `capture_launch_forbidden`,
    `recording`, `poison`, `poison_all`, `result_arrays`) are public.
  - `track_gen.testing.pytest_plugin`, opt-in (no `pytest11` entry point): load it with
    `pytest_plugins = ["track_gen.testing.pytest_plugin"]` for the
    `track_gen_conformance(name, params=None, *, kind="generator", **kwargs)` fixture
    factory, which fails the test listing every failed check, plus
    `generator_names(kind, *, builtin=None, exclude=())` and `parametrize_generators(...)`
    over the registry.

### Behaviour changes

- **Duplicate names raise.** `registry.add` raises `DuplicateSpecError`, naming both
  origins (the spec's module, and the distribution for a plugin), when the name is
  already registered in that kind. `replace=True` overrides it and emits a
  `PluginOverrideWarning` naming both. Built-ins register first, so a plugin that
  reuses a built-in name without `replace=True` never replaces it: the plugin is
  quarantined (`warn`) or raises (`strict`).
- **Built-ins are registered explicitly.** Each shipped generator module now defines a
  `register(reg)` function instead of registering at import time (the module-level
  `register(...)` calls and the gate modules' `register_specs()` are gone);
  `track_gen._src.builtin_plugins.register(reg)` imports the modules and calls them.
  Importing a generator module no longer registers anything.
- **The private registries are shims over `track_gen.plugin.registry`.**
  `track_gen._src.generator_registry` and `gate_generator_registry` still define
  `GeneratorSpec` / `GateGeneratorSpec` and keep `get`, `available`, `register` and
  `_ensure_loaded`, which now delegate to the registry. Each old `register(spec)` keeps
  its duplicate rule: the track registry's overwrites an existing name
  (`registry.add(spec, replace=True)`); the gate registry's still raises when another
  module registered the name (now `DuplicateSpecError`, still a `ValueError`) and
  overwrites a spec from the same module. Both emit a `DeprecationWarning` when they
  overwrite, and re-registering the identical spec object is a silent no-op.
  `GENERATORS` and `GATE_GENERATORS` are now read-only views of the registry: code that
  assigned into them (or reset `_LOADED`) should use `registry.scoped()` and
  `registry.add(...)`.
- `TrackGenerator`, `GateGenerator`, `warp_pipeline` and `Course` resolve generator
  specs through `track_gen.plugin.registry`. `import track_gen` still imports no
  generator module, loads no plugin and imports neither torch nor scipy. Built-in
  generator outputs are byte-identical to the previous release.
- **The built-in generators are built from `track_gen.ops`** (roadmap PR7). Every track
  generator (`bezier`, `hull`, `polar`, `voronoi`, `checkpoint`, `repulsive`) calls the
  public ops instead of the `warp_pipeline` (`_pipe._*`) helpers, and the gate
  generators' shared sampling, ordering and normalisation steps run on the same ops. No
  gate module imports an underscore name from a track generator module any more: what
  they share is now public in the generator modules (`warp_generate.CORNER_COUNT_SALT` /
  `CORNER_SALT`, `warp_generate_hull.POINT_COUNT_SALT` / `POINT_SALT` /
  `POINT_SALT_OFFSET`, `warp_generate_polar.BASE_RADIUS` /
  `polar_num_knots` / `polar_jitters` / `sample_polar_controls`,
  `warp_generate_voronoi.BOX_SIZE` / `voronoi_layout_mode` / `sample_voronoi_sites` /
  `select_voronoi_anchors`, `warp_generate_checkpoint.BASE_RADIUS` /
  `sample_checkpoint_rings`), which stay in `track_gen._src`. The `bezier` and `hull` gate
  generators no longer pass a `GateGenConfig` where a `TrackGenConfig` is expected: they
  call `ops.sample_counts` / `ops.sample_grid_points` with the track generators' salts.
  Two steps keep a private form: `bezier` and `hull` sort with the `ops.angle_sort`
  kernel but the legacy count floor of 1 (so their output never depends on
  `TrackGenConfig`'s `min_num_points >= 2` check, which a config mutated after
  construction skips), and `polar` keeps its one-thread-per-env spline kernel (see
  Deferred to PR16). `tests/test_ops.py` checks on the source that no generator module
  imports `warp_pipeline` or a private `track_gen.ops` name beyond the legacy wrappers'
  launchers, and that no gate module uses an underscore name of a track generator.
- **Duplicated kernels are merged where the result is byte-identical**: the count sampler
  (`bezier`, `hull`); the grid point sampler (`bezier` with a scalar or per-env scale,
  `hull`); the centroid angle sort (`bezier`, `hull`, the gates' `"ccw"` ordering); the
  extent normalisation (`polar`, `voronoi`, `checkpoint`, plus the gates' count-aware
  variant as `normalize_extent(count=...)`); the Catmull-Rom point (`polar`, `voronoi`,
  `hull`'s inline copy) and the per-sample dense kernels of `voronoi` and `hull`; the
  polygon fallback (`hull`, `voronoi`);
  the safe 2-D normalisation (`bezier`, `warp_gate`, `repulsive`'s `_safe_dir`); the
  `warp_pipeline` and `warp_gate` fill kernels; and the `_ccw`, `_nan0`, `_cross2` and
  `_proper_segment_intersection` device functions. The resample, self-intersection,
  turning-number and thickness kernels moved from `warp_pipeline` into `track_gen.ops`
  unchanged apart from the count clamp below. Built-in outputs are byte-identical to
  before the port on CPU: a strict byte comparison of a before/after snapshot of every
  track and gate generator and every legacy helper (including seeds whose `seed * salt`
  overflows int32, and configurations that force the polygon fallback), and
  `tests/test_generator_fingerprints.py`. Each merged kernel computes the same operations
  in the same order and keeps its kernel structure (one thread per sample or per env as
  before), so CUDA, where Warp contracts multiply-adds into FMAs, should match too; the
  CUDA comparison has not been run yet and is required before merging.
- **A `count[e]` above the stride is clamped** in the count-aware resample,
  self-intersection, turning-number, thickness and curve kernels. It read (and the
  uniform resample wrote) past the env's row before, into the next env or past the end
  of the buffer. This also applies to the legacy helpers that share the kernels
  (`resample_uniform`, `self_intersections_inplace`, `turning_number_inplace`); valid
  counts give the same results.
- **The `warp_pipeline` helpers delegate to the ops.** `_arc_resample_inplace`,
  `_arc_resample_selected_inplace`, `self_intersections_inplace`,
  `arc_length_resample_inplace`, `arc_length_resample_selected_inplace`,
  `resample_uniform`, `resample_constant_spacing` and `turning_number_inplace` keep their
  names, parameter lists, `ctx=None` defaults, allocate-if-`None` branches and debug-sync
  placement; so do `warp_generate.corner_count_sample_inplace`, `corner_sample_inplace`
  and `ccw_sort_inplace`, and `warp_generate_hull.point_count_sample_inplace` and
  `point_sample_inplace`. The kernels and functions that moved out of `warp_pipeline`
  (`_pipe._fill_i32_k`, `_fill_f32_k`, `_ccw`, `_nan0`, the resample, crossing,
  turning-number and thickness kernels) are importable under their old names and are now
  the `track_gen.ops` objects (same argument types and order; their Warp module is now a
  `track_gen.ops` module). `warp_gate`'s fill kernels are likewise the `ops.fill` kernels,
  and `warp_gate` now fills through `ops.fill`. Of the generator modules' private names,
  only those with a same-arithmetic replacement keep resolving:
  `warp_generate_polar._polar_num_knots` and `_catmull_rom`,
  `warp_generate_voronoi._voronoi_layout_mode`, and `_safe_normalize2` (`warp_generate`,
  hence `_pipe._safe_normalize2`, and `warp_gate`) and `warp_generate_repulsive._safe_dir`,
  all now the public objects.
- **Removed private names** (never part of the pinned ABI; a plugin that copied a
  built-in module may have imported them): `warp_generate._corner_count_sample_k`,
  `_corner_sample_k`, `_corner_sample_style_k` and `_ccw_sort_k` (also reachable as
  `_pipe.<name>` before); `warp_generate_hull._point_count_sample_k`, `_point_sample_k`,
  `_angle_sort_k`, `_catmull_rom_k`, `_aug_vertex` and `_assemble_polygon_selected_k`;
  `warp_generate_polar._normalize_centerline_k`; `warp_generate_voronoi._catmull_rom`,
  `_aug_at`, `_catmull_rom_dense_k`, `_normalize_centerline_k` and
  `_assemble_polygon_selected_k`; `warp_generate_checkpoint._normalize_centerline_k`;
  `warp_gate._order_ccw_k` and `_normalize_positions_k`; and every generator module's
  `_pipe` attribute (the module no longer imports `warp_pipeline`). Their replacements
  are the ops in the generator guide's table; the kernels' arguments differ (explicit
  salts, strides and masks).
- **`GenScratch` moved** from `warp_pipeline` to `warp_generate`, the module whose
  scratch it is (see Deprecated).
- With `TRACKGEN_DEBUG_SYNC=1`, the steps the built-ins now run through `track_gen.ops`
  no longer debug-sync one by one, since the ops never sync; each built-in `generate`
  keeps its final debug sync. Without the variable nothing changes (the sync was already
  a no-op).
- **Generator-specific validation moved from the configs to generator construction**
  (roadmap PR8). These rules ran in `TrackGenConfig.__post_init__` /
  `GateGenConfig.__post_init__` for every config. They now live in the built-in Params
  classes' `validate()` and raise the same `ValueError`, with the same message and
  condition, when a `TrackGenerator` or `GateGenerator` (and so a `Course`) is
  constructed, and only when the generator that owns them is selected:
  - track `bezier` and `hull`: `min_num_points >= 2`; `max_num_points >= min_num_points`;
    `0 < min_point_distance <= 0.5`;
  - track `bezier`, `hull`, `polar` and `voronoi`: `num_points_per_segment >= 2`;
  - track `voronoi`: `voronoi_control_points >= 6`; `voronoi_num_sites >=
    voronoi_control_points`; `voronoi_site_layout` in `{ring, void_ring, clustered,
    mixed}`; `voronoi_radial_variation >= 0`; `voronoi_angular_jitter >= 0`;
  - track `repulsive`: `repulsive_grow_mult_min >= 1`; `repulsive_grow_mult_max > 1` and
    `>= repulsive_grow_mult_min`; `repulsive_domain_frac > 0`;
    `repulsive_domain_init_ratio > 1`; `repulsive_obstacle_count_min >= 1`;
    `repulsive_obstacle_count_max >= repulsive_obstacle_count_min`;
    `repulsive_obstacle_radius_min_frac > 0`; `repulsive_obstacle_radius_max_frac >=
    repulsive_obstacle_radius_min_frac`; `repulsive_ratchet_rate > 0`;
    `repulsive_alpha > 0`; `repulsive_beta > 0`; `repulsive_stages` non-empty, positive
    multiples of 4 and
    strictly increasing; `repulsive_settle_iters`, `repulsive_resample_every` and
    `repulsive_stall_window >= 1`; and `repulsive_stages[-1] == num_points`, which
    already applied to `generator="repulsive"` only and now also waits for construction;
  - track `checkpoint`: `checkpoint_count >= 3`; `checkpoint_best_of_k >= 1`;
    `checkpoint_radius_min_frac` in `[0, 1)`;
  - gate `bezier` and `hull`: `min_point_distance > 0`; `min_num_points >= 2`;
    `max_num_points >= min_num_points`;
  - gate `voronoi`: `voronoi_control_points >= 3`; `voronoi_num_sites >=
    voronoi_control_points`; `voronoi_radial_variation >= 0`; `voronoi_angular_jitter >=
    0`; `voronoi_site_layout` in the same set;
  - gate `checkpoint`: `checkpoint_count >= 3`; `checkpoint_radius_min_frac` in `[0, 1)`.

  What follows: an invalid knob of a generator that is not selected is now accepted and
  ignored (`TrackGenConfig(generator="bezier", voronoi_control_points=3)` used to raise);
  an unknown generator name is reported before any parameter error; so are the facades'
  own checks: `TrackGenerator` and `GateGenerator` check `rng` (`None`, a wrong seed
  count) first, and `Course` refuses a capture and allocates its rng first, so a `Course`
  built with a bad generator knob inside a capture raises `CaptureError` instead of
  `ValueError`; when several rules
  fail, the first one reported can differ, since the rules run in the selected
  generator's `validate()` order (for `repulsive`, `repulsive_stages[-1] == num_points`
  now runs after the `settle_iters`, `resample_every` and `stall_window` rules); direct
  callers of the private helpers (`spec.alloc_scratch(cfg)`, `_inflate_warp_alloc`,
  `_run_pipeline`) no longer get these checks at all; and `viz/param_explorer.py`
  reports these errors from `generate_batch` instead of `build_config` (its handler
  already catches `ValueError`). `TrackGenConfig.__post_init__` no longer names any
  generator. The core rules (`num_envs`, `half_width`, relaxation, smoothing,
  `output_mode`, spacing, Z, and the gate fields) still run at config construction.
- **New errors when a generator is constructed:** an unknown key in a `generator_params`
  mapping (`ValueError` listing the valid fields); a non-empty `generator_params` for a
  generator without a Params class (`ValueError`, "takes no generator_params"); a Params
  instance of another generator's class, for example after `dataclasses.replace(cfg,
  generator=...)` without resetting `generator_params` (`TypeError` naming the expected
  class); a flat field and `generator_params` setting the selected built-in's knob to
  different values (`ValueError` naming the fields); a value outside a field's
  `lo` / `hi` or `choices` (`ValueError`). At config construction, a `generator_params`
  that is neither `None`, a mapping nor a dataclass instance raises `TypeError`.
- **The gate-count check sees `generator_params`.** `GateGenerator` checks
  `supported_orderings` and `min_gates <= max_gates(config) <= config.max_gates` against
  the resolved config, so `GateGenConfig(generator="polar",
  generator_params={"polar_num_knots": 9})` makes 9 gates. Messages are unchanged.
- `TrackGenerator.config` and `GateGenerator.config` return the config as passed, with a
  mapping `generator_params` normalised to its Params instance: `TrackGenerator(gen.config,
  rng)` builds the same generator. The pipeline reads a private copy with the resolved
  built-in Params written onto the flat fields; with `generator_params=None` the two are
  equal, as before.
- `SpecInfo` gains five trailing fields with defaults: `params_schema` (`{}`) and
  `writes` (`{}`), compared by `==` and ignored by `hash()`, and `capture` (`None`),
  `output` (`None`) and `provides` (empty). `GeneratorSpec` and `GateGeneratorSpec` gain a last field,
  `params` (default `None`), so positional construction is unchanged.
- `registry.add` now checks a Stage class's declaration and a spec's `params=` class
  before registering it; a plugin that fails a check is quarantined under `warn` (raises
  under `strict`).
- Code that walks the registry should read capture support from
  `registry.describe(kind, name).capturable`: a Stage class has no `capturable`,
  `alloc_scratch` or `generate` attribute. This includes the deprecated
  `generator_registry.get(name)` and `generator_registry.GENERATORS`, whose values may
  now be Stage classes. `warp_pipeline._run_pipeline` and
  `_inflate_warp_alloc` raise `TypeError` for a Stage class ("Stage generators run
  through TrackGenerator"), and `_inflate_warp_alloc` takes a private `seed=` keyword.
- `TrackGenerator` allocates its effective-seed buffer before the pipeline buffers, so a
  Stage can bind it as `StageBuilder.seed`. Built-in and legacy generators add no launch
  and their outputs are byte-identical (`tests/test_generator_fingerprints.py` passes
  unchanged).

### Deprecated

- `track_gen._src.warp_pipeline.GenScratch`: reading it, or
  `from track_gen._src.warp_pipeline import GenScratch`, emits a `DeprecationWarning` and
  returns `track_gen._src.warp_generate.GenScratch`. The old name will be removed in 1.0.
- The private helpers plugins written before `track_gen.ops` call are deprecated in
  favour of the ops. They do not warn yet and keep working through 0.x:
  `tests/test_legacy_plugin_abi.py` pins their names and parameter lists. They are
  `warp_pipeline._init`, `_sync`, `_fill_i32_k`, `_ccw`, `_arc_resample_inplace`,
  `_arc_resample_selected_inplace`, `arc_length_resample_inplace`,
  `arc_length_resample_selected_inplace`, `self_intersections_inplace`,
  `resample_uniform`, `resample_constant_spacing` and `turning_number_inplace`;
  `warp_generate.corner_count_sample_inplace`, `corner_sample_inplace` and
  `ccw_sort_inplace`; and `warp_generate_hull.point_count_sample_inplace` and
  `point_sample_inplace`. The generator guide maps each one to its op.
- `TrackGenConfig` fields that `TrackGenerator` never reads (roadmap PR8): `energy_steps`,
  `energy_lr`, `energy_w_sep`, `energy_w_len`, `energy_w_bend`, `energy_w_anchor`,
  `tp_iters`, `tp_tau`, `tp_alpha`, `tp_beta`, `smooth_finish`, `smooth_finish_iters`,
  `smooth_finish_tau`, `relax_chunk_size`, `relax_use_warp`, `max_regen_iters`,
  `min_angle`, `num_harmonics`, `decay_p`, `amplitude` and `num_centerline_samples`.
  Setting any of them to a value other than its default (the config class's own, for a
  subclass) emits one `DeprecationWarning` at config construction naming every such
  field, attributed to the code constructing the config (also through a subclass's
  `__post_init__`). Only the torch oracle, a benchmark or
  `track_gen._experimental.fourier` read them, and setting any of them leaves every
  built-in's output byte-identical. They will be removed in 1.0; `TrackGenerator` still
  raises for `smooth_finish=True`, after the warning. `output_mode` is deprecated in the
  docs only: it already rejects every value but its default, so it never warns. The
  inert `GateGenConfig` fields (`num_points_per_segment`, `rad`, `edgy`,
  `handle_clamp_frac`, `hull_displacement`) are not deprecated yet.
- Subclassing `TrackGenConfig` to add a plugin generator's knobs still works but is
  superseded by a Params class and `generator_params`. A plugin that moves its knobs to
  Params can override `Params.from_legacy` to keep reading its users' subclass fields.

### Deferred to PR16

Merges and fixes that would change built-in outputs or semantics, left out of PR7 so the
outputs stay byte-identical:

- **Bezier's polygon fallback stays its own kernel.** Its Bernstein form
  `(b0 + b1) * c0 + (b2 + b3) * c1` (with the pruned-corner NaN) rounds differently in
  float32 from the lerp of `ops.closed_polyline`.
- **Polar's dense spline stays its own kernel** (`_polar_spline_dense_k`, one thread per
  env looping over the segments). `ops.closed_catmull_rom` (one thread per sample) gives
  the same values on CPU, but on CUDA the loop structure decides which multiply-adds the
  compiler contracts into FMAs (loop-invariant terms are hoisted out of the inner loop),
  so the merge waits for a CUDA comparison.
- **Hashed RNG keying.** An `ops.rng.init(seed, salt, offset)` keyed as
  `rand_init(hash32(seed ^ hash32(salt)), offset)` would replace `seed * salt + offset`.
  At seed 0 every linear key collapses to `rand_init(offset)`, so `bezier`'s count and
  corner streams are then the same stream (both `rand_init(attempt)`).
- **Salts reused across generators**: 7919 (`hull` points, `polar` controls, the gate
  ordering's additive salt) and 2741 (`bezier` style, `checkpoint`'s candidate stride).
  The fix is a per-program salt registry that rejects collisions.
- **One endpoint convention for the curves.** `hull` samples each segment
  endpoint-inclusive (`u = s / (n - 1)`), `polar` and `voronoi` exclusive (`u = s / n`);
  the curve ops keep both behind `endpoint=`.
- **`voronoi` normalises its extent twice**: the envs that did not cross are normalised
  again after the selected fallback resample.
- **A NaN- and length-aware `resample_constant_spacing` with a per-env `spacing`
  array** (in the roadmap's PR7 scope). It changes the result on partially-NaN rows, so it
  lands with the resample stage.
- **`ccw_sort_inplace` treats `count < 1` as one point.** The legacy helper and the
  `bezier` / `hull` built-ins keep it (they run the `angle_sort` kernel with a count
  floor of 1); the public `angle_sort` treats `count <= 0` as an empty row.
- **One crossing predicate.** `self_intersections` (scale-relative tolerance), the
  `checkpoint` clip (strict sign test on `ccw`) and gate validity (`segments_cross`)
  differ on purpose today.
- Not done because no output would change and nothing needs them yet: merging the
  selected and unselected arc-resample kernels (it needs a dummy `active` array), reusing
  the angle sort's centroid in `hull`'s midpoint displacement, and a `funcs.cubic_bezier`
  for `bezier`'s assembly (roadmap PR8).

### Deferred to PR10 to PR13

PR8 changes nothing outside the generator stage. Left for later roadmap steps:

- **Gate Stages** (`output="anchors"`) and the stations adapter (PR11). Gate generators
  get static Params, `generator_params` and schemas now; PR11 accepts `output="anchors"`
  and reuses the gate Params classes, so `GateGenConfig.generator_params` keeps its
  meaning.
- **The rest of the `StageBuilder`** (`read`, `temp_like`, `carried`, `fail_bit`,
  `state`) and the planner that calls `Stage.build` natively (PR10); `ctx.bounded_loop`
  for `BOUNDED` and host buffers for `HOST` (PR12); `write(interp=, export=)`, more
  `raw.*` channels and `provides` (PR13). `state` waits for PR10 because device state
  mutated on every generation would break masked regeneration until the planner has
  mask plumbing.
- **Dynamic built-in knobs.** Every built-in field stays static (their kernels take host
  scalars, and no kernel arithmetic changes). The relax, inflate, validity and Z stages
  keep their flat fields and get no Params.
- `Course`-level `params` / `inputs` members (use `course.generator.params` and
  `course.generator.inputs`), and `randomize.uniform` writing the dynamic arrays (PR13).

### Testing

- CPU tests install a fake distribution (a monkeypatched
  `importlib.metadata.entry_points`) that registers a toy `toy.circle` track generator
  and a toy gate generator: the facades run them on CPU, `available()` lists them and
  `describe()` reports the distribution. One test installs a real `dist-info` (with
  `entry_points.txt`) on `sys.path` and loads it through the real
  `importlib.metadata`. They also cover duplicates and `replace=True`,
  a plugin colliding with a built-in (quarantined under `warn`, raising under
  `strict`), a broken plugin (quarantined, named in the unknown-name error, re-raised
  when requested by name), `TRACK_GEN_PLUGINS=strict` and `off`, an API-version
  mismatch (including `track_gen_api` declared in the entry point's package and an
  unrelated non-int `API_VERSION` that must be ignored), `scoped()` restoring state and
  the set of kinds, loading once from two threads on a fresh registry, and the old
  `generator_registry` / `gate_generator_registry` shims. Robustness cases: a
  distribution with unreadable metadata, two broken plugins sharing an entry-point
  name, namespace-package attribution, non-string names (`ValueError`), a plugin
  unregistering a built-in, re-adding the identical spec, an invalid
  `TRACK_GEN_PLUGINS` value, patching `importlib.metadata.entry_points`, and the
  pinned `TRACK_GEN_PLUGINS` / `track_gen.plugins` / `API_VERSION` literals.
- Tests that patched `GENERATORS`, `GATE_GENERATORS` or `_LOADED` now use
  `registry.scoped()`. `tests/test_gate_generator_registry.py` checks that the built-ins
  are among the available gate generators instead of pinning the exact list.
  `tests/test_public_api.py` also asserts that `import track_gen` loads no entry point.
- `tests/conftest.py` defaults `TRACK_GEN_PLUGINS` to `off` (unless it is already set),
  so the suite's registry-wide parametrizations see only the built-ins, whatever plugins
  the environment has installed; the plugin tests clear it.
- `tests/test_ops.py` (CPU) covers every public op: host-side validation (wrong type,
  dtype, shape or device, a context on another device, bad scalars), raised before any
  launch; hand-built semantics (NaN compaction and degenerate rows, `active` rows left
  bit for bit, count and NaN padding, known crossing counts, the sign of the turning
  number, the thickness of a circle, the normalised bounding box, sort order and count
  clamps, sampler ranges, determinism and salts, spline interpolation); **bitwise
  equivalence** with frozen copies of the kernels each op replaces
  (`tests/_legacy_kernels_pr7.py`) on random inputs with NaN rows, counts from 0 to past
  the stride and seeds that overflow `seed * salt`; and a tripwire that fails if an op
  allocates (with a positive control). It also checks the `__all__` list, the docstrings,
  the `ops.funcs` functions inside a user kernel, that `runtime._check_arr` is
  `check_array`, that `import track_gen.ops` loads no generator module and no torch, and
  that no ops module imports `track_gen._src` or `track_gen.exec` at module level. Also:
  a `count` above the stride behaves as `count == stride` in every count-aware op, and a
  subprocess runs those ops in Warp's debug mode (bounds-checked array access); `out`
  aliasing `src` or the scratch is rejected; device aliases are canonicalised;
  `grid_cells` and `fill` validate their arguments; and the generator modules' import
  rules (no `warp_pipeline`, no private `track_gen.ops` name beyond the legacy launchers,
  no gate module using a track generator's underscore name) are checked on the source.
- `tests/test_legacy_plugin_abi.py` (CPU) pins the names and parameter lists (name,
  kind, default) of the legacy helpers as recorded before the port, checks that
  `_pipe._init` / `_sync` are the runtime helpers, that `_pipe._fill_i32_k` keeps its
  arguments and launches, that `_pipe._ccw` works in a plugin kernel and that the
  historical kernel names still resolve, and runs every pinned helper on CPU with
  `ctx=None` inside `LaunchContext.scope()`, as a legacy plugin calls them. It also
  covers the `GenScratch` deprecation, `corner_sample_inplace` with a numpy per-env scale
  (any array Warp's launch accepts, as before), and the legacy private aliases.
- `tests/test_generator_fingerprints.py` compares sha256 fingerprints of every output
  field of every built-in track and gate generator (fallback-forcing configurations
  included) with `tests/goldens/pr7_fingerprints.json`, captured at 1077b18 with
  `tests/tools/capture_pr7_fingerprints.py`. The file is keyed by platform and Warp
  version (`darwin-arm64-warp1.18.0` so far) and the test skips on other keys.
- `tests/test_generator_docs.py` runs the generator guide's three examples (quick start,
  scatter gates, star generator) from the `.rst` source on CPU, and checks that every
  private name the `docs/generators/*.rst` pages cite still exists.
- `tests/test_no_host_sync.py` gains an `ops[all]` driver (every public op, with and
  without each optional argument) and a `wrappers[legacy_generator]` driver (the legacy
  samplers and sort, which the built-ins no longer call), and its driven-wrapper check
  now covers the ops modules. `tests/test_launch_stream.py` reuses the drivers, so every
  op is also checked for stream-explicit launches, and `test_ops_fill` covers `quatf`.
- `tests/test_generator_registry.py` reads `GenScratch` from `warp_generate`;
  `tests/test_generate_repulsive.py` forbids `runtime._sync` instead of the removed
  `rep._pipe._sync`.
- `tests/test_plugin_params.py` (CPU): `static` / `dynamic` metadata, `NoParams`, the
  dtype mapping, every registration rejection (non-frozen or undecorated class, a field
  without default or with a default out of bounds, a reserved name, a non-numeric
  dynamic field, a dynamic field on a `GeneratorSpec`), resolution of `None`, mapping
  and instance values (unknown keys, wrong class), a schema-shaped JSON mapping
  resolving to an equal, hashable instance (tuples, enums), inclusive `lo` / `hi` with
  NaN passing, `choices`, the value types of static `int` / `float` / `bool` fields (and
  the built-ins' leniency), the exact, `json.dumps`-able schema of the toy Params, and
  the private `StageParams` / `ParamTable` constructors.
- `tests/test_plugin_stage.py` (CPU): a toy Stage `toy.wobble` (one static and one
  dynamic field, a raw capacity of 128 points, a salted stream), with `HOST` and
  per-env-input variants, registered directly and through a fake distribution and run
  end to end; the declaration checks; `describe()`; `set` with a scalar, a per-env
  array and a direct write of the live array, each byte-equal to a generator built with
  that value; `set` rejections; the masked latch (envs left out stay bitwise
  unchanged); inputs; the salt formula and collisions; `StageBuilder` misuse;
  `gen.valid` defaulting to 1; the private pipeline helpers refusing a Stage class; a
  broken Stage plugin quarantined; a Params `validate` running only for the selected
  generator; `Course` passing `generator_params` through; `build`, `capacity` and
  `capture_class` refusing dynamic fields; unknown `capacity()` keys and a numpy
  capacity; and the rejection of other Stage kinds and of `BOUNDED`.
- `tests/test_generator_params.py` (CPU): for every built-in track and gate generator,
  `from_legacy` round-trips the flat fields and the flat form, a mapping and an
  instance give byte-identical outputs; the flat fields each built-in reads; conflicts;
  `generator_params` as the complete parameter set; switching generators; the
  unknown-name order; the `config` round trip; the gate count seeing
  `generator_params`; legacy specs with and without `params=`, and a duck-typed legacy
  spec with an unrelated `params` attribute; a config-subclass plugin, and a subclass
  default conflicting with a partial `generator_params`; and the built-in schemas.
- `tests/test_types.py`: the moved rules are one table, each row checked against the
  owning generator's construction with its old message, with config construction and a
  generator that does not own the knob accepting it; one `DeprecationWarning` per dead
  field (attributed to the caller, also through a subclass's `__post_init__`), silence
  at defaults (a subclass's own included), and the `generator_params` type check.
- `tests/test_params_cuda.py` (CUDA): the toy Stage recorded into a caller's graph
  equals an eager twin; a `set` between replays shows in the next replay with no
  re-recording; a `set` recorded inside the caller's graph reads its source array at
  replay; the self-graph reflects a `set`; a recorded masked generation after a write
  leaves the other envs bitwise unchanged; and a `HOST` stage raises
  `NotCapturableError` before any launch.
- `tests/test_plugin_registry.py` checks the new `track_gen.plugin` exports and the
  trailing `params` field of both spec types. `tests/test_generator_docs.py` also runs
  the guide's Stage quick start (registers, generates, `set("amp", 0.35)` changes the
  next generation) and compares its `describe()` schema with the JSON the guide prints,
  and checks that every built-in Params field is listed in its class's table in
  `docs/configuration/reference.rst`. `tests/test_param_explorer.py` and
  `tests/test_compare_generators.py` cover the schema-built controls and per-generator
  `generator_params`. `tests/test_warp_graph.py`, `tests/test_shape_variety.py` and
  `tests/test_caller_capture_cuda.py` read `describe().capturable` and pass
  per-generator overrides as `generator_params`.
- `tests/test_generator_fingerprints.py` passes unchanged: the flat configs reproduce the
  PR7 fingerprints of every built-in.
- `tests/test_conformance.py` runs the conformance kit (roadmap PR9). Every built-in
  track generator (`repulsive` on its small test config and 8 envs, `voronoi` also at
  64 sites, `hull` and `voronoi` also on the pinned configs their end-to-end tests used:
  `num_points=128, N_max=256`, voronoi with `half_width=0.1`) and every built-in gate
  generator (each in its default ordering, `bezier` and `hull` also in `random_pairs`)
  passes `check_generator` / `check_gate_generator` on 64 envs, on CPU in the fast lane
  and on `cuda:0` under the `cuda` marker, and keeps the built-ins' promises beyond the
  contract: every env generator-valid with a finite raw centerline, a finite result
  before `count[e]` and `count[e] >= 1` for every env, and a raw output that moves by
  more than `1e-3` between two seeds and between two envs. The plugin toys (`toy.circle`, `toy.lobed`,
  `toy.wobble` and its `HOST` and per-env-input variants, gate `toy.ring`) pass through
  the opt-in pytest plugin's `track_gen_conformance` fixture, the `HOST` variant with its
  waived checks reported as skipped. Fifteen deliberately broken toys
  (`tests/_broken_toys.py`, each `toy.circle` with one defect) fail exactly the check
  they break, with a message that locates the defect: a seed readback
  (`hot_path.host_sync`), a per-call allocation (`hot_path.allocation`), NaN in valid
  envs (`output.finite`), launch dims that follow a call counter (`hot_path.topology`),
  env `e` reading env `e + 1`'s seed (`batch_independence`), a legacy spec's `wp.launch`
  without `stream=` (`hot_path.stream`), a device counter (`statelessness`), ignored seeds
  (`diversity`), half a circle (`output.closed`), a module-level buffer and a buffer
  shared through a per-instance scratch object (`hot_path.buffers`), a last point never
  written (`output.finite`), `out_valid` never written (`output.shape`), a
  `wp.launch_tiled` without `stream=` and an `out.assign(staging)` (`hot_path.stream`);
  on CUDA, the readback and the allocation also fail `capture.caller`. Three edge toys
  that look unusual but conform pass every check: a loop closed by one of its two long
  straights, envs rejected with NaN and `out_valid = 0`, and a `wp.struct` scratch. The
  module also covers the kit's arguments, `skip=`, `modules=`, a failed `construct`, the
  `Report` API and table, the parametrize helpers, the origin of a `functools.partial`
  generate, every guard category in strict and record modes (a swallowed violation
  still fails, `allow=`, every allocation path and native object, views not
  allocating, module prefixes, every launch path: tiled, recorded, `assign`, bound at
  import, Warp's defining module, a Warp utility; `np.asarray`, other threads unseen,
  classes kept, Warp restored also after an exit in another context or under a foreign
  patch, the debug sync switched off), the launch trace (struct arguments flattened)
  and buffer ownership (registry specs not entered), the gate raw anchors, the capture
  helpers that run without a GPU, and that `import track_gen` does not import the kit
  and the kit does not import torch.
- **Per-generator contract tests folded into the kit.** The tests below duplicated a
  conformance check, which `tests/test_conformance.py` now runs for every generator;
  each generator module keeps its algorithm-specific tests (yield, crossing-free loops,
  shape variety, site counts, ...):
  - `test_polar_registered`, `test_checkpoint_registered`, `test_hull_is_registered`,
    `test_voronoi_is_registered`, `test_repulsive_is_registered`: registration and
    `capturable` are checked by `test_the_builtins_are_all_checked`,
    `test_builtins_are_capturable_and_stream_explicit` and `construct`;
  - `test_polar_e2e_centerline_finite_and_n_points`,
    `test_checkpoint_e2e_centerline_finite`, `test_checkpoint_e2e_n_points_per_env`:
    `output.shape` and `output.finite` (plus the built-ins' finite raw centerline);
  - `test_polar_e2e_closed_loop`: `output.closed`;
  - `test_checkpoint_e2e_valid_count` (now `test_checkpoint_e2e_reasonable_yield`, which
    keeps the yield assertion) and the count, finite-prefix and NaN-padding assertions
    of the hull and voronoi end-to-end tests: `output.count`, `output.finite` and
    `output.padding`, on the same 64 envs and, for hull and voronoi, the same pinned
    configs; their `count >= 1` for every env (invalid ones included) is
    `_assert_builtin_promises`' `output.count` minimum;
  - `test_polar_determinism_same_seed`, `test_checkpoint_determinism_same_seed`,
    `test_hull_deterministic` (CPU and CUDA), the determinism half of
    `test_voronoi_deterministic_and_cell_count_changes_tracks_cpu` (now
    `test_voronoi_cell_count_changes_tracks_cpu`) and
    `test_gate_generator_independent_instances_with_same_seed_are_deterministic`:
    `determinism`, on CPU and CUDA;
  - `test_polar_diversity_different_envs`, `test_checkpoint_different_seeds_differ`,
    `test_distinct_per_env_seeds_produce_diverse_gates`,
    `test_different_global_seeds_produce_different_gates`: `diversity`, whose
    `min_raw_delta` / `min_pair_delta` the built-in tests hold above `1e-3` (the
    removed tests' `not allclose`);
  - `test_repulsive_contract_through_tail` and `test_repulsive_determinism_and_diversity`
    (CPU and CUDA): `output.*`, `determinism`, `diversity`, `statelessness` and, on
    CUDA, `capture.self_graph`;
  - `test_repeated_generate_is_deterministic_for_a_fixed_rng` (`test_track_generator_facade.py`)
    and the repeated-call half of `test_hull_deterministic`: `statelessness`.
- The private tripwires now come from the kit: `tests/test_no_host_sync.py` and
  `tests/test_launch_stream.py` use `track_gen.testing.contract` (`no_host_sync` with
  `allow=` for the known-readback table, `stream_explicit`) instead of their own
  monkeypatched copies, and keep their drivers and the driven-wrapper check;
  `tests/test_caller_capture_cuda.py` records with `track_gen.testing.capture`
  (`side_stream`, `recording`, `poison_all`, ...); `tests/test_caller_capture_cpu.py`
  and `tests/test_ops.py` use `no_allocation`, which also sees `wp.array(data)` and
  every factory at Warp's allocation chokepoint.

## 0.2.0.dev1 (2026-10-22, tag `v0.2.0.dev1`)

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
