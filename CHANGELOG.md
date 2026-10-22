# Changelog

## 0.2.0.dev2 (unreleased, `v2` branch)

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
    name, origin module, distribution `(name, version)` of the registering entry point
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
