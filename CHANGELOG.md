# Changelog

## 0.2.0.dev3 (unreleased, `v2` branch)

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
