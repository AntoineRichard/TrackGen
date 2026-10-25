# PR16 design: the 1.0 cleanup (revision 2)

Roadmap PR16 (docs/library-readiness/target-architecture.md "### PR16", its "From PR13c /
PR13d" removals, D1, D7, D8, D12, G3, G11, G12, §12 "Results" / "Seeds", §13 "Legacy" /
"Mutation semantics"), the 0.2.0 CHANGELOG (Deprecated sections of dev0, dev2, dev4, dev5;
"Deferred to PR16"; the dev6 Deferred list), the migration guide's "Removed in 1.0" table
(docs/upgrading/from-0.1.rst:1900-1961) and docs/contributing/plugin-abis.rst "Status".
Base: worktree `/Users/antoiner/code/TrackGen/.claude/worktrees/v2`, branch `v2`, HEAD
`fe43a28` (release 0.2.0; PR #3 merges it into `main`). **No repository file was edited.**
Every experiment ran on CPU (Warp 1.18.0, Python 3.12, macOS arm64) from
`scratchpad/pr16-scratch/` against `git archive fe43a28` copies (`base/`), three patched
copies (`xfrozen/`, `xpoison/`, `xrng/`) and the worktree; evidence in §11.

**Revision 2** answers the 33-issue critic review (§12: 29 accepted, 3 of them with one
sub-point rejected; 4 accepted with a different fix; none rejected outright). The main changes: no internal
caller of a legacy helper may remain before the 1.0 warnings land (O9); the style fields
stay through 1.x instead of being tombstoned (U8); the dead fields are tombstoned at class
level, so subclasses, `replace` and attribute reads keep working (U9);
`GateGeneratorSpec` registrations are retired per name instead of breaking a plugin's
import (U7); the reproduction mode is `rng="0.2"`, kept for all of 1.x and silent (U2);
stored configs get a one-time notice (U3); configs keep `on_invalid="keep"` (U5); a 1.x
output-stability policy (U6); everything that never warned goes in 2.0 (U10); the legacy
keying runs the untouched 0.2 kernels (O4); salts no longer depend on stage ids (O2).

---

## 0.1 Sign-off

The racing repositories pin v0.2.0 while 1.0 lands. **USER** decisions are every
deliberate output change and every downstream-visible break or policy (recommendation
first; "deviates" marks a departure from the roadmap / spec text). **ORCHESTRATOR**
decisions are internal and reversible, with no downstream effect. Evidence ids (E*) refer
to §0.

### USER decisions

**A. Output changes and reproducibility**

- **U1. Hashed RNG keying for the built-ins: the 1.0 default for presets, `BatchSpec` and
  configs.** Every built-in seeded stream (15: 11 over the 6 generators,
  `order.random_pairs`, 3 elevation profiles; §2.3) is keyed `ops.rng.init(seed, salt)` /
  `init_at(seed, salt, i)` instead of `rand_init(seed * K + c)`, with
  `salt = hash32_host(fnv1a_32("<kind>.<name>/<stream>"))`: a fixed namespace, so renaming
  a stage never changes its draws (a second instance of the same built-in in one Program
  adds its stage id). **Every seeded track, gate sequence and elevation changes for a
  given seed**, CPU and CUDA, and so does everything `Course` derives from geometry. One
  shape changes too: a `Course` whose plan declares no checkpoint bound (an elevated
  `TrackGenConfig`, a variable width, a plugin geometry stage) sizes `max_checkpoints`
  from the construction batch, so the per-env stride of `course.checkpoints.*` can change
  (E5: one PR15 digest). from-0.2.rst gives the fix: pass `max_checkpoints` explicitly (its
  0.2 value is `course.checkpoints.position.shape[0] // num_envs`). Why: with the default
  seeds `0..E-1`, env 0 draws its `bezier` corner count, corner positions and first
  elevation knot from ONE stream (seed-0 collapse), and linear keys collide across envs
  (E3: 8 cross-env collisions in a 4096-env bezier+uniform batch, 35 in voronoi gates;
  hashed: 0 same-env, 0-1 cross-env, the birthday rate). Distributions: unchanged within
  noise on a prototype (E4); the real keying must pass the 16c distribution gate (O12).
  Unchanged: `randomize.uniform` (already hashed), plugin Stages (`b.salt`),
  `GeneratorSpec` plugins, `PerEnvSeededRNG`, `external`, the public `ops` samplers'
  default (U14). **Recommended: accept.** Alternatives: (R3) the config tier keeps the
  0.2 keying for all of 1.x (the NumPy `RandomState` model; configs get the fix only by
  opting in); landing the deferred declared `z_min` / `z_max` bounds in 1.0 so elevated
  configs size from the plan (rejected here: it adds clamping to two unbounded dynamic
  knobs, a second output change, and the readback is deterministic anyway).
- **U2. `rng="0.2"` reproduces 0.2.0: spelling, lifetime, warning.** *Deviates from the
  roadmap / spec ("`legacy_rng=True` kept for one minor release").*
  - Spelling: `rng: "0.2" | "1.0"` on `BatchSpec`, `presets.track/gates(rng=)`,
    `TrackGenConfig` / `GateGenConfig` (default `None`, U3), read-only `Program.rng` and
    `Course.rng`, written in /3 dicts. A versioned value scales to a later keying (a bool
    would need a second flag), and datasets log it next to the seeds and the Warp version.
    Alternative: the roadmap's `legacy_rng: bool` (no alias either way: the name is new in
    1.0).
  - Lifetime: **supported for all of 1.x, removed in 2.0.** It is the only way to
    regenerate a 0.2 seed-defined evaluation set, benchmark or dataset; it is a public
    keyword (semver); nothing ties the 1.0 -> 1.1 gap to calendar time; D8 rejected a
    one-minor-release compatibility window for the configs as "too aggressive for RL
    users", and the same reasoning applies here. Cost: the 0.2 kernels stay, which they
    do anyway for the frozen functions until 2.0 (U10, O4).
  - Warning: **none in 1.0** (a supported, frozen reproduction mode, so a pinned
    repository runs under `-W error::DeprecationWarning` and `filterwarnings = error`). It
    gains a `DeprecationWarning` in the 1.x release that announces 2.0, at least 6 months
    before 2.0.
  - Guarantee: with `rng="0.2"` and `on_invalid="keep"`, every built-in output equals
    0.2.0's for the same config and seeds on the same device architecture, CPU features
    and Warp version, with no exception (`style_sampling` included under U8, `edgy` under
    U4 C). Launch counts are not 0.2's (the U4 latch).
  - Alternative: the roadmap's removal in 1.1 (then 1.1 refuses `rng="0.2"` and /2 dicts,
    U3 c).
- **U3. Stored specs and stored configs.**
  (a) `PipelineSpec.to_dict()` writes `track_gen.pipeline/3` (adds `batch.rng`); 0.2
  refuses a /3 dict loudly ("unsupported format"). A /2 dict (everything 0.2 saved)
  **loads with `rng="0.2"`** and its stored `on_invalid`, with one `DeprecationWarning`
  ("re-save under 1.x after re-baselining"), so a logged spec reproduces its 0.2 bytes. /1
  dicts (warned since 0.2.0.dev4) are refused.
  (b) Config objects and config dicts (asdict / Hydra / YAML dumps, no format version)
  cannot be told apart from 1.x ones: their `rng` defaults to `None`, which means `"1.0"`
  **plus one `FutureWarning` per process** at the first Program built from such a config:
  "this config uses the 1.0 RNG keying: tracks differ from 0.2.0 for the same seeds; set
  rng='1.0' to accept (silences this) or rng='0.2' to reproduce 0.2.0". Specs the library
  writes from a config carry the resolved value.
  (c) 2.0, when `"0.2"` goes: a /2 dict, a /3 dict with `rng: "0.2"` and a config with
  `rng="0.2"` raise `ValueError` ("needs track_gen<2 to reproduce; re-save with rng='1.0'
  after re-baselining"); never a silent fallback to the new keying.
  **Recommended: as written.** Alternatives: /2 loads with the new keying (a silent change
  for every stored spec); no config notice (docs only); the notice also on presets /
  `BatchSpec` built in code (more noise for new users of the 1.x API).
- **U4. `edgy`: (C) keep it static and add a dynamic `tangent_blend`.** The roadmap's (A)
  reads `edgy` per env and computes `p = atan(edgy)/pi + 0.5` in the kernel; that is not
  the 0.2 host float64 transform: in float64 math it changes `p` for 112 of 1001
  two-digit decimals in [-5, 5] (91 / 2001 three-digit decimals in [0, 2]; float32 math:
  473 / 383), exact at the default 0 (E6). (B) = (A) plus the 0.2 host path under
  `rng="0.2"`. **(C, recommended):** `edgy` stays static; a new dynamic `tangent_blend`
  (the blend weight `p` itself, float32 in [0, 1]) whose initial value is the 0.2 host
  value `f32(atan(edgy)/pi + 0.5)`: no output change for any config, per-env arrays equal
  scalar runs, curricula draw a bounded, linear knob. Costs and duties: a new Params
  field; the 0.2 CHANGELOG promised "`edgy` dynamic", so from-0.2.rst and bezier.rst give
  the mapping (an `edgy` range [a, b] is the `tangent_blend` range [f(a), f(b)],
  f(x) = atan(x)/pi + 0.5) and say that a uniform draw on `tangent_blend` is not a uniform
  draw on `edgy`; a `generator.edgy` randomizer raises with the converted range;
  `to_dict()` writes `tangent_blend: None` unless it was set, so editing `edgy` in a saved
  /3 YAML stays legal (both given and inconsistent: `ValueError`).
- **U5. `on_invalid="poison"`: the default for presets and `BatchSpec` only; configs keep
  `"keep"` for all of 1.x.** *Narrows the roadmap (D12).* `BatchSpec.on_invalid` and
  `presets.track/gates(on_invalid=)` default to `"poison"`. `TrackGenConfig` /
  `GateGenConfig` gain `on_invalid` defaulting to `"keep"` (0.2's docs promise "a config
  always uses `keep`", results.rst:721; `Course(CourseConfig)` is the racing repositories'
  path); poison is opt-in there, and `legacy.to_pipeline_spec` passes the field instead of
  inheriting the `BatchSpec` default. What changes for preset users (E2, presets only):
  the `Track` / `GateSequence` rows of invalid envs (count 0, NaN geometry, `length` /
  `winding` 0; `valid`, `fail_mask`, `truncated` unchanged) and, in gates mode,
  `course.checkpoints` (a zero-copy view of the `GateSequence`); every `StepResult` entry
  and every owned component result is unchanged (5640 comparisons, 0 differences).
  from-0.2.rst names the downstream hazards: `count[e]` can be 0, so guard `% count[e]`
  (integer modulo by zero is undefined in Warp: 21 for `(7*3) % 0` on arm64 CPU, E13;
  typically a SIGFPE trap on x86-64; an unspecified index on CUDA); mask with a select, not `x * valid`
  (`NaN * 0 = NaN`); read `valid` before using a pose. Cost: +1 launch per unmasked preset
  generation (default track 156 -> 157, gates 16 -> 17); config launch counts unchanged.
  **Recommended: as written.** Alternatives: the roadmap's flip everywhere; a config
  sentinel that warns once when `on_invalid` is not set.
- **U6. A 1.x output-stability policy, and no other output change in 1.0.** *New.* For a
  fixed config or spec, seeds, device architecture, CPU features and Warp version, the
  built-ins' outputs are byte-stable across 1.x. An output-changing fix ships opt-in (a
  keyword whose default keeps the 1.0 output) and becomes the default only in 2.0. The
  `v1_outputs.json` goldens are re-baselined only in a major release (an opt-in adds
  entries, never edits one). So the PR7 merges deferred "to PR16" (bezier polygon
  fallback into `ops.closed_polyline`, polar's dense spline into `closed_catmull_rom`, one
  endpoint convention, voronoi's double normalisation, NaN-aware constant-spacing resample
  with per-env spacing, `ccw_sort` at `count < 1`, one crossing predicate) are 2.0
  candidates or 1.x opt-ins, not part of 1.0. Sub-decision: may a 1.x patch fix by default
  a defect where a VALID env violates the documented contract (a NaN, a crash), changing
  only those envs, with a release note? Recommended: yes. Alternatives: bundle the PR7
  merges into the 1.0 re-baseline (each needs a 0.2 branch for `rng="0.2"`); allow
  release-noted output changes in 1.x minors.
- **U17. Before 1.0: announce, offer forward-compatible pins, soak.** *New.*
  (a) A "Coming in 1.0" section in the 0.2 docs published from `main` (a docs-only PR
  after #3 merges) and in from-0.1.rst's 1.0 table: new tracks for the same seeds and the
  `rng="0.2"` pin, poison for presets and the `on_invalid="keep"` pin, the removals. Today
  no user-facing 0.2 page announces either default change (from-0.1.rst "Removed in 1.0,
  and what did not change" is silent; writing-a-generator.rst:1720 says "planned for a
  later release").
  (b) Optional `0.2.1` (branch from `v0.2.0`): accepts `rng="0.2"` (a no-op; `"1.0"`
  raises "needs track_gen>=1.0") on `BatchSpec`, presets and configs, and `on_invalid=` on
  the configs, so a repository writes its pins once on 0.2.1 and keeps identical outputs
  across the upgrade; no new warning.
  (c) Soak: 1.0.0 no earlier than N weeks after `v0.2.0` (suggested 6: 0.2.0's last
  deprecations, `style_sampling` in dev5, shipped the same day as 0.2.0) and only after
  the rc gate (O1) passes. **Recommended: (a), (b), (c) with N = 6.**

**B. Removals and breaks**

- **U7. Removed in 1.0: what warned in 0.2.0** (§1.1; E1: each warning fires on its
  documented usage, attributed to user code): `track_gen.set_capturing` /
  `track_gen.course.set_capturing` / `Course.set_capturing`; `CourseLine.track`;
  `registry.get("generator", g).params` (5 built-ins); **running** a `GateGeneratorSpec`
  (the adapter stage, `check_gate_generator` support, the `presets.gates` legacy branch);
  the `warp_gate` helpers `alloc_order_scratch` / `order_points` / `finish_ordered_gates`
  / `_GateStaging`; `warp_pipeline.GenScratch`; loading `track_gen.pipeline/1` dicts; the
  21 dead `TrackGenConfig` fields leave the schema (tombstoned, U9); `style_sampling=True`
  at the 1.0 keying (U8). Registering a `GateGeneratorSpec` never warned (0.2 warned at
  bind), so the class stays importable and registering one **retires that name only**:
  `registry.add` and `_src.gate_generator_registry.register` record it with a
  `DeprecationWarning`, binding it raises `PluginLoadError` with the port recipe, and the
  plugin's other registrations survive (deleting the class would make the plugin's import
  fail and quarantine all of it). The class goes in 2.0. **Recommended: accept.**
- **U8. `style_sampling`: removed from the 1.0 keying; the four fields stay, deprecated,
  until 2.0.** 0.2 warned only when a bezier generator was BUILT with
  `style_sampling=True`. Never warned: constructing a config or `BezierParams`;
  `style_sampling=True` on the other five generators (inert, E10); a range with style off
  (inert); `style_sampling=False`, the spelling the 0.2 warning recommends; every asdict /
  `to_dict` round trip, which carries the four keys at their defaults (E10); the frozen
  `generate_centerline_warp`, which honours `style_sampling` silently (a legacy
  `GeneratorSpec` that wraps it under its own name draws styles in 0.2 with no warning).
  **Recommended (S):** `style_sampling`, `rad_range`, `scale_range`,
  `handle_clamp_frac_range` stay fields of `TrackGenConfig` and `BezierParams` (and of the
  spec dicts) through 1.x. A non-default value warns at construction (`DeprecationWarning`:
  "runs only with rng='0.2'; removed in 2.0", with the `randomize=` recipe). Building a
  bezier generator with `style_sampling=True` raises `ValueError` with the recipe, unless
  `rng="0.2"`: there the 0.2 path runs byte for byte (its kernels stay for the frozen
  function anyway, U10), so `rng="0.2"` reproduces 0.2 style runs and the PR7
  `bezier_style` golden case stays. The other built-ins ignore the fields as in 0.2; the
  frozen bezier function keeps honouring them. 2.0 removes the fields and tombstones
  their default values (as 1.0 does for the dead fields). Alternatives: (T) tombstone now
  (smaller schema, but `rng="0.2"` cannot reproduce style runs and a plugin wrapping the
  frozen bezier function loses its draws with only a 1.0 warning); (H) hard removal
  (`TypeError` for the recommended 0.2 spelling).
- **U9. Dead `TrackGenConfig` fields: out of the schema, tombstoned at class level.** 0.2
  warned only for NON-default values at construction; round trips with the defaults
  (`TrackGenConfig(**asdict(cfg))`, a subclass's `Sub(**asdict(sub))`: E10), attribute
  reads and positional calls never warned. Recommended: the 21 names and `output_mode`
  leave the dataclass (not in `fields`, `asdict`, `repr`, `replace`). A small metaclass on
  `TrackGenConfig`, inherited by every subclass, pops them from the constructor's keywords
  before the generated `__init__` runs, with one `DeprecationWarning` naming them
  ("removed in 1.0 and ignored; drop them from stored configs; a TypeError from 2.0"),
  attributed to the caller. `smooth_finish=True` still raises `ValueError` (0.2 refused
  it at every build), and so does an `output_mode` other than `"constant_spacing"`. A name
  that is a field of the subclass itself (the oracle's private config) is left alone.
  Read-only properties return the old defaults with a `DeprecationWarning` (`cfg.min_angle`,
  `getattr`-based loggers). `dataclasses.replace` / `copy.replace` pass fields only, so they
  are silent (the sentinel the first revision gave `output_mode` would have warned on
  every `replace`: E10). Prototype E12. Breaks that cannot be warned (from-0.2.rst breaking
  table, with grep patterns): code reading the keys from `vars(cfg)` / `asdict(cfg)`;
  8+ positional arguments (a `KW_ONLY` marker after `min_point_distance`, field 7, turns
  the silent shift into a `TypeError`; no positional use in the repository); a subclass
  that also uses another metaclass (combine it with ours). Alternative: hard removal.
- **U10. Not removed in 1.0: everything slated for 1.0 that never warned in 0.2.0** (§1.2).
  Each gets a `DeprecationWarning` in 1.0 (O9, O10). **Removal: 2.0 for all of it,** no
  earlier than 6 months after 1.0. *Deviates from the roadmap's 1.0 removal (impossible
  under the spec's rule) and from revision 1's "`_src` in 1.1".* Why 2.0: every item is
  listed with a lifetime in plugin-abis.rst's status table or the 0.2 docs (de facto
  plugin ABI that 0.1-era generators call), a minor-release removal could land weeks after
  the first warning, and a repository jumping from 0.2 to 1.1 would never see it. The list:
  `_CAPTURING` (3 modules; the guide's "warns: yes" is wrong), the `warp_pipeline` PEP 562
  re-exports, the `_src` plugin helpers and private aliases (§1.2: 60+ names), the
  built-ins' frozen module functions and scalar kernels (with the style kernels, U8), the
  `_src` registries, `GeneratorSpec(accepts_ctx=False)`, `warp_relax`'s ignored
  `capturing=`, `validity.gate_default`'s inert `gate_width` / `check_crossing`,
  `tools.Progress(checkpoints=)` (documented "an alias kept for 0.x"), the
  `GateGeneratorSpec` class (U7). Alternatives: `_src` names documented nowhere in 1.1,
  with the 6-month minimum stated in from-0.2.rst; optionally also deprecate the inert
  `GateGenConfig` fields (`num_points_per_segment`, `rad`, `edgy`, `handle_clamp_frac`,
  `hull_displacement`, "not deprecated yet" in 0.2) for 2.0.
- **U11. Configs: deprecate mutation in 1.x, freeze in 2.0 (not `frozen=True` now).** A
  `@dataclass(frozen=True)` `TrackGenConfig` / `GateGenConfig` (E7) makes every non-frozen
  dataclass subclass fail at import (`TypeError: cannot inherit non-frozen dataclass from
  a frozen one`: the documented 0.2 plugin pattern) and every post-construction assignment
  raise, with no 0.2 warning. Recommended (b): the U9 metaclass marks an instance built
  when its construction returns; assigning a declared dataclass field of a built instance
  warns in 1.x ("configs are immutable from 2.0: use dataclasses.replace(cfg, field=v);
  for half_width also pass spacing=None to derive spacing again") and raises
  `FrozenInstanceError` in 2.0; other attributes (a subclass's caches) are not guarded;
  `replace`, `deepcopy` and pickle are silent (E12). `replace` keeps 0.2's semantics: it
  does not re-derive `spacing` (doing so would change outputs of 0.2 code). Alternative
  (a): hard freeze in 1.0 per the roadmap.
- **U12. Required `Track` / `GateSequence` fields: not in 1.0.** The 0.2 docs state a
  partial hand-built `Track` is legal (results.rst:197, `tests/test_types.py`), custom
  producers build results by hand, and nothing warned. Recommended: optional in 1.x, the
  library guarantee documented. Alternative: a construction-time `DeprecationWarning` for a
  result missing `layout` / `generation` / `fail_mask`, required in 2.0.

**C. Scope and unchanged items**

- **U13. The CPU APIC self-graph opt-in moves past 1.0** (additive, any 1.x). E8: x3.4 at
  E=1, x1.02 to x1.05 at E >= 64, while an open APIC capture blocks every other capture in
  the process; caller capture on CPU already works. Alternative: `policy="graph"` opt-in
  in 16d.
- **U14. Public `ops` samplers keep the linear keying by default.** `ops.sample_counts` /
  `ops.sample_grid_points` (and the `*_per_env` helpers) gain keyword-only
  `keying="linear" | "hashed"` (default `"linear"`, so plugins built on them keep their
  outputs); new public `ops.rng` (`init`, `init_at`, frozen formulas, §2.2).
- **U15. `TrackGenConfig`, `GateGenConfig`, `Course(CourseConfig)`, the function-based
  `GeneratorSpec(accepts_ctx=True)` and config subclasses carrying plugin knobs stay
  supported and undeprecated in 1.x** (CHANGELOG dev6 Deferred: "deprecating
  `CourseConfig` (roadmap PR16's decision)").
- **U16. Unchanged by design (confirm):** conditional lowerings stay on by default in
  Warp-registered captures and opt-in under `torch.cuda.graph`
  (`registered(conditional=False)`: the torch caching-allocator caveat); the Warp floor
  stays `>=1.18,<2` (D1 supersedes the roadmap's 1.16).

### ORCHESTRATOR decisions

- O1. Five PRs on a `v1` integration branch (§7): 16a removals, tombstones and 1.0
  warnings (1.0.0.dev0), 16b RNG plumbing at zero output change (dev1), 16c the 1.0
  defaults and the single golden re-baseline (dev2), 16d configs and internal cleanups
  (dev3), 16e docs and release (1.0.0rc1). Each PR ships its own from-0.2.rst section and
  CHANGELOG section. 1.0.0 after the GB10 validation and the rc gate (§7).
- O2. Stream salts `<kind>.<name>/<stream>` (§2.3), independent of stage ids; the gate
  generators reuse their track generator's strings (a bezier gate sequence sits on the
  bezier corners of the same seed); `init_at` = Warp's own `rand_init(key, index)`.
- O3. A Program-wide salt registry (`PlanError` naming both streams on any collision
  across stages) and a frozen table test of every built-in salt (§2.6).
- O4. **The legacy keying launches the untouched 0.2 kernels.** Every RNG kernel the 0.2
  Stages launch keeps its source byte for byte; the hashed keying gets one new sibling
  kernel per RNG kernel that differs only in its state-init block; the Stage picks one at
  build (§2.5). Kernels shared with the frozen functions and the public ops
  (`_sample_sites_k`, `ops._sample_counts_k`, `_sample_grid_points*_k`) are not modified.
- O5. Goldens: the five 0.2 golden files stay byte for byte and are checked under
  `rng="0.2", on_invalid="keep"` (the PR7 `bezier_style` case included); one new family
  `tests/goldens/v1_outputs.json` captured in ONE commit for `darwin-arm64`,
  `linux-x86_64` (GitHub `workflow_dispatch` artifact with a self-check) and
  `linux-aarch64` (spark-home); `TRACKGEN_REQUIRE_GOLDENS=1` on the CI floor leg (§2.8).
- O6. The conformance kit pins `on_invalid="keep"` in every producer it builds
  (generator, stage, gate-generator checks; components already do) and `rng="1.0"`
  explicitly (E2b: under a poison default 4 kit tests change behaviour).
- O7. Internal: `ctx` required on private internal wrappers (not on public utilities, the
  RNG sampler, the legacy helpers or the frozen functions); library config snapshots
  written with `object.__setattr__` (5 sites, E7); `_compat`'s pre-1.18 import fallback
  removed; the three planner messages citing dev releases and the "removed in PR16" hint
  reworded; the two nitpick-allowlist entries resolved (`FrameContact` public through
  `tools`, `_SlotInfo` annotation dropped).
- O8. `tests/test_golden_migration.py` (the unkeyed `pre_vec3f.npz`) fails at fe43a28 on
  darwin-arm64 (1 ulp, pre-existing): re-key it by platform and run it under `rng="0.2"`,
  or retire it (superseded by the PR7 fingerprints).
- O9. **No internal caller of a deprecated name** (first commits of 16a, before any 1.0
  warning). Live stages call legacy helpers today: `stages/track.py:256`
  (`_wpl.resample_constant_spacing`, every default track), `:367`
  (`_wpl.resample_uniform`), `:542` (the frozen `_zprof.apply_z_profile`, the flat
  elevation of every default gate pipeline), `:96` (`_wpl._fill_f32_k`), and the test
  reference composition `inflate_warp` / `_run_pipeline` (`warp_pipeline.py:580, 1585,
  1600, 1637, 1641`). A simulated decorator turned two default track generations and one
  default gate generation into 8 `DeprecationWarning`s about names the user never called
  (E10). Each internal call moves to the op or to a private, byte-identical
  implementation (`_resample_constant_spacing`, `_resample_uniform`, a private
  `_apply_z_profile` path for the flat gate profile, `ops.fill`); the frozen functions call
  private implementations too, so a frozen-function call warns once. The public legacy
  names become thin warning shims over the private ones. Test (16a): build and generate
  every built-in track and gate pipeline, `Course` (both config and preset routes, every
  component), the kit checks, under `warnings.simplefilter("error", DeprecationWarning)`;
  a call-graph probe asserts that no frozen function or legacy helper is reachable from
  any Program.
- O10. **Warning mechanics for the §1.2 `_src` names**: the names leave their module's
  dict and a module `__getattr__` returns them with a warning, for functions, kernels and
  `wp.func`s alike (a decorator cannot wrap a `wp.Kernel` / `wp.Function` without
  breaking its use inside user kernels). The stack walk skips `track_gen.*` and `warp.*`
  frames: a call or import warns at the user's line; a `wp.func` or kernel referenced
  inside a user kernel warns once per name at the user's `@wp.kernel` line, visible from
  `__main__`, also with a warm kernel cache (E11; without the `warp.*` skip it was
  attributed to `warp/_src/codegen.py`, hidden). Once per name.
- O11. **In-repo users and docs promises** (16a): the oracle (`tests/_oracle`),
  `tests/_warp_compare.py` (`min_angle` in `_corner_angles_gate_k`),
  `benchmarks/benchmark_relaxation.py`, `benchmark_yield_sweep.py`,
  `benchmark_pipeline.py` (`max_regen_iters`), `viz/plot_ablations.py` (`min_angle`
  sweep, Fourier fields), `viz/make_report.py`, `_experimental/fourier.py` move to private
  config subclasses (or the `min_angle` ablation figure is deleted); the three executed
  migration snippets that exercise removed surface (from-0.1.rst `set-capturing` :1197,
  `style` :1220, `plugins` :1406) switch to `expect-error=` or become non-executed 0.2
  examples; api.rst's policy is rewritten (§8); every "0.x" / "1.0" lifetime promise is
  classified (§1.4).
- O12. **Distribution gate on the real keying** (16c, CPU): E >= 4096 per generator over
  two seed blocks, every track and gate generator (repulsive included), elevation
  `uniform` / `random_walk` / `noise` with nonzero `z_max_step` / `z_noise_amplitude`,
  gates with `order.random_pairs`; invalid fraction (binomial), corner / gate counts,
  length, curvature and z statistics (KS), hashed vs 0.2 keying. It replaces C4's yield
  clause (the 13-env snapshot is too small).

---

## 0. Evidence at a glance

| Id | Script / file (`pr16-scratch/`) | Question | Result |
|---|---|---|---|
| E1 | `probe_removals.py` -> `audit.txt`, `audit.json` | Does each slated item warn on its documented usage? | 44 cases: 23 warn (22 attributed to the caller; the `dataclasses.replace` route to `dataclasses.py`, hidden), 21 silent (§1) |
| E2 | `exp_poison_course.py` -> `exp_poison_course.out` | Course outputs keep vs poison (presets only) | 0 of 5640 owned arrays differ; gates `course.checkpoints` (aliased) differ; Track/GateSequence rows of invalid envs differ |
| E2b | `lane_xpoison.txt` | What a poison default breaks in the suite | 351 failed. The patch flipped `BatchSpec`'s default, which `legacy.to_pipeline_spec` inherits, so configs were flipped too: U5's set is a subset |
| E3 | `exp_salts.py` -> `exp_salts.out` | Legacy vs hashed collisions | legacy: seed-0 collapse, 8 / 1 / 1 / 35 cross-env collisions (bezier / hull / checkpoint / voronoi gates, E=4096); hashed: 0 same-env, 0-1 cross-env |
| E4 | `exp_yield.py` -> `yield_*.json` | Distribution under a hashed seed (prototype `xrng/`) | invalid counts / lengths within noise for 10 generators; does NOT cover `random_walk` / `noise` (std 0: `z_max_step`, `z_noise_amplitude` were 0), repulsive, `random_pairs` (O12) |
| E5 | `lane_xrng.txt` | Which tests pin RNG bytes | 27 failed: goldens (PR7, PR11 ref, PR13a, PR14, PR15 x11, pre_vec3f), 4 seed-picked tests, 1 doc snippet, 6 patch artefacts |
| E6 | `exp_edgy.py` -> `exp_edgy.out` | edgy in-kernel transform vs 0.2 | f64: 1677/45000, 112/1001, 91/2001 differ; f32: 8633, 473, 383; default 0 exact |
| E7 | `lane_xfrozen.txt`, `lane_xfrozen2.txt` | `frozen=True` configs | 1863 failed as is; 18 after fixing 5 internal sites (mutation, non-frozen subclasses) |
| E8 | `exp_cpu_apic.py` -> `exp_cpu_apic.out` | CPU eager vs APIC replay | x3.42 (E=1), x1.02-1.05 (E >= 64); bytes equal |
| E9 | `lane_base.txt` | Baseline CPU lane | 3413 passed, 1 failed (`test_golden_migration`) |
| E10 | `critic-deprec/` (re-run by the architect) | Unwarned routes | `p_decorator_sim.py`: 8 warnings from default generation; `p_style_nonbezier.py`: `style_sampling=True` + ranges on polar / hull / voronoi / checkpoint / repulsive build and generate, 0 warnings; `p_bezparams.py`: `BezierParams(style_sampling=False, rad_range=None)`, `generator_params={"style_sampling": False, "scale_range": ...}` silent, `to_dict()` and `legacy.to_pipeline_spec(...).to_dict()` carry the 4 keys; `p_roundtrip.py`: `MyCfg(**asdict(MyCfg()))` silent, `dataclasses.replace` passes 109 kwargs incl. `output_mode`; `cases/style_frozen_function.py`: frozen bezier honours style, 0 warnings |
| E11 | `arch-r2/funcwarn2/` (and `critic-deprec/funcwarn/`) | Warning a `wp.func` used inside a user kernel | without a `warp.*` skip: 4 warnings at `warp/_src/codegen.py:6518/6519` (hidden); with it and once-per-name: 1 warning at the user's `@wp.kernel` line, shown from `__main__`, also with a warm cache |
| E12 | `arch-r2/meta_proto.py` | Class-level tombstone + mutation guard | base and subclass 0.2 dicts: 1 warning at the caller; `replace` / `deepcopy` / pickle silent; `asdict` clean; reads warn; field assignment warns, private attribute silent; a subclass field of the same name kept; `smooth_finish=True` raises; `KW_ONLY` |
| E13 | `critic_downstream/modzero.py` | `% count` at count 0 in a Warp CPU kernel | `(7*3) % 0` returns 21 (undefined behaviour) |

---

## 1. Removals audit

### 1.1 Removed in 1.0 (each warned in 0.2.0)

"Fires" = E1 on the documented usage, attributed to the calling line (visible under
default filters from `__main__`, hidden in a library module, as any `DeprecationWarning`).

| Item | 0.2.0 warning | Fires on documented usage | 1.0 plan (16a unless noted) |
|---|---|---|---|
| `track_gen.set_capturing`, `track_gen.course.set_capturing`, `Course.set_capturing` | `_src/runtime.py:132` (alias `_src/course.py:1761`), "... has no effect ... removed in 1.0" | yes, all three | delete the function, alias and exports (`track_gen/__init__.py:25,52`, `track_gen/course.py:13,15`) |
| `CourseLine.track` | `_src/course_line.py:185` | yes | delete the property |
| `registry.get("generator", g).params` (5 built-ins) | `_src/builtin_stage.py:89` (`DeprecatedParamsAlias`) | yes, all five | delete the descriptor and the 5 attributes |
| `style_sampling=True` on a BUILT bezier generator | `_src/warp_generate.py:890`, once per build (+ the compose `UserWarning`, `builtin_params.py:393`) | yes when built (TrackGenerator, `Course(CourseConfig)`, presets `generator_params`, `from_dict` + build); **no** for every other route (U8, E10) | `ValueError` with the recipe at `rng="1.0"` (code in 16b, the default from 16c); runs as 0.2 under `rng="0.2"`; fields kept and warned (U8) |
| running a `GateGeneratorSpec` (adapter stage, `check_gate_generator` support, `presets.gates` legacy branch) | `_src/generator_binding.py:96`, once per (registry, name), at bind | yes when BOUND; registration silent | delete the adapter (`stages/gates.py` `_LegacyGateSlot`) and binding branch; the class stays, a registration retires its name (U7) |
| `warp_gate.alloc_order_scratch`, `order_points`, `finish_ordered_gates` | `_src/warp_gate.py:865` | yes | delete |
| `warp_gate._GateStaging` | `_src/warp_gate.py:973` | yes | delete the `__getattr__` branch |
| `warp_pipeline.GenScratch` | `_src/warp_pipeline.py:1699` | yes (attribute and `from ... import`) | delete that branch (the hook stays, §1.2) |
| `track_gen.pipeline/1` dicts | `pipeline/spec.py:575` | yes | delete `_migrate_v1`; `/1` -> `ValueError` ("load and re-save with 0.2") |
| 21 dead `TrackGenConfig` fields | `_src/types.py:89`, non-default values only | yes for a non-default value; **no** for defaults (round trips), reads, positional calls; `replace` attributed to `dataclasses.py` | leave the schema, tombstoned (U9, §1.3) |

### 1.2 Not removed in 1.0 (no 0.2.0 warning): a 1.0 warning, removal in 2.0 (U10)

| Item | 0.2.0 state | 1.0 warning site |
|---|---|---|
| `runtime._CAPTURING`, `warp_pipeline._CAPTURING`, `warp_gate._CAPTURING` | plain constants, silent; the guide's "warns: yes" is wrong (corrected) | module `__getattr__` (O10) |
| `warp_pipeline.<warp_generate name>` (PEP 562 re-export, 48 names) | silent (`warp_pipeline.py:1695`) | the same hook |
| `_src` plugin helpers and private aliases, the full set `tests/test_legacy_plugin_abi.py` pins: **`warp_pipeline`** functions `_init`, `_sync`, `_arc_resample_inplace`, `_arc_resample_selected_inplace`, `self_intersections_inplace`, `arc_length_resample_inplace`, `arc_length_resample_selected_inplace`, `resample_uniform`, `resample_constant_spacing`, `turning_number_inplace`; funcs `_ccw`, `_nan0`, `_self_intersections_func`, `_turning_func`, `_thickness_func`, `_safe_normalize2`; kernels `_self_intersections_by_i_k`, `_thickness_k`, `_turning_k`, `_resample_scan_k`, `_resample_lookup_k`, `_cs_scan_k`, `_cs_lookup_k`, `_arc_scan_k`, `_arc_lookup_k`, `_arc_scan_selected_k`, `_arc_lookup_selected_k`, `_fill_f32_k`, `_fill_i32_k`; **`warp_generate`** `corner_count_sample_inplace`, `corner_sample_inplace`, `ccw_sort_inplace`, `_safe_normalize2`; **`warp_generate_hull`** `point_count_sample_inplace`, `point_sample_inplace`; **`warp_gate`** `_init`, `_sync`, `_fill_i32_k`, `_fill_vec2_k`, `_fill_vec3_k`, `_fill_quat_k`, `_safe_normalize2`; `warp_generate_voronoi._voronoi_layout_mode`; `warp_generate_polar._polar_num_knots`, `_catmull_rom`, `_polar_spline_dense_k`; `warp_generate_repulsive._safe_dir` | "deprecated, they do not warn" (CHANGELOG dev2; plugin-abis.rst:332) | module `__getattr__`, stack walk skipping `track_gen.*` and `warp.*`, once per name (O10, E11); internal callers moved first (O9) |
| frozen module functions and their scalar kernels: `generate_centerline_warp` / `bezier_alloc_scratch` (with `style_sample_inplace`, `assemble_inplace(rad_per_env=, clamp_per_env=)`, `_style_sample_k`, `_assemble_style_k`, `GenScratch.gen_style_*`), `generate_hull_warp` / `hull_alloc_scratch`, polar, voronoi, checkpoint, `apply_z_profile`, `apply_z_profile_knots`, `sample_polar_controls`, `sample_checkpoint_rings`, `select_voronoi_anchors` | silent ("Warns in 0.2: no", from-0.1.rst); 36 test-side files use them as byte references | each function, once per name; byte-frozen to 2.0 (they keep the 0.2 keying and style path); the survivors (`*_per_env`, `sample_voronoi_sites`, which the Stage and the voronoi gates call) gain a private hashed launcher (O4) |
| `_src.generator_registry` / `gate_generator_registry` `register` (new name and overwrite), `get`, `available`, `GENERATORS`, `GATE_GENERATORS` | silent, except an overwrite (`plugin/_registry.py:683`); a 0.1 registration runs with no warning (guide `reg-01`) | module functions and the views' `__getattr__`; the gate `register` retires a `GateGeneratorSpec` name like `registry.add` (U7); `GeneratorSpec` moves to a private module, re-exported by `track_gen.plugin` |
| `GeneratorSpec(accepts_ctx=False)` (the 5-arg adapter), with `_CAPTURE_LOCK` and `LegacyPluginStreamError` | documented "Supported ... every 0.x release", no warning | at bind, once per (registry, name) |
| `ctx=None` defaults of the `_src` wrappers | the legacy helpers' convention | covered by the helper warnings; private internal wrappers get a required `ctx` (O7); the utilities' keyword `ctx=None` and `PerEnvSeededRNG` are public API and stay |
| `warp_relax.xpbd_solve_inplace` / `band_l0_inplace` `capturing=` | accepted, ignored, silent | when passed |
| `TrackGenConfig.output_mode` | rejects every value but its default, never warns | the U9 tombstone (warns when passed), the read property |
| `validity.gate_default`'s `gate_width` / `check_crossing` (inert since dev5) | silent; `presets.gates` writes `gate_width` into every gate spec | when set to a non-default value; 1.0 presets stop writing them; /2 dicts drop them inside the /2 migration's one warning |
| `tools.Progress(checkpoints=)` | "an alias kept for 0.x" (`tools/_adapters.py:517`, writing-a-component.rst:763), silent | when passed; docstring and docs reworded |
| the `GateGeneratorSpec` class and `registry.add` of one | registration silent | at registration (U7) |

Also not removed (not slated): `Course.generate(seeds=)`, `Course.from_config`, the
`CheckpointSampler` / `PropSampler` readback sizing ("not planned"), the inert
`GateGenConfig` fields ("not deprecated yet"; optional deprecation in U10),
`_CAPTURE_LOCK` (needed by the legacy ABI), the four style fields (U8).

### 1.3 The config metaclass (U9, U11) and the style fields (U8)

- `class _ConfigMeta(type)`, the metaclass of `TrackGenConfig` and `GateGenConfig`,
  inherited by every subclass. `__call__(cls, *args, **kwargs)`: pops the tombstoned names
  (21 dead fields, `output_mode`) that are not fields of `cls`; raises `ValueError` for
  `smooth_finish=True` or a non-default `output_mode`; warns once (stacklevel 2: the
  construction line); calls `super().__call__`; then marks the instance built
  (`object.__setattr__(inst, "_tg_built", True)`). `GateGenConfig` has no tombstoned
  names; it uses the metaclass for U11 only.
- Read-only properties for the 22 names, generated in a loop, returning the 0.2 defaults
  with a `DeprecationWarning`. The library never reads them (today only
  `legacy.py:138` reads `smooth_finish` and `types.py:858` `output_mode`; both move into
  the metaclass).
- `__setattr__` (16d): warns when `_tg_built` is set and the name is a declared dataclass
  field.
- `_warn_dead_fields` and its frame walk are deleted. A positional `KW_ONLY` marker
  follows `min_point_distance`.
- The style fields (U8) stay ordinary fields of `TrackGenConfig` and `BezierParams`; a
  non-default value warns in `__post_init__` (defaults are silent, so round trips and
  `replace` are silent); the bezier Stage's `build` raises at `rng="1.0"`. The /2
  migration keeps them.
- The same tombstone list is applied when a /2 dict's config-derived params are loaded
  and when `generator_params` mappings are coerced (they never contain the dead names
  today; the check guards hand-written mappings).

### 1.4 Lifetime promises in the code and docs (O11)

`grep -rnE "in 1\.0|at 1\.0|until 1\.0|through 0\.x|for 0\.x|every 0\.x|..."` at fe43a28:
20 lines in 11 library modules (`warp_gate.py` 7, `course_line.py` 3, `types.py` 2, ...)
and 34 lines in 11 docs pages outside `upgrading/` (plugin-abis.rst 13, configuration
reference.rst 4, reference/plugin.rst 3, api.rst 3, bezier.rst 3, gate-pipeline.rst 2,
writing-a-generator.rst 2, results.rst, gate-sequences.rst, pipeline-stages.rst,
writing-a-gate-generator.rst); plus 61 "0.x" mentions in 20 library modules (31 in
`_src/course.py`, naming the 0.x `StepResult` fields and entry points) and 33 in docs.
Each hit is classified in 16a: removed in 1.0 (text deleted with the item), kept to 2.0
("until 2.0" / "removed in 2.0": configuration reference.rst:1266, gate-pipeline.rst:705,
gate-sequences.rst:282 promise the inert gate fields go "in 1.0"; they cannot), or a
name for the legacy surface ("the 0.x fields" -> "the 0.2 fields").

---

## 2. The RNG switch

### 2.1 What 0.2 does

Every built-in stream is `state = wp.rand_init(seeds[e] * K + c)` in int32 wrap-around,
`seeds` being the effective seed (`_effective_seed_k`: the seed at epoch 0, a 31-bit hash
at epoch > 0). With the default seeds `0..E-1`, env 0 has every key equal to its offset
`c`, and every stream with `c = 0` coincides (E3: `bezier.corner_count`, `bezier.corners`
and `elevation.uniform[0]` share state `129708002` at seed 0). Linear keys also collide
across envs. The collapse vanishes at epoch > 0 (hashed seeds), so it hits the
construction batch and every env never advanced. `randomize.uniform` already uses the
hashed keying (`stages/randomize.py:69`).

### 2.2 `ops.rng` (new public module, frozen formulas)

```python
# track_gen/ops/rng.py   (@wp.func; host twins key_host / init_host for tests)
init(seed: int32, salt: uint32) -> uint32
    = wp.rand_init(int32(hash32(uint32(seed) ^ salt) & 0x7FFFFFFF))         # = b.salt docs
init_at(seed: int32, salt: uint32, index: int32) -> uint32
    = wp.rand_init(int32(hash32(uint32(seed) ^ salt) & 0x7FFFFFFF), index)  # Warp's 2-arg form
```

`init` is the keying the plugin guide has recommended since 0.2.0.dev2, so plugin
streams already reproduce bit for bit. Frozen for plugin API 1. `ops.sample_counts` /
`ops.sample_grid_points` and the `*_per_env` helpers gain keyword-only
`keying="linear" | "hashed"` (U14; hashed: `init_at(seed, salt, offset)`, through new
sibling kernels, O4).

### 2.3 Streams and salts

`salt = hash32_host(fnv1a_32("<kind>.<name>/<stream>"))` (computed with the library's
`_fnv1a_32` / `hash32_host`; all 15 distinct). A second instance of the same built-in in
one Program uses `"<kind>.<name>@<stage_id>/<stream>"`. The gate generators use their
track generator's strings (O2).

| Stage | 0.2 kernel (kept, `rng="0.2"`) | 0.2 key | 1.0 salt string -> value | 1.0 key |
|---|---|---|---|---|
| `bezier` | `ops._sample_counts_k` / count twin | `seed*6151 + 0` | `generator.bezier/corner_count` 0xb739affb | `init_at(seed, salt, 0)` |
| | `ops._sample_grid_points[_scaled]_k` | `seed*9781 + 0` | `generator.bezier/corners` 0xc2c74e45 | `init_at(.., 0)` |
| | `_style_sample_k` (U8: `rng="0.2"` only) | `seed*2741` | none (refused at the 1.0 keying) | n/a |
| `hull` | `ops._sample_counts_k` | `seed*5119 + 1` | `generator.hull/point_count` 0x55e558eb | `init_at(.., 0)` |
| | `ops._sample_grid_points_k` | `seed*7919 + 1` | `generator.hull/points` 0xc60040a4 | `init_at(.., 0)` |
| | `_midpoint_displace_dyn_k` | `seed*3083 + 1` | `generator.hull/displacement` 0x8c818de1 | `init` |
| `polar` | `_polar_controls_dyn_k` | `seed*7919 + 17` | `generator.polar/controls` 0x23dab0d9 | `init` |
| `voronoi` | `_sample_sites_k` (shared with the frozen function and the gates) | `seed*8171 + 19` | `generator.voronoi/sites` 0x877be6df | `init` |
| | `_select_anchor_sites_dyn_k` | `seed*3911 + 31` | `generator.voronoi/anchors` 0xc9f3730a | `init` |
| `checkpoint` | `_sample_checkpoints_dyn_k`, candidate k | `seed*4099 + k*2741 + 1` | `generator.checkpoint/rings` 0x7357f137 | `init_at(.., k)` |
| `repulsive` | `_sample_obstacles_k` | `seed*6247 + 23` | `generator.repulsive/obstacles` 0x15710913 | `init` |
| | `_seed_lfinal_k` | `seed*5209 + 41` | `generator.repulsive/growth` 0xcd2aa706 | `init` |
| gate `bezier` / `hull` / `polar` / `voronoi` / `checkpoint` | the same kernels | the track generator's keys | the track generator's strings | idem |
| `order.random_pairs` | `_order_random_pairs_k`, pair p | `seed*3187 + 7919 + p*104729` | `order.random_pairs/pairs` 0x977a7d68 | `init_at(.., p)` |
| `elevation.uniform` | `_z_uniform_dyn_k`, knot / gate i | `seed*15679 + i` | `elevation.uniform/z` 0x7d59d787 | `init_at(.., i)` |
| `elevation.random_walk` | `_z_walk_dyn_k` | `seed*15683` | `elevation.random_walk/z` 0x9717c554 | `init` |
| `elevation.noise` | `_z_noise_dyn_k`, harmonic k | `seed*15731 + k` | `elevation.noise/z` 0xd8aea614 | `init_at(.., k)` |
| `randomize.uniform` | `stages/randomize.py:69` | already hashed (`b.salt`, stage id) | unchanged | unchanged |
| `external`, `stations.*`, `width.*`, `relax.*`, `frames`, `offset`, `elevation.flat`, `lift.*`, `tangents`, `gate_relax`, `gate_arclen`, `gate_frames`, `normalize.extent`, `validity.*`, commit | no RNG | n/a | n/a | n/a |
| effective-seed prologue, `PerEnvSeededRNG` | `_effective_seed_k`; `rand_init(seed)` | unchanged | unchanged | unchanged |

The frozen module functions keep the 0.2 keys (frozen, §1.2).

### 2.4 `rng`

- Where: `BatchSpec.rng: str` (`"0.2"` default during 16b, `"1.0"` from 16c), serialized
  in /3; `presets.track/gates(rng=)`; `TrackGenConfig.rng` / `GateGenConfig.rng`
  (`None` default, U3 b; appended after `generator_params`, keyword-only);
  `legacy.to_pipeline_spec` resolves `None` and passes the value; `CourseConfig` through
  `gen`. Read by the built-in stages through a private `StageBuilder._rng`; plugin stages
  never see it. `Program.rng`, `Course.rng` read-only.
- No warning when `"0.2"` (U2). The config notice of U3 b is a `FutureWarning`, once per
  process.
- Guarantee: §0.1 U2. Scope: same device architecture, CPU features and Warp version
  (Warp's CPU backend compiles with `-march=native`, §2.8).

### 2.5 Kernel implementation (O4)

- The 0.2 kernels stay byte for byte (a test hashes each RNG kernel's Python source
  against a table frozen from `v0.2.0`), and `rng="0.2"` launches them: the same kernel
  objects, the same generated source per kernel, no new branch, so NVRTC / LLVM see 0.2's
  input for that kernel (the module gains siblings beside it).
- Each RNG kernel the Stages launch gets a sibling `*_h_k` that differs only in its
  state-init block (`state = rng.init(seeds[e], salt)` or `init_at(..., i)`, `salt:
  uint32` an argument); the float path after the state is the same text. The Stage
  chooses at build from `b._rng` (static, no launch difference). Shared kernels
  (`_sample_sites_k`, `ops._sample_counts_k`, `ops._sample_grid_points*_k`) are not
  modified; the Stage, the gate generators and `ops` `keying="hashed"` launch their
  siblings through private launchers.
- Proof on CPU: the source-hash table above, plus the extended CUDA-source lock
  (`test_difficulty_axes.py::test_the_cuda_source_of_each_twin_reduces_to_its_original`):
  each sibling's generated CUDA source with its init block dropped equals its 0.2
  kernel's, for every shared kernel too. Proof on GPU: §2.9 C1.
- Cost: about 17 more kernels (module compile time measured in 16b, CPU and GB10); at run
  time one `hash32` + one `pcg` per stream init.

### 2.6 Salt collision checks (O3)

1. Per stage (exists): two streams of one stage with equal salts raise in `b.salt`.
2. **Per Program (new):** the planner collects every stage's salts after `build` and
   raises `PlanError` naming both streams on a collision (built-ins, plugins and
   randomizers alike; a plugin stage whose id/stream string equals a built-in's collides
   by construction).
3. **Frozen salt table test:** every built-in (kind, name, stream), its string and value
   (§2.3), uniqueness, the gate-generator reuse; renaming a built-in stage's id does not
   change any output; a rename of a stream string is a deliberate output change.
4. **State-collision test (host only):** `hash32_host` + a host `pcg` twin, seeds
   `0..65535` and a hashed-epoch batch, every default pipeline: no env has two streams
   with one initial state; cross-env coincidences at most the birthday expectation.

### 2.7 Spec format /3 and `from_dict`

/3 = /2 + `batch.rng`; presets stop writing `validity.gate_default`'s inert keys.
`from_dict(/2)`: `rng="0.2"`, `on_invalid` as stored (`"keep"` if missing), the inert
`gate_width` / `check_crossing` of `validity.gate_default` dropped (inert: `gate_width`
0 or equal to `gate_frames`'), the style keys kept (U8), one `DeprecationWarning`.
`from_dict(/3)`: a missing `rng` reads `"1.0"`, a missing `on_invalid` `"poison"` (the
`BatchSpec` defaults; the library always writes both). `/1`: `ValueError`. 2.0: §0.1 U3 c.

### 2.8 Golden re-baseline (one commit, every platform key) (O5)

Today (fe43a28): `pr7_fingerprints.json`, `pr11_gate_stages.json`,
`pr13a_pr11_relaxed_stations.json` hold `darwin-arm64-warp1.18.0` and
`linux-x86_64-warp1.18.0`; `pr14_utilities.json` and `pr15_course.json` only
`darwin-arm64`; `pre_vec3f.npz` is unkeyed (O8). A missing key SKIPS. Warp's CPU backend
compiles with `-march=native` and FP-contraction "Standard" (`warp/_src/context.py:2812`,
`native/clang/clang.cpp:612`), so a key is really (OS, ISA, CPU features, Warp).

1. **The 0.2 files stay byte for byte.** Their tests build with `rng="0.2",
   on_invalid="keep"`, the PR7 `bezier_style` case included (U8). Add the
   `linux-aarch64-warp1.18.0` key by running the 0.2 capture tools against `v0.2.0` on
   spark-home (CPU).
2. **One new family** `tests/goldens/v1_outputs.json` (track / gate generator fields,
   gate-stage digests, relaxed stations, utilities, course states for presets AND
   `Course(CourseConfig)`, launch topology, at the 1.0 defaults), written by one tool
   `tests/tools/capture_v1_goldens.py --fragment <file>` / `--merge <fragments...>`. Under
   U6 it changes only in a major release.
3. Freeze the output-defining code of 16c (its last code commit), then capture:
   `darwin-arm64` locally; `linux-aarch64` on spark-home (CPU, same commit, Warp 1.18.0);
   `linux-x86_64` by a `workflow_dispatch` job `.github/workflows/capture-goldens.yml` on
   `ubuntu-latest` that (a) self-checks the runner (recomputes the committed 0.2
   `linux-x86_64` entries of `pr7` / `pr11` / `pr13a` under `rng="0.2"`, fails unless
   byte-equal), (b) writes the fragment, (c) uploads it; fetched with `gh run download`.
   Fallback: any x86_64 Linux host passing the self-check; if it fails, the key gains a
   CPU qualifier and the 0.2 entries are re-captured from `v0.2.0` on that runner.
4. Merge the three fragments in **one commit** on top of 16c, touching only
   `tests/goldens/v1_outputs.json`.
5. CI: `TRACKGEN_REQUIRE_GOLDENS=1` on the CPU lane's floor leg.

### 2.9 GPU validation (spark-home, NVIDIA GB10) and its scope

Same machine, two venvs (Warp 1.18.0 and latest), trees `v0.2.0` and the PR head. A
scratch tool `pr16_snapshot.py <device> <out.npz> [--rng 0.2|1.0] [--keep] [--edgy]`
(names valid in both trees; `rng` / `on_invalid` passed only where they exist) dumps every
output array of: every track generator (repulsive at E=8) x elevation {flat, uniform,
random_walk, noise} for bezier, {flat, uniform} for the others, seeds `0..7` and the PR7
overflow seeds, calls construction batch / `generate()` / `generate(advance=True)` / a
masked advance; bezier with `style_sampling=True` (`--rng 0.2`); every gate generator x
order {ccw, random_pairs, raw where allowed} x {flat, uniform}; `stations` /
`relaxed_stations`; randomized knobs and `edgy` in {0, 0.1, 0.3, 1, 2.5} (or
`tangent_blend`); `Course(CourseConfig)` and preset courses (the PR15 golden cases), 40
steps; each eager, self-graph and caller capture.

| Check | Expected |
|---|---|
| C0 v0.2.0 twice; head twice | 0 differences |
| C1 v0.2.0 vs head `rng="0.2", on_invalid="keep"` | 0 differences, every case (style and edgy included) |
| C2 v0.2.0 `on_invalid="poison"` vs head `rng="0.2"` (presets, default poison) | 0 differences |
| C3 head default vs head explicit `rng="1.0"` + the defaults | 0 differences |
| C4 head default vs v0.2.0 | differs exactly on seeded built-in outputs; `external` and unseeded pipelines equal (distributions: O12, not this check) |
| C5 eager vs self-graph vs caller capture (head) | 0 differences |

Plus the CUDA lane at both Warp versions, the 16a / 16b / 16d snapshots against v0.2.0
(C1 / C2 hold for them: no output change), timings at E=4096 / 8192 (poison commit,
hashed init, module compile time).

**Scope.** The byte guarantee is "same device architecture, CPU features and Warp
version". O4 makes it structural (the 0.2 kernels run unchanged); the GB10 is one
integrated-memory GPU. Before 1.0.0, C1 also runs on one discrete architecture typical
downstream (sm_86 / 89 / 90) if a host is available; in any case the rc gate (O1, §7)
has each racing repository show byte equality to v0.2.0 with the pins on its own GPU at
the same Warp version.

---

## 3. `on_invalid="poison"` for presets and `BatchSpec` (U5)

- Defaults: `BatchSpec.on_invalid = "poison"`, `presets.track/gates(on_invalid="poison")`,
  `from_dict(/3)` missing -> `"poison"`; `TrackGenConfig.on_invalid` /
  `GateGenConfig.on_invalid = "keep"` (new, appended), passed by `legacy.to_pipeline_spec`.
- What poison does (unchanged code, `stages/commit.py`): a committed env with
  `fail_mask != 0` leaves with `count = 0`, NaN geometry, `length = winding = 0`,
  `valid = 0`, attribute padding; `fail_mask` and `truncated` kept. Gates: `count = 0`,
  every per-gate field NaN.
- Course (E2, presets): track (n_max 100), track (half_width 0.25, uniform z, SDF,
  heightfield, props), gates (posts), gates (frames): 0 differences in owned component
  results and `StepResult` entries (1290, 1590, 1440, 1320 comparisons); differences in
  the course result rows of invalid envs and, in gates mode, `course.checkpoints`
  (`GateCheckpoints` aliases the `GateSequence`). The readback capacity paths read valid
  envs only (`checkpoints.py:236`, `props.py:372`), so poison changes no shape.
- 16c extends E2 into a test: `Course(CourseConfig)` (keep: byte-equal to a 0.2 run under
  `rng="0.2"`), preset courses under keep and poison, and the documented user recipes
  (rl-auto-reset.rst `respawn` / `_agent_k`, which guards `n > 0`; cuda-graph-in-a-sim.rst
  `_sim_k`): no NaN reaches an agent pose the recipe writes, or the recipe is fixed.
- Cost: +1 launch per unmasked preset generation; masked generations unchanged.
- Kit (O6): pinned to keep.

## 4. Configs: mutation guard, required result fields (U11, U12)

What a hard `frozen=True` breaks (E7): library snapshot writes
(`generator_binding.py:160, 185, 201`, `pipeline/legacy.py:268`, `builtin_params.py:230`)
and the derived `spacing` in `__post_init__` (1863 tests fail unfixed, 18 after
`object.__setattr__` there); user patterns: assignment after construction (the guide's
`snapshot` example) and **non-frozen dataclass subclasses failing at class creation**
(the guide's `knobs` example; a frozen subclass of today's class is a `TypeError` too, so
users cannot pre-adapt in 0.2); tests: 18 in 9 files. Unaffected: `dataclasses.replace`
(re-runs `__post_init__`; the documented `spacing=None` reset, from-0.1.rst:506, still
needed), `deepcopy`, pickle, `==`.

Recommended (b), 16d: the §1.3 metaclass flag plus a `__setattr__` guard on declared
dataclass fields only: warns in 1.x, raises in 2.0. The warning names
`dataclasses.replace(cfg, field=v)` and, for `half_width`, `spacing=None`. `CourseConfig`,
`BatchSpec`, `StageRef`, `PipelineSpec` and every `Params` are frozen already.

Required result fields: U12. If chosen anyway: `truncated`, `generation`, `fail_mask`,
`half_width`, `layout` (Track), `generation`, `fail_mask`, `layout` (GateSequence); 12
test modules, `tests/_collision_fixtures.py` and `tests/_oracle` build results by hand.

## 5. `edgy` (U4)

- Today: `BezierParams.edgy` static; `p = atan(edgy)/pi + 0.5` in float64 on the host
  (`warp_generate.py:538` frozen path, `:683` Stage path), a kernel float argument of
  `_vertex_tangents_k`.
- (A) roadmap: `edgy` dynamic, `p` in float64 in the kernel; output change for any
  non-default `edgy`. (B) (A) + the 0.2 host path under `rng="0.2"` (`edgy` static there;
  a device write to the exposed array is ignored: a trap).
- **(C) recommended:** `edgy` static; `tangent_blend: dynamic(lo=0, hi=1)` is the
  kernel's `p` (`_vertex_tangents_dyn_k` reads `p[e]`, arithmetic unchanged); initial
  value `np.float32(atan(edgy)/pi + 0.5)` computed on the host as 0.2 does (a Python float
  reaching a kernel float argument and `np.float32` give the same bits: PR13c E1c, 12018
  values); both given and inconsistent: `ValueError`; `to_dict()` writes `None` unless
  set; `randomize={"generator.edgy": (a, b)}` raises naming `generator.tangent_blend` and
  `(f(a), f(b))`. +1 latch per bezier pipeline, bytes identical at every construction
  value, no `rng` special case.

## 6. Warp floor, conditional lowerings, CPU APIC (U13, U16)

| Roadmap PR16 item | State at fe43a28 | PR16 |
|---|---|---|
| Floor `warp-lang>=1.16` | `>=1.18,<2` since 0.2.0.dev3 (D1, pyproject, CI {1.18.0, latest}); only `_compat`'s pre-1.18 import fallback left (`exec/_compat.py:57, 258`) | remove the fallback; update the roadmap text |
| Conditional lowerings on by default where supported | on in every Warp-registered capture when `wp.is_conditional_graph_supported()`; off by design under `torch.cuda.graph` unless `registered(conditional=True)`, in `exec.unconditional()`, on CPU and in eager CUDA | nothing (U16) |
| CPU APIC self-graph opt-in | not implemented; CPU runs eager; caller CPU captures work | deferred past 1.0 (U13, E8) |

## 7. Split, dependencies, versions, release gate (O1)

Integration branch `v1`, cut from `main` once PR #3 has merged (from `v2@fe43a28`
otherwise, rebased after). Each part bumps `_version.py` / `pyproject.toml`, adds its
CHANGELOG section **and its from-0.2.rst section** (the page is created in 16a, so a
downstream trying any dev release has the guide for it), and is tagged after its GPU
check. Before 16a: U17 (a) on `main`, and 0.2.1 if chosen.

| PR | Content | Output change | Depends on | Version |
|---|---|---|---|---|
| **16a** removals | O9 first (internal callers off the legacy names, warning-free test); §1.1 removals; the metaclass and tombstones (§1.3); `GateGeneratorSpec` retirement (U7) with the byte-preserving port recipe; the style fields' construction warning (U8; style still runs: the keying is still 0.2's); the 1.0 warnings of §1.2 (O10); O11 moves and docs; from-0.2.rst created | none (GB10 snapshot equal to v0.2.0, every case) | PR #3 | 1.0.0.dev0 |
| **16b** RNG plumbing | `ops.rng`; hashed sibling kernels (O4); stream strings (O2); Program salt registry; `rng` on `BatchSpec` / presets / configs (default `"0.2"`; config `None` resolves to `"0.2"` silently in this PR); `Program.rng` / `Course.rng`; spec /3 + /2 migration; ops `keying=`; `tangent_blend` (U4 C); the bezier Stage's style refusal at `rng="1.0"`; property tests on the hashed path; source locks | none at defaults; bezier +1 latch | 16a | 1.0.0.dev1 |
| **16c** 1.0 defaults | `rng="1.0"` default and the config notice (U3 b); `on_invalid="poison"` for presets / `BatchSpec`, the config field `"keep"` (U5); kit pins; 0.2 references pinned to `rng="0.2"` / keep; launch re-pins (preset paths only); seed-picked tests re-picked; O12 distribution gate; E2 extension test; **last commit: the v1 goldens for three platform keys** | U1, U5 | 16b | 1.0.0.dev2 |
| **16d** configs and cleanups | mutation guard (U11 b) or freeze (a); U12 if chosen; O7 internals; planner messages; nitpick allowlist | none | 16c | 1.0.0.dev3 |
| **16e** docs and release | from-0.2.rst final pass; every page; CHANGELOG `1.0.0`; regenerated `docs/_data` (CPU here, GPU on the GB10) and seeded figures; roadmap "Shipped" note | none | 16a-d | 1.0.0rc1 -> 1.0.0 |

16a and 16b touch disjoint code except the bezier stage (16b rebases on 16a). 16d could
run beside 16b; serializing after 16c avoids conflicts in `TrackGenConfig` (16b/16c add
fields).

**rc gate (1.0.0rc1 -> 1.0.0).** (1) The GB10 checks of §2.9 at both Warp versions.
(2) The named racing repositories (the user names them) run rc1 on their own GPUs at
Warp 1.18.0: with the pins (`rng="0.2"`, plus `on_invalid="keep"` on presets), byte
equality to their v0.2.0 outputs; without the pins, their evaluation metrics within a
tolerance the user sets and training runs with no NaN reaching the policy; under
`-W error::DeprecationWarning`, only the 1.0 warnings from-0.2.rst lists. (3) The U17
soak has elapsed.

## 8. Docs: "0.2 -> 1.0"

New page `docs/upgrading/from-0.2.rst` (toctree "Upgrading", next to from-0.1), built like
from-0.1 (impact order, `:Type:` / `:Affects:` / `:Find it:` fields, executed snippets),
grown by each PR:

1. **How to upgrade**: pin first (configs: `rng="0.2"`; presets / `BatchSpec`:
   `rng="0.2", on_invalid="keep"`): same outputs as 0.2.0, and the pins do not warn; fix
   the breaking items; run with `-W error::DeprecationWarning` to find the 1.0 warnings;
   then drop the pins, re-baseline, and log `rng` with the seeds and the Warp version.
2. **Triage table** with grep patterns (`set_capturing`, `.track`, `.params`,
   `style_sampling`, `GateGeneratorSpec`, `order_points`, `GenScratch`,
   `"track_gen.pipeline/1"`, the dead field names, `asdict(`/`vars(` consumers of them,
   8+ positional `TrackGenConfig(` arguments, `cfg.<field> =`, `edgy`, `% count`,
   `* valid`, `max_checkpoints`).
3. **Silent changes**: the RNG keying (every seeded track from every entry point;
   configs say so once, U3 b; a stored config dict takes the new keying);
   `max_checkpoints` on the readback path (a shape change, with the fix); poison for
   presets (`count` can be 0: guard `% count`; select, not multiply; what a `Course`
   user sees: §3); configs keep `keep`.
4. **Breaking**: each removal with its replacement; `GateGeneratorSpec` names retired
   (port recipe); spec format /3 (0.2 cannot read 1.0 dicts); `/1` refused; tombstones
   and what they cannot cover (U9); `style_sampling=True` without `rng="0.2"`.
5. **Deprecated in 1.0** (§1.2, the style fields, config mutation, `Progress(checkpoints=)`),
   each "removed in 2.0".
6. **New**: `ops.rng`, `keying=`, `rng`, `Program.rng` / `Course.rng`, the config
   `on_invalid`, `tangent_blend` (with the `edgy` mapping and the distribution caveat).
7. **Stability in 1.x** (U6) and **the 2.0 schedule** (§10).

Every other page: writing-a-generator.rst's RNG section (the stream table of §2.3; the
linear table moves to an `rng="0.2"` note); plugin-abis.rst's status table ("every 1.x
release" / "deprecated, removed in 2.0") and its legacy sections; writing-a-gate-generator
.rst: a byte-preserving port of a `GateGeneratorSpec` (an anchors Stage that provides
`order` and writes the same buffers, so no order / normalize stage runs after it, as with
the adapter); configuration/reference.rst (style, dead fields, inert gate fields,
`on_invalid`, `rng`); generators/bezier.rst (style under `rng="0.2"`, `tangent_blend`);
user-guide/results.rst (`CourseLine.track`, poison for presets, keep for configs,
optional fields), generating.rst, curricula.rst; how-it-works/gate-pipeline.rst;
reference/api.rst (policy: "a deprecated name warns, attributed to your code, and keeps
working through the current major version; it is removed at the next major. Names under
`track_gen._src` that plugin-abis.rst documents follow the same rule; other private names
may change in any release"), plugin.rst, runtime.rst, the ops page (`ops.rng`); every hit
of §1.4; from-0.1.rst gets a banner, its "Removed in 1.0" table corrected (`_CAPTURING`,
the not-removed rows now "2.0") and its three snippets fixed (O11), snippets that assert
0.2 numbers re-measured (E5: `from-0.1.rst::gates`); `docs/_data` regenerated; seeded
figures re-rendered; the docs check's version-jargon rule keeps `upgrading/*` as the only
place that names releases.

## 9. Test plan

Common gate per part: the CPU lane (`-m "not cuda and not slow and not benchmark"`) at
Warp 1.18.0 and latest (CI ubuntu; local darwin-arm64; spark-home linux-aarch64), the
CUDA lane on the GB10 at both versions, `sphinx -W` + doc snippets, the §2.9 snapshots.

- **16a**: O9's warning-free test (every built-in pipeline, `Course` via config and
  presets, the kit, under `simplefilter("error", DeprecationWarning)`) and call-graph
  probe; deleted surface -> `AttributeError` / `ImportError` / `ValueError` naming the
  replacement; every §1.2 warning: fires once, attributed to the caller, names "2.0"
  (parametrized over the full alias list), `wp.func` / kernel names inside a user kernel
  warn at the kernel's definition line (E11); tombstones: `TrackGenConfig(**asdict(cfg_0_2))`
  and `Sub(**asdict(sub_0_2))` load with one warning, `dataclasses.replace(TrackGenConfig(),
  num_envs=4)` emits nothing, `cfg.min_angle` warns and returns 0.3, `smooth_finish=True`
  and a bad `output_mode` raise, 8 positional arguments raise, the oracle's private
  subclass keeps its fields; style: non-default values warn at construction (config,
  `BezierParams`, `generator_params`), defaults and round trips silent, outputs unchanged;
  `GateGeneratorSpec`: an entry point registering one plus a Stage generator keeps the
  Stage generator, binding the spec raises the port message, the v0.1 toy gate plugin of
  `test_legacy_plugin_abi.py` ported to a Stage reproduces its pinned bytes;
  `Progress(checkpoints=)` warns; /1 refused; snapshot C1 / C2 on the GB10.
- **16b**: all goldens and parity suites unchanged at defaults; the 0.2 kernel source
  table; the hashed path (`rng="1.0"`): `check_generator` / `check_stage` for every
  built-in, per-env == scalar runs, determinism, epoch advance, seed-0 streams distinct,
  masked exactness; salt table, rename-invariance, Program-wide collision `PlanError`;
  host state-collision test; CUDA-source lock for every sibling and shared kernel; `ops`
  `keying="linear"` bitwise vs head, `"hashed"` equals `ops.rng.init_at`; /2 -> `"0.2"`,
  /3 round trip; style refused at `"1.0"`, byte-equal at `"0.2"`; `tangent_blend`:
  per-env arrays == scalar runs, initial value bit-equal to the host transform for E6's
  value sets, `to_dict` writes `None`, the `edgy` randomizer message; launch pins +1
  (bezier).
- **16c**: the 0.2 golden tests under `rng="0.2", on_invalid="keep"` (+ aarch64 keys);
  `v1_outputs.json` on three keys; launch re-pins for preset paths (+1 unmasked poison
  commit), config paths unchanged; parity tests vs frozen references pinned to `"0.2"` /
  keep; seed-picked tests re-picked (E5); kit pins; the E2 extension (§3); the config
  notice (one `FutureWarning` per process, silent with `rng` set, absent from
  library-written specs); O12; `TRACKGEN_REQUIRE_GOLDENS`; snapshots C1-C5.
- **16d**: U11 (b): a field assignment after construction warns once per site, private
  attributes silent, library snapshots silent, `replace` / `deepcopy` / pickle silent,
  subclass `__post_init__` assignments silent; the 18 E7 tests rewritten to `replace`;
  or (a); internal `ctx` required (LaunchTrace probe unchanged); snapshot C1 / C2.
- **16e**: docs build, link check, every snippet (CPU and CUDA groups), the docs tools
  tests; final CPU lane at both Warp versions on the three platforms; CUDA lane and §2.9
  on the GB10; the rc gate.

## 10. Risks, 1.x policy, 2.0 schedule

- RNG siblings beside every built-in RNG kernel: the 0.2 kernels are untouched (source
  hashes), the siblings checked by the source locks and C3 / O12.
- `linux-x86_64` goldens from a GitHub runner: guarded by the self-check (§2.8).
- The poison + RNG re-baseline in one commit: fragments captured from one frozen commit;
  any later output-affecting fix re-captures all three.
- The FutureWarning notice for configs (U3 b) is one line per process; downstream suites
  with `filterwarnings = error` must set `rng` (the point of the notice).
- The metaclass: a user subclass that also declares another metaclass must combine them
  (documented).
- **1.x policy (U6):** byte-stable built-in outputs for fixed config, seeds, device
  architecture, CPU features and Warp version; output-changing fixes opt-in until 2.0.
- **2.0 schedule** (no earlier than 6 months after 1.0; each item warned through 1.x):
  `rng="0.2"` and the 0.2 kernels; /2 dicts and `rng: "0.2"` dicts / configs raise
  (U3 c); the frozen functions, scalar and style kernels; every §1.2 name; the style
  fields (default values tombstoned through 2.x); the dead-field tombstones become
  `TypeError`; `GateGeneratorSpec`; config mutation raises (U11); the default flips of
  any 1.x opt-in fixes.
- Non-goals: MixtureProgram, open topology (post-1.0); the PR7 merges (U6);
  `build(inputs=)`, `MixedBatch`, int32 / categorical draws; GPU-specific goldens;
  declared `z_min` / `z_max` bounds (U1).

## 11. Evidence files (`/private/tmp/claude-501/-Users-antoiner-code-TrackGen/643abbd6-07f3-4c7f-a177-08464b07efe0/scratchpad/pr16-scratch/`)

`probe_removals.py`, `audit.txt`, `audit.json` (E1); `exp_poison_course.py`,
`exp_poison_course.out` (E2); `lane_xpoison.txt` (E2b; patch: `xpoison/`); `exp_salts.py`,
`exp_salts.out` (E3); `exp_yield.py`, `yield_base.json`, `yield_xrng.json` (E4);
`lane_xrng.txt` (E5; patch: `xrng/`); `exp_edgy.py`, `exp_edgy.out` (E6);
`lane_xfrozen.txt`, `lane_xfrozen2.txt` (E7; patch: `xfrozen/`); `exp_cpu_apic.py`,
`exp_cpu_apic.out` (E8); `lane_base.txt` (E9; `base/` = `git archive fe43a28`);
`critic-deprec/` (E10: `p_decorator_sim.py`, `p_style_nonbezier.py`, `p_bezparams.py`,
`p_roundtrip.py`, `cases/`, `run_cases.out`); `arch-r2/funcwarn2/` and
`critic-deprec/funcwarn/` (E11); `arch-r2/meta_proto.py` (E12);
`critic_downstream/modzero.py` (E13).

## 12. Critic review (revision 2)

Every issue was checked against fe43a28 (code read, probes re-run: E10-E13). Verdicts:
**A** accepted as proposed; **A\*** accepted, with a different fix (reason given);
**R** a sub-point rejected (reason given).

| # | Issue | Verdict | Resolution |
|---|---|---|---|
| 1 | Live stages call legacy helpers / frozen `apply_z_profile` (8 warnings from default generation) | A | Verified (`stages/track.py:256, 367, 542`, E10). O9: internal callers move first; warning-free test and call-graph probe in 16a |
| 2 | `style_sampling=True` on non-bezier, ranges with style off: silent in 0.2, `ValueError` in rev. 1 | A\* | Verified (E10). Not raised. But "accept and ignore" for non-bezier would silently drop a legacy plugin's frozen-bezier style draws, so U8 keeps the fields (they still reach that plugin) |
| 3 | `BezierParams` routes (construction, `replace`, reads, StageRef, schema) unwarned | A\* | Verified. U8 keeps the four fields in `BezierParams` through 1.x (warn on non-default), so every route behaves alike with no wrapper |
| 4 | Frozen bezier function honours style silently vs deleting the style kernels | A | Verified (`warp_generate.py:616, 828`; the Stage path uses `_style_sample_k` too). Option 1: style kernels kept with the frozen function to 2.0; listed in §1.2 |
| 5 | Subclass `__init__` misses the tombstone | A | Metaclass `__call__`, inherited (§1.3, E12) |
| 6 | `output_mode` sentinel warns on every `replace` | A | Verified (109 kwargs). `output_mode` joins the tombstone + read property; test `replace(TrackGenConfig(), num_envs=4)` silent |
| 7 | Helper list incomplete; a decorator cannot wrap kernels / funcs; codegen attribution | A | Full alias set in §1.2; module `__getattr__` with a `warp.*` skip, once per name (O10). Measured: the in-kernel use then warns at the user's `@wp.kernel` line, visible (E11), so no "best-effort" caveat is needed |
| 8 | Removing `GateGeneratorSpec` quarantines a plugin's other generators | A | Class kept; registration retires the name only; test with a mixed entry point (U7) |
| 9 | `tools.Progress(checkpoints=)` alias "kept for 0.x"; lifetime phrase audit | A | §1.2 row; §1.4 audit (54 promise lines, 94 "0.x" mentions) |
| 10 | /2 lifetime after `legacy_rng`; config dicts asymmetric | A | U3 (c) 2.0 refusal; U3 (b) config notice; from-0.2 pins |
| 11 | `_sample_sites_k` shared with a frozen function; shared ops kernels | A\* | Solved structurally: 0.2 kernels untouched, hashed siblings (O4); `sample_voronoi_sites` listed with the survivors |
| 12 | `_src` overwrite row misplaced; gate `register` has no valid payload | A | Row moved to §1.2; the gate `register` retires the name like `registry.add` |
| 13 | Dead-field reads, `smooth_finish=True`, positional break | A | Read properties; `smooth_finish=True` raises; positional break listed with grep pattern |
| 14 | Executed snippets on removed surface; api.rst; viz / benchmark / `_warp_compare` users | A | O11 |
| 15 | `legacy_rng` removed in 1.1 contradicts D8 | A / R | Recommendation changed to 2.0 (U2). R: D8 rejects a one-minor window for the config tier, not this hatch; cited as analogous reasoning, and flagged as a deviation from the roadmap / spec |
| 16 | Poison default for the config tier and E2's scope | A | U5: configs keep `"keep"` for 1.x; E2 extension test (§3); hazards in from-0.2 (E13) |
| 17 | Config-tier reproducibility; no 0.2 announcement | A | U17 (a) docs on `main`, (b) 0.2.1 option; U3 (b) notice; U2 `Program.rng` / `Course.rng` |
| 18 | No 1.x output-stability policy | A | U6 policy; PR7 merges become 2.0 candidates or opt-ins |
| 19 | Ranges with style off raise with no 0.2 warning | A\* | Same as #2 (U8) |
| 20 | `legacy_rng` cannot reproduce style runs | A | U8: style runs under `rng="0.2"`; PR7 `bezier_style` case kept |
| 21 | Subclass round trips (dup of #5) | A | §1.3 metaclass; also carries the U11 flag |
| 22 | `max_checkpoints` shape change on the readback path | A / R | Documented with the fix (U1, from-0.2). R: landing declared z bounds in 1.0 adds clamping to two unbounded knobs (a second output change); the readback is deterministic, so it does not threaten 1.x stability |
| 23 | `_src` ABI documented in plugin-abis.rst; 1.1 removal too fast | A | U10: 2.0 for everything, ≥ 6 months after 1.0 |
| 24 | Legacy specs once the hatch is gone (dup of #10) | A | U3 (c) |
| 25 | E4 too narrow; C4 yield clause underpowered | A | O12 in 16c; C4's yield clause dropped |
| 26 | GPU byte guarantee checked on one integrated GPU | A | O4 makes it structural; guarantee scoped; discrete-arch C1 if available; downstream rc gate on their GPUs |
| 27 | Pinned `legacy_rng` errors under `-W error` | A | No warning on `rng="0.2"` in 1.0 (U2) |
| 28 | `replace` keeps a stale `spacing`; guard on all attributes | A / R | Guard limited to declared fields; message names `spacing=None`. R: auto re-deriving `spacing` in `replace` would change the outputs of 0.2 code that relies on it |
| 29 | Salts depend on stage ids | A | Namespace strings (§2.3, O2); rename-invariance test |
| 30 | Boolean `legacy_rng` does not scale; keying not logged | A | `rng="0.2" | "1.0"`, exposed on Program / Course / spec (U2) |
| 31 | `tangent_blend` vs the promised dynamic `edgy` | A | Mapping and caveat documented; converted range in the error; `to_dict` writes `None` (U4) |
| 32 | Ported gate generators change output | A | Byte-preserving port recipe (provides `order`) and a test porting the v0.1 toy plugin (§8, §9) |
| 33 | from-0.2 only in 16e; rc gate undefined; no soak | A | Each PR ships its section; concrete rc gate (§7); U17 (c) soak |

---

## 13. USER DECISIONS (2026-10-25) — these override §0.1 and everything above

The user answered the U-list differently from the recommendations, and added: "this is not
production software, so we can break stuff" and "I also would not call this 1.0, maybe 0.3?
So we have time to do more fixes."

1. **Version: 0.3, not 1.0.** The cleanup ships as 0.3.0 (dev releases 0.3.0.devN on branch
   `v3`, worktree /Users/antoiner/code/TrackGen/.claude/worktrees/v3, cut from main = v0.2.0).
   1.0 is a later, separate milestone. Rename "1.0" to "0.3" everywhere the cleanup is
   meant (docs promises "removed in 1.0" become "removed in 0.3" where the removal happens
   now).
2. **RNG: breaking is fine, NO byte-for-byte reproduction mode** (no `rng="0.2"`, no
   `legacy_rng`). The built-ins switch to the hashed keying; goldens re-baselined once.
3. **`on_invalid="poison"` default EVERYWHERE**, configs included.
4. **AGGRESSIVE removals:** remove all deprecated and legacy surface in 0.3 regardless of
   whether it warned in 0.2.0 (no tombstones, no deprecation cycles, no escape hatches):
   the `_src` helpers / registries / re-export hooks, the frozen `generate_*_warp` /
   `*_alloc_scratch` / `apply_z_profile*` / float sampling helpers and their scalar and style
   kernels, GeneratorSpec / GateGeneratorSpec legacy ABIs (decide: keep `GeneratorSpec`
   only if it is still the documented plugin path; otherwise Stage only), `style_sampling`
   and its ranges, `output_mode`, the dead config fields, `set_capturing` / `_CAPTURING`,
   the `.params` alias, `CourseLine.track`, the `/1` spec format, inert params
   (`validity.gate_default`'s gate_width / check_crossing), `tools.Progress(checkpoints=)`,
   the `capturing=` args, etc. **Frozen configs (`frozen=True`) and required Track /
   GateSequence fields now.**
5. **Timeline:** dev releases now; 0.3.0 later, after an rc the racing repos test.

Breaking is the default answer; still list every break in the CHANGELOG and a "0.2 -> 0.3"
migration page, and still keep internal byte-identity guards wherever no output change is
intended (removals must not change outputs; only the RNG switch and the poison default
change outputs).
