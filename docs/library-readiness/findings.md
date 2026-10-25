# TrackGen library-readiness review — all verified findings

> Companion to [`README.md`](README.md). Commit `1d8a6e5`. Ranked by post-verification severity. Static analysis only.

Each finding was checked by two independent skeptics: one checked it against the code, one questioned its impact. Severity is the one after verification. *Category* is the original reviewer's tag.


## Critical (3)

**A1. Generation APIs (TrackGenerator/GateGenerator/Course.generate) cannot be recorded by a caller's CUDA graph** — *graphability* · `track_gen/_src/track_generator.py:182`, `track_gen/_src/track_generator.py:198`, `track_gen/_src/track_generator.py:200`  
Every CUDA generate() takes the process-wide threading.Lock (_CAPTURE_LOCK, track_generator.py:182, gate_generator.py:165, course.py:493-521). On the first call it runs 3 warmups, then wp.synchronize() (track_generator.py:198, gate_generator.py:177, course.py:513), then opens its own `with wp.ScopedCapture(device=dev)` (track_generator.py:200, gate_generator.py:179, course.py:514). On every call it ends with …  
→ *Fix:* Split each facade into two layers: (1) A capture-safe `launch()`/`record()` that only enqueues the pipeline on the current stream of `config.device`: no lock, no sync, no ScopedCapture, no allocation. It is valid under a caller's Warp or external capture and also eager. (2) An optional convenience `generate()` that …

**A2. Course.generate() cannot be recorded inside a caller's CUDA graph (nested capture, graph replay, host syncs, allocation, private lock)** — *graphability* · `track_gen/_src/course.py:486`, `track_gen/_src/course.py:481`, `track_gen/_src/course.py:501`  
First CUDA call: course.py:514-517 runs `with wp.ScopedCapture(device=self._device) as cap: self._refresh()`, followed by `wp.capture_launch(self._refresh_graph); wp.synchronize()` (520-521). Later calls run `wp.capture_launch(self._refresh_graph); wp.synchronize()` (501-502). Before either, generate() calls self.generator.generate() (486), which itself does wp.synchronize() (track_generator.py:198, 208; …  
→ *Fix:* Split each self-capturing API into two layers: (a) A pure enqueue method, e.g. Course.enqueue_generate() = generator._run() + self._refresh(). It only launches kernels on the current stream: no ScopedCapture, no capture_launch, no synchronize, no allocation, no lock. Callers can record it into any graph or put it …

**A3. Repulsive generator cannot be graph-captured, by itself or by a caller, so TrackGenerator.generate(generator='repulsive') is uncapturable** — *graphability* · `track_gen/_src/warp_generate_repulsive.py:845`, `track_gen/_src/warp_generate_repulsive.py:784`, `track_gen/_src/warp_generate_repulsive.py:861`  
Exhaustive list of blockers in generate_repulsive_warp: (1) A host readback plus a data-dependent exit, `if int(s.frozen.numpy().sum()) >= E: break` (845-846). This is a D2H copy with an implicit synchronize, and it is not gated by any capture flag. (2) Each stage transition calls the PUBLIC `_pipe.arc_length_resample_inplace` (784), which ends in `_sync(dev)` (warp_pipeline.py:829) and therefore wp.synchronize() …  
→ *Fix:* Make the growth loop fixed-budget and rely on the existing device-side frozen mask. (a) Remove the readback and break, or keep them only under an explicit `early_exit` eager-only option. All O(N^2) kernels already early-return for frozen envs, so post-convergence iterations cost only launch latency plus O(E*N) work (a …


## High (7)

**A4. Mutable config objects held by reference; CUDA graphs bake scalar args and buffer sizes, so post-capture mutation silently diverges from CPU or writes out of bounds** — *correctness* · `track_gen/_src/types.py:53`, `track_gen/_src/types.py:863`, `track_gen/_src/types.py:851`  
TrackGenConfig and GateGenConfig are plain mutable `@dataclass`es (types.py:53, 863), and __post_init__ itself mutates `spacing` (851-852). TrackGenerator stores the caller's object (`self._config = config`, track_generator.py:84). Course passes `config.gen` through (course.py:331); CourseConfig is frozen, but the inner gen config is not. Every launch reads config at call time and passes kernel scalars by value: …  
→ *Fix:* Make the configs `frozen=True` (set derived fields via object.__setattr__ in __post_init__), copy them at construction, and expose a read-only `generator.config`. If some scalars should be tunable between replays (e.g. per-env style ranges), move them into device-resident parameter arrays read by kernels. Anything …

**A5. Generator registries are private, with hard-coded module lists, no entry points, unsafe duplicate semantics, and private building blocks** — *extensibility* · `track_gen/_src/generator_registry.py:48`, `track_gen/_src/generator_registry.py:57`, `track_gen/_src/gate_generator_registry.py:27`  
GeneratorSpec, register, get and available live only in track_gen._src.generator_registry and are not re-exported (track_gen/__init__.py:9-40). _ensure_loaded hard-codes six imports ('Add one import line per new generator', generator_registry.py:57-63). The gate registry hard-codes `_GATE_GENERATOR_MODULES` (gate_generator_registry.py:27-32). Duplicate handling differs between the two registries: - Track register() …  
→ *Fix:* Add a public `track_gen.generators` module with: - `TrackGeneratorSpec` / `GateGeneratorSpec` (or Protocol classes), `register_track_generator`, `register_gate_generator`, and `available()`. - An explicit `override=` duplicate policy. - Lazy entry-point discovery …

**A6. Z-profile set is closed: string if/elif dispatch plus name tuples scattered across three modules, knobs on monolithic configs** — *extensibility* · `track_gen/_src/warp_zprofile.py:202`, `track_gen/_src/warp_zprofile.py:216`, `track_gen/_src/types.py:24`  
apply_z_profile chains `if profile == "flat" ... elif "uniform" ... elif "random_walk" ... else: # "noise"` (warp_zprofile.py:202-222), so an unknown name silently runs noise if validation is bypassed, for example by mutating the config. Membership is a literal set in _validate_z_fields (types.py:24-29). Whether a profile is knot-based is a name tuple in two places: _fill_knot_scratch (warp_pipeline.py:1505) and …  
→ *Fix:* Add a ZProfileSpec registry {name, params_cls, kernel launcher (seeds, count, stride, cum, perim, z, params), parameterization: 'per_point'|'knots', needs_cum_perim}. Validate names against the registry. Derive the knot/per-point choice from the spec. Pass profile params as an object rather than flat z_* fields. Raise …

**A7. Pipeline stage sequence is hard-coded; relaxation, inflation/width and validity are not pluggable** — *extensibility* · `track_gen/_src/warp_pipeline.py:889`, `track_gen/_src/warp_pipeline.py:1699`, `track_gen/_src/track_generator.py:76`  
_run_pipeline (warp_pipeline.py:889-968) is a fixed sequence: generator → resample_constant_spacing → warp_relax.band_l0_inplace → warp_relax.xpbd_solve_inplace → inflate_warp. inflate_warp (1699-1950) is also fixed: resample_uniform → frame → arclength → winding → offset → z-profile → validity → lift. - Relaxation: TrackGenerator asserts `relax_solver == 'xpbd'` and `not smooth_finish` (track_generator.py:76-83), …  
→ *Fix:* Define a Stage protocol: `alloc(config, layout) -> state` and a capture-safe `launch(ctx)`, operating on a typed buffer bag with declared reads and writes (centerline2d, count, w, z, outer, inner, ...). Build the pipeline from named stage registries (relaxers, inflaters, elevation, validators), configured per stage. …

**A8. Constant half-width and a 2D-centerline-only generator output are baked through every stage; there is no per-point attribute channel** — *extensibility* · `track_gen/_src/generator_registry.py:27`, `track_gen/_src/warp_pipeline.py:1553`, `track_gen/_src/warp_pipeline.py:1066`  
GeneratorSpec.generate writes only `[E*num_points] vec2f` + `[E] valid` (generator_registry.py:27-30). The orchestrator owns only gen_centerline/gen_valid (warp_pipeline.py:1553-1554), and resample_constant_spacing resamples only the vec2f centerline. XPBD uses `target = 2.0 * hw * (1.0 + margin)` and `R_min` from config.half_width (warp_relax.py:384-387), band setup uses `two_hw` (551), inflation runs …  
→ *Fix:* Introduce a CenterlineBatch output: required xy plus optional named per-point channels declared by the spec (half_width, z, bank...). Have the resampler interpolate all channels with the same (seg, s) tables. Make relax, offset and validity take a per-point width array (the constant path fills it). Add …

**A9. Gate-generator contract is private and coupled to track-generator internals** — *modularity* · `track_gen/_src/warp_gate.py:777`, `track_gen/_src/warp_gate.py:625`, `track_gen/_src/warp_gate.py:843`  
Gate generators receive `_GateStaging(pos2, out.count)` (a private class, warp_gate.py:777-790, 842-843) and must fill it via `warp_gate.finish_ordered_gates` and `alloc_order_scratch` (538-548, 625-648). Built-in gate generators import private kernels and constants from the track generators: `_polar_controls_k, _BASE_RADIUS, _BEZIER_EXTENT` (warp_generate_polar_gates.py:6-11), `_sample_sites_k, …  
→ *Fix:* Introduce a public `AnchorSampler` stage (seeds, params -> [E*K] vec2f anchors + [E] count) that both a track generator (anchors -> curve) and a gate generator (anchors -> ordered gates) can consume. Make GateStaging public and documented. Move the shared kernels into a public `track_gen.primitives` module, and add a …

**A10. Generator contract is '2D anchors + count' only; the pipeline unconditionally overwrites z, tangents, orientation and size, assumes a closed loop, and checks spacing in XY** — *extensibility* · `track_gen/_src/warp_gate.py:777`, `track_gen/_src/warp_gate.py:814`, `track_gen/_src/warp_gate.py:853`  
The generator receives only `_GateStaging(position=pos2 vec2f, count)` (777-790; pos2 is `wp.empty(E*G, dtype=wp.vec2f)` at 814). After generate(): - z comes solely from config.z_profile (853-855), and `_lift_positions_k` overwrites out.position (856-861); - tangents are recomputed by closed-loop central differences, `prev_i = (i + cnt - 1) % cnt` (354-356; 862); - orientation is the roll-free `_frame_quat(fwd)` …  
→ *Fix:* Widen the staging contract. - Stage vec3f positions, plus optional tangent, roll, half_size and per-gate tag channels. - Add capability flags on the spec (e.g. provides_z, provides_tangent, provides_roll, provides_size, closed: bool). - The pipeline skips or adapts the stages a generator already provides: z-profile …


## Medium (40)

**A11. Host-sync suppression relies on three private, process-global _CAPTURING flags; only one is public, and coverage depends on hand-wired transitive knowledge** — *graphability* · `track_gen/_src/runtime.py:24`, `track_gen/_src/runtime.py:57`, `track_gen/_src/warp_pipeline.py:31`  
There are three separate module-global flags: - runtime._CAPTURING (runtime.py:24), toggled by the public set_capturing (runtime.py:57-68). - warp_pipeline._CAPTURING (warp_pipeline.py:31). - warp_gate._CAPTURING (warp_gate.py:15). warp_relax instead takes an explicit `capturing=False` parameter (warp_relax.py:337-338, 510, 556), which _run_pipeline fills from warp_pipeline._CAPTURING (warp_pipeline.py:944, 952). …  
→ *Fix:* Remove the post-launch host syncs entirely. Warp launches are stream-ordered, and every host readback path (`.numpy()`) already synchronizes, which also removes all three flags. If eager-mode syncing is kept as a convenience, gate it on `wp.get_device(device).is_capturing`, which detects Warp captures and external …

**A12. _sync uses global wp.synchronize(): syncs every GPU, raises if any device is capturing, and stalls unrelated simulator streams (~48 hot-path call sites)** — *threading-state* · `track_gen/_src/runtime.py:53`, `track_gen/_src/warp_pipeline.py:47`, `track_gen/_src/warp_gate.py:30`  
Every _sync does `if "cuda" in str(device): wp.synchronize()` (runtime.py:53-54, warp_pipeline.py:47-48, warp_gate.py:30-31, warp_relax.py:510-511, 556-557). Warp's synchronize() loops over ALL CUDA devices that have a context, raises if `device.is_capturing` for any of them, and otherwise runs cuCtxSynchronize on each (warp context.py:12411-12432). Device.is_capturing is true for any Warp capture on that device …  
→ *Fix:* Remove the per-call host syncs. Warp launches are stream-ordered, so consecutive library and Warp operations need no host sync, and .numpy()/wp.to_torch consumers either sync themselves or follow stream rules. If a blocking mode is wanted for debugging, make it opt-in (e.g. `sync: bool = False` on constructors) and …

**A13. _CAPTURE_LOCK: process-wide, non-reentrant, partial coverage; serializes all GPUs, deadlocks composed plugins, and forces Course.generate to be non-atomic** — *threading-state* · `track_gen/_src/runtime.py:34`, `track_gen/_src/runtime.py:11`, `track_gen/_src/track_generator.py:182`  
`_CAPTURE_LOCK = threading.Lock()` (runtime.py:34) rests on the premise 'there is one CUDA device stream, so at most one graph capture/replay may be in flight process-wide' (runtime.py:11-13). It is taken only in TrackGenerator.generate (track_generator.py:182), GateGenerator.generate (gate_generator.py:165) and Course.generate (course.py:474, 493, 500, 506). (a) The comment admits 'a concurrent capture, replay, or …  
→ *Fix:* Remove the root cause, the shared current stream, instead of locking around it. Give each generator or Course a private `wp.Stream(device)`, pass `stream=` explicitly to every wp.launch, wp.copy, capture_begin/ScopedCapture and capture_launch (ScopedStream is process-global, so it cannot be used here), and hand off to …

**A14. No stream parameter; library relies on Warp's process-global current stream, so caller/torch streams are not respected and external captures silently omit library work** — *api-surface* · `track_gen/_src/track_generator.py:200`, `track_gen/_src/track_generator.py:207`, `track_gen/_src/runtime.py:45`  
No wp.launch, wp.copy, ScopedCapture or capture_launch in track_gen passes stream=, and a grep finds no ScopedStream, get_stream, stream= or is_capturing in track_gen or tests. All work therefore goes to `device.stream`, which Warp stores per Device and process-wide (warp context.py:6124-6145). ScopedStream changes it for all threads (utils.py:1628-1638). Correct ordering against torch consumers currently comes only …  
→ *Fix:* Accept `stream: wp.Stream | None = None` on constructors and/or per call. Default it to `wp.get_stream(device)` resolved at call time, and pass it explicitly to every launch, copy and graph operation. Document the interop recipe (stream_from_torch, or event handoff). Optionally expose `record_event()` / `wait_event()` …

**A15. All track knobs are host scalars frozen at capture; curriculum or per-env domain randomization requires rebuilding and rebinding everything** — *graphability* · `track_gen/_src/types.py:566`, `track_gen/_src/track_generator.py:94`, `track_gen/_src/warp_pipeline.py:1776`  
Only rad, scale and handle_clamp_frac have a device-side per-env path (style_sampling with *_range, types.py:566-570), and even their ranges are host scalars. Everything else (half_width at warp_pipeline.py:1776, spacing, z ranges, relax params, generator knobs) is passed by value. The Track is allocated per TrackGenerator instance (track_generator.py:94-96), so changing a parameter means a new instance and new …  
→ *Fix:* Give each stage a device-resident params buffer (a wp.struct array of size [E] or [1]) that kernels read by pointer. Expose `set_params(**kw)` / `params_warp` that write into it in place (capturable), and keep shapes (N_max, num_points, iteration counts) as the only construction-time values.

**A16. Monolithic TrackGenConfig: generator-specific and dead fields, no per-generator parameter namespace, validation hard-codes generator names** — *extensibility* · `track_gen/_src/types.py:537`, `track_gen/_src/types.py:553`, `track_gen/_src/types.py:583`  
TrackGenConfig has 107 fields (types.py:537-674). Prefix counts: checkpoint_ 8, voronoi_ 5, repulsive_ 19, polar_ 3, hull_ 1. There are also 9 bezier/hull point-sampler fields and 4 bezier-only style-sampling fields, for about 48 generator-specific fields. Fields never read by the runtime: energy_* 6, tp_* 4, smooth_finish* 3, relax_chunk_size, relax_use_warp, max_regen_iters, min_angle (no runtime reads), …  
→ *Fix:* Split the config into a core section (num_envs, device, N_max, num_points, half_width, spacing) and per-stage sections. Give each generator and stage its own frozen params dataclass, declared on its Spec (`params_type`, `validate(core, params)`). TrackGenConfig then holds `generator_params: GeneratorParams` (and …

**A17. No device-side reseed/advance: replaying a caller's graph regenerates identical tracks** — *graphability* · `track_gen/_src/track_generator.py:188`, `track_gen/_src/track_generator.py:138`, `track_gen/_src/gate_generator.py:166`  
Every pipeline kernel derives randomness statelessly from the seed buffer, for example `wp.rand_init(seeds[e] * _P_UNIFORM + i)` (warp_zprofile.py:86, 104, 150) and the generator salts (warp_generate.py:51,77,111). The facades refresh the buffer by a host-issued `wp.copy(self._seed_buf, self._rng.seeds_warp)` before every run or replay (track_generator.py:188, 216; gate_generator.py:166, 189). The docstring admits …  
→ *Fix:* Add a capturable `advance(mask: wp.array | None)` on the seed source: a 1-thread-per-env kernel `seeds[e] = hash(seeds[e], states[e]); states[e] += 1` for masked envs. Alternatively keep base seeds fixed and add a device epoch counter that kernels consume via `wp.rand_init(seed, epoch * STAGE + k)` and that the graph …

**A18. Registry duplicate/ordering semantics: silent clobbering in one registry, a full lookup failure in the other** — *extensibility* · `track_gen/_src/generator_registry.py:48`, `track_gen/_src/generator_registry.py:58`, `track_gen/_src/gate_generator_registry.py:35`  
Track: `GENERATORS[spec.name] = spec` overwrites unconditionally (48-49). Because `import track_gen` does not load generator modules (nothing in the package imports warp_generate at module level), a plugin that registers a name that already exists before the first get() is later overwritten by `from . import warp_generate` (58) without notice. Registering after load silently replaces the built-in process-wide; tests …  
→ *Fix:* Use one shared Registry class for both kinds, with `register(spec, *, replace=False)` that raises on collision unless replace=True. Make loading idempotent per module with no import side effects. Add `unregister(name)` and a context manager for tests. Guard first-time loading with a lock.

**A19. Validity gate is one monolithic fused kernel with fixed checks and a single bit** — *extensibility* · `track_gen/_src/warp_pipeline.py:624`, `track_gen/_src/warp_pipeline.py:712`, `track_gen/_src/warp_pipeline.py:1184`  
`out[e] = gv & turn_ok & w_ok & no_nan & th_ok & border_ok & grade_ok` (712). It is configurable only through validity_border_check (1916) and z_valid_grade (694). validity_inplace pulls fixed scalars from config (1184-1186).  
→ *Fix:* Use a per-env uint32 fail_mask. Each registered check kernel ORs in its own bit, and valid = (mask == 0). Expose the mask for diagnostics and benchmarks.

**A20. vec2f, closed, simple-loop topology is hard-wired; generators cannot supply z, open tracks or crossings** — *extensibility* · `track_gen/_src/warp_pipeline.py:1553`, `track_gen/_src/generator_registry.py:27`, `track_gen/_src/warp_pipeline.py:657`  
The generator output buffer is `wp.empty(E * N_gen, dtype=wp.vec2f)` (1553), and the contract is a "closed centerline ([E*num_points] vec2f)" (generator_registry.py:27-29). All kernels wrap indices modulo count (e.g. 82, 155-157, 389). offset picks outer/inner by shoelace |area| (89-92, 124-131), which is only meaningful for closed loops. Validity requires `abs(abs(turn) - 2*pi) <= turning_tol` (657), so a figure-8 …  
→ *Fix:* Declare topology per generator (closed/open, crossings allowed given vertical clearance). Add optional generator output channels (z and generic per-point float attributes) carried through the resamplers (same idx/frac lerp) and relax (xy-only, attributes re-interpolated by arc fraction). Add a 'generator' z-profile …

**A21. Generator output contract is fixed (vec2f, exactly num_points, int valid) and its implicit invariants are undocumented for plugin authors, with no conformance kit** — *api-surface* · `track_gen/_src/generator_registry.py:27`, `track_gen/_src/warp_pipeline.py:1553`, `track_gen/_src/warp_pipeline.py:395`  
- The spec's generate signature hard-wires `out_centerline` [E*num_points] vec2f and `out_valid_wp` [E] int32 (generator_registry.py:27-29; allocated at warp_pipeline.py:1553-1554). - Implicit invariants a plugin must satisfy: - Every one of the num_points points must be finite. _cs_scan_k is not NaN-aware: `if not (total > 0.0): count[e] = 0` (395-397) silently turns any NaN-padded or variable-length output into an …  
→ *Fix:* - Introduce a declared output schema (GeneratorOutputs with required centerline/count and optional channels such as z, half_width, attributes), and let the orchestrator allocate from the spec's declaration. - Make the downstream resampler accept a per-env count, or document 'finite, fixed N' explicitly. - State the …

**A22. Track and GateSequence have fixed schemas; stages and generators cannot attach per-point or per-env attributes, and Track conflates centerline with road band** — *extensibility* · `track_gen/_src/types.py:1318`, `track_gen/_src/types.py:1329`, `track_gen/_src/types.py:1202`  
Track has exactly 10 fields (types.py:1318-1327), and clone() enumerates them explicitly, returning a base `Track(...)` (1329-1347). GateSequence is the same (1202-1224). Allocation lives inside warp_pipeline._inflate_warp_alloc (1536-1547). Width is a single host scalar, `hw = float(config.half_width)` (warp_pipeline.py:1776), with no width field on Track ('Half-width is not stored', types.py:1233). CourseLine …  
→ *Fix:* Decompose the results into a centerline core (center, tangent, arclen, length, count, valid, closed) and optional typed layers (band: outer, inner, normal, width; frames; elevation). Add a registered `attributes: dict[str, wp.array]` channel, with each stage declaring the attributes it allocates and writes. Make …

**A23. Relaxation and elevation hard-code a closed-loop, planar (vec2f) topology** — *extensibility* · `track_gen/_src/warp_relax.py:96`, `track_gen/_src/warp_relax.py:144`, `track_gen/_src/warp_relax.py:323`  
Every relax kernel indexes circularly: the perimeter includes the closing chord (warp_relax.py:96), neighbours use `(i + 1) % ne` / `(i + ne - 1) % ne` (144-145, 258-259, 323-324), and separation uses circular index distance `wp.min(dd, ne - dd)` (129, 198). All buffers are wp.vec2f (77, 102, 285). Every z-profile assumes a loop: the closing chord is added to perim (warp_zprofile.py:58), random_walk subtracts a …  
→ *Fix:* Make topology an explicit per-batch contract: a `closed: bool` (or per-env flag) consumed by relax and profile kernels, with clamped instead of wrapped neighbours and a non-bridged walk. Plan for 3D-aware separation (separation in xyz, or z-aware exclusion) as a separate relax solver behind the registry above. …

**A24. _Scratch.__getattr__ resolves inflate's buffers through the generator's scratch first, so plugin attributes named w/z/kappa/area_a/area_b silently hijack them** — *extensibility* · `track_gen/_src/warp_pipeline.py:1486`, `track_gen/_src/warp_pipeline.py:1854`, `track_gen/_src/warp_pipeline.py:1875`  
`for group in (self.gen, self.relax, self.inflate): ... if name in __slots__ or __dict__: return getattr(group, name)` (1480-1495). inflate_warp reads scratch.w (1854, 1869), scratch.kappa (1875), scratch.area_a/area_b (1885) and scratch.z (1919-1922) through this fallthrough. Bezier's GenScratch avoids collisions only by prefixing every slot with gen_. Voronoi/Checkpoint/RepulsiveScratch use bare names (for example …  
→ *Fix:* Remove the flat-name fallthrough (have inflate_warp read scratch.inflate.w and similar explicitly), or at least never search .gen. Document the generator scratch as opaque.

**A25. Generators must import private warp_pipeline internals and follow an undocumented sync rule** — *extensibility* · `track_gen/_src/warp_pipeline.py:42`, `track_gen/_src/warp_pipeline.py:172`, `track_gen/_src/warp_pipeline.py:774`  
Every shipped generator does `from . import warp_pipeline as _pipe` and calls `_pipe._init`, `_pipe._sync`, `_pipe._fill_i32_k`, `_pipe._arc_resample_inplace`, `_pipe._arc_resample_selected_inplace`, `_pipe.self_intersections_inplace` and `_pipe.GenScratch` (warp_generate.py:732). warp_generate_checkpoint.py:348-351 uses the wp.func `_pipe._ccw` inside a kernel. track_gen/__init__.py exports nothing from …  
→ *Fix:* Publish a stable, sync-free primitives module (e.g. `track_gen.ops`) covering fills, NaN-aware arc-length resample (plus a selected variant), self-intersection counts, segment predicates as wp.func, and documented shape contracts. Port the built-ins to it and reference only public names in the generator docs.

**A26. N_max truncation is silent on CUDA; the only diagnostic is a host readback that never runs on the graph path** — *correctness* · `track_gen/_src/warp_pipeline.py:401`, `track_gen/_src/warp_pipeline.py:1127`, `track_gen/_src/warp_pipeline.py:1130`  
`_cs_scan_k` clamps `count[e] = wp.min(wp.max(k, 1), n_max)` (401) and records nothing. The warning does `int(_np.max(count_wp.numpy()))` (1130) only when `not _CAPTURING` (1127). TrackGenerator sets `_CAPTURING = True` for the three warm-up runs as well as the capture (track_generator.py:193-197), so on CUDA with a capturable generator the warning can never fire, not even once. `_validity_k` has no truncation term …  
→ *Fix:* In `_cs_scan_k`, write the unclamped k (or a truncated bit) into an [E] buffer before clamping. AND `!truncated` into `_validity_k`, or expose it as `Track.truncated`, mirroring CheckpointSampler.truncated. Delete the host readback.

**A27. capturable is all-or-nothing for the whole pipeline: one eager generator makes the capturable tail eager too, including its readback** — *graphability* · `track_gen/_src/track_generator.py:88`, `track_gen/_src/track_generator.py:189`, `track_gen/_src/track_generator.py:213`  
TrackGenerator branches only on `self._capturable = self._generator_spec.capturable` (88). When it is False, `self._run()` executes the whole _run_pipeline eagerly (213): generation, resample_constant_spacing (with a `count_wp.numpy()` readback at warp_pipeline.py:1127-1130 because _CAPTURING is False), band/L0 and XPBD (syncs), and inflate (sync). GeneratorSpec.capturable is a static bool (generator_registry.py:43).  
→ *Fix:* Capture per stage. Build Graph A for the tail (resample, relax, inflate), which always reads gen_centerline and gen_valid. Run the generator either inside the same graph (when capturable) or eagerly before replaying the tail graph. Make capturability a function `capturable(config) -> bool` on the spec, or a per-stage …

**A28. GateGeneratorSpec has no `capturable` opt-out: every gate generator is force-captured on CUDA, so host-driven generators either fail or silently freeze capture-time decisions** — *graphability* · `track_gen/_src/gate_generator_registry.py:14`, `track_gen/_src/gate_generator.py:158`, `track_gen/_src/generator_registry.py:41`  
GateGeneratorSpec has fields (name, alloc_scratch, generate, max_gates, supported_orderings) only (registry:14-21). By contrast GeneratorSpec has `capturable: bool = True` (generator_registry.py:41), and TrackGenerator branches on it (track_generator.py:88, 189-213). GateGenerator always captures on CUDA (gate_generator.py:158-187). The plugin's Python `generate` is executed 4 times on the first call (3 warm-ups …  
→ *Fix:* Add `capturable: bool = True` to GateGeneratorSpec, mirror TrackGenerator's eager path, and document the capture contract for gate plugins (Python side effects run only during warm-up and capture). Longer term, offer wp.capture_while/capture_if helpers so iterative generators can stay capturable.

**A29. Ordering is delegated to each generator through private helpers and is not enforced; the ordering set is closed** — *extensibility* · `track_gen/_src/warp_gate.py:599`, `track_gen/_src/warp_gate.py:625`, `track_gen/_src/warp_gate.py:538`  
Every built-in generator calls `warp_gate.finish_ordered_gates(...)` itself, with scratch from the private `warp_gate.alloc_order_scratch`. Most also call the underscore kernel `warp_gate._fill_i32_k` (polar_gates:26-30, 57, 71; voronoi_gates:28-33, 58, 82; checkpoint_gates:25-29, 49, 64; warp_generate_gates.py:20-24, 40, 59). The pipeline only validates `config.gate_ordering in supported_orderings` …  
→ *Fix:* Make ordering a pipeline-owned stage. The generator emits raw anchors and counts into a staging buffer of capacity spec.capacity(config), and the pipeline applies the ordering and optional normalization (declared via spec flags). Back orderings with a registry (name to kernel launcher) and validate gate_ordering …

**A30. GateGeneratorSpec contract is undocumented and inconsistent with the design spec; max_gates is overloaded** — *docs* · `track_gen/_src/gate_generator_registry.py:15`, `track_gen/_src/gate_generator.py:71`, `docs/superpowers/specs/2026-06-23-gate-sequence-generation-design.md:53`  
The spec docstring is just 'One registered native gate generator.' (15). The design spec says `generate(seeds_wp, config, out_gate_sequence, scratch)` (spec:53), but the implementation passes the private `_GateStaging` (warp_gate.py:842-843). spec.max_gates(config) is used both as 'can produce at most' for the min_gates check (gate_generator.py:72-77) and as 'required capacity' (78-83); no minimum producible count …  
→ *Fix:* Define a typed Protocol plus a 'Writing a gate generator' doc covering: buffers, stride, count and NaN rules, extent and units, closed vs open, determinism, the capture rules (no sync, no alloc, Python side effects frozen after capture), and capability flags. Rename max_gates to capacity and add min_count.

**A31. Hard-coded stage sequence and closed-set dispatch for z-profile, alignment and validity; stages cannot be swapped or inserted** — *extensibility* · `track_gen/_src/warp_gate.py:820`, `track_gen/_src/warp_gate.py:730`, `track_gen/_src/warp_gate.py:434`  
_run_gate_pipeline is a fixed script: generate, relax_gate_spheres, gate_cum_perim, apply_z_profile, lift, tangents, finalize (820-864). The z-profile is a closed set (types.py:24-29) with if/elif dispatch (warp_zprofile.py:203-222). gate_align is a closed set (types.py:1141-1145) collapsed to `int(config.gate_align == "full_tangent")` (warp_gate.py:730). Validity is a single kernel with every predicate inlined …  
→ *Fix:* Model the pipeline as an ordered list of Stage objects, each with alloc(config) and launch(ctx). Back each stage kind with a registry: spacing, z-profile, tangent, frame, validity predicates. Compose validity as an AND of predicate kernels writing into valid (e.g. atomic_min). Keep the current sequence as the default …

**A32. Reusable building blocks are private, tied to config field names, and copy-pasted across generators** — *modularity* · `track_gen/_src/warp_generate.py:83`, `track_gen/_src/warp_generate.py:135`, `track_gen/_src/warp_generate_hull.py:56`  
Duplicated kernels: - Grid point sampler with dedup: _corner_sample_k (warp_generate.py:83-132), _corner_sample_style_k (135-171), hull _point_sample_k (hull:56-90). They differ only in salt and scale source. - Count sampler: warp_generate.py:33-54 vs hull:39-52. - Centroid angle sort: _ccw_sort_k (warp_generate.py:174-217) vs _angle_sort_k (hull:94-135). Identical. - Catmull-Rom: inline in hull _catmull_rom_k …  
→ *Fix:* - Create a public `track_gen.primitives` (or `track_gen.generators.blocks`) layer with explicit-argument stage functions and no config. Examples: - sample_grid_points(seeds, salt, P, cell_size, scale_or_array, out, scratch) - sample_counts(seeds, salt, lo, hi, out) - angle_sort(points, count, P, out, scratch, …

**A33. Course hard-codes two generator families and a fixed set of runtime utilities** — *extensibility* · `track_gen/_src/course.py:144`, `track_gen/_src/course.py:150`, `track_gen/_src/course.py:330`  
Hard-coding at each stage: - Config: CourseConfig accepts only `self.mode not in ("track", "gates")` → raise (course.py:144-146), and requires `isinstance(self.gen, TrackGenConfig)` or `GateGenConfig` (150-153, 200-203). - Construction: __init__ picks `TrackGenerator(config.gen, self.rng)` or `GateGenerator(...)` by mode (330-333). - Sub-tools: _build_subtools is an if/else on mode that names CheckpointSampler, …  
→ *Fix:* Make Course a composition root over public protocols: - A CourseGenerator protocol (num_envs, device, result, enqueue/generate, optional graph). - A CourseComponent protocol (build(result, ctx), refresh(), bind(inputs), step() -> output, reset(mask)). - An ordered list of components; Graph B captures `for c in …

**A34. No common utility protocol; Course hard-codes the tool set with isinstance dispatch and private-attribute access** — *extensibility* · `track_gen/_src/course.py:438`, `track_gen/_src/course.py:575`, `track_gen/_src/course.py:287`  
Lifecycle method names differ per class: `bind_inputs(position, orientation, half_extents)` on the box checkers, `bind_inputs(position)` plus `bind_window(next_cp)` on FrameChecker, `bind(position)` on the tracker and localizer. The refresh step is `bake()` on SDF/heightfield, `sample()` on the samplers, `refresh()` on CourseLine. Course therefore dispatches by type: `if isinstance(self.collision, FrameChecker): ... …  
→ *Fix:* Define a public `RuntimeTool` Protocol, e.g. `on_generate()` (capture-safe refresh), `bind(**buffers)`, `step() -> result`, `reset(mask)`, plus `needs` and `outputs` metadata. Course would then hold an ordered list or registry of tools and capture `on_generate` for all of them in its refresh graph. CourseConfig would …

**A35. Course cannot be given a generator, RNG or result source; it always constructs its own** — *extensibility* · `track_gen/_src/course.py:317`, `track_gen/_src/course.py:328`, `track_gen/_src/course.py:330`  
Course.__init__(self, config) takes only a CourseConfig. It builds `PerEnvSeededRNG(seeds=config.seeds, ...)` (328-329) and `TrackGenerator(config.gen, self.rng)` or `GateGenerator(config.gen, self.rng)` (330-333). It relies on an implicit, undocumented generator protocol: generate() returns the same stable-pointer Track or GateSequence, and config.gen has num_envs and device (320, 324).  
→ *Fix:* Accept generator= (any object satisfying a documented, public CourseGenerator protocol) and rng= in __init__, and keep config-driven construction as a factory (Course.from_config). Publish the protocol in a public module with a conformance test.

**A36. Track used as a universal representation: CourseLine fakes band fields with NaN, and consumers do not check which capabilities they need** — *modularity* · `track_gen/_src/course_line.py:132`, `track_gen/_src/heightfield.py:144`, `track_gen/_src/course.py:211`  
CourseLine builds `Track(outer=wp.full(n, nan3), ..., inner=wp.full(n, nan3), normal=wp.full(n, nan3), winding=zeros)` (course_line.py:132-143; contract stated at lines 3-6). HeightFieldBaker takes its grid AABB from track.outer via _track_aabb_k (heightfield.py:144-147), so it only makes sense for tracks with a road band. Course blocks the bad combination only with a config rule (course.py:211-213). No consumer …  
→ *Fix:* Split the contract into small capability protocols: - Centerline: center, tangent, arclen, length, count, valid. - Band: inner, outer. - Frames: normal, orientation. Each consumer validates the capability it needs at construction. Allow an extensible per-point attribute map on results.

**A37. Closed-loop topology is hard-coded in every utility kernel, and Track cannot express open courses** — *extensibility* · `track_gen/_src/types.py:1318`, `track_gen/_src/collision.py:158`, `track_gen/_src/collision_sdf.py:103`  
The segments and SDF kernels wrap the boundary with `i2 = i + 1; if i2 == m: i2 = 0` (collision.py:158-160; collision_sdf.py:103-105). The scan adds the closing edge: `dc = points[base] - prev # closing edge back to point 0` (props.py:132-133), and segment props close the ring (230-231). The localizer wraps segments and `s` at the seam (localize.py:146-149, 189-190). speed_profile runs 'Two wrap laps so the closed …  
→ *Fix:* Add an explicit per-batch or per-env topology (`closed: wp.array[int32]`, or a Track-level `closed: bool`) and pass it to every kernel: skip the closing segment, clamp rather than wrap `s`, finish instead of lap. Until then, validate and reject open input loudly. Document closedness as part of the Track contract that …

**A38. No regeneration signal; stateful and baked utilities depend on caller-ordered reset/bake/sample after each generate()** — *extensibility* · `docs/utilities/overview.rst:67`, `track_gen/_src/progress.py:19`, `track_gen/_src/localize.py:211`  
The docs table lists per-tool manual steps after each regeneration (overview.rst:67-90). ProgressTracker says 'Callers MUST reset after regenerating' (progress.py:19-22), TrackLocalizer says 'callers MUST reset() all envs after regenerating' (localize.py:211-215), and SDF 'requires bake() after each generate()' (collision.py:268-271). The kernel only defensively clamps a stale target (progress.py:140-144).  
→ *Fix:* Add a device-side per-env `epoch` [E] int32 to Track/GateSequence, incremented by the pipeline. Stateful kernels compare a stored epoch and self-reset on the device, which is graph-safe and needs no host involvement. Also let generators accept registered post-generate hooks (the RuntimeTool.on_generate protocol) that …

**A39. No reusable or shipped conformance suite for generator plugins; contract tests are duplicated per generator and read private scratch** — *testing* · `tests/test_generate_polar.py:42`, `tests/test_generate_checkpoint.py:56`, `tests/test_generate_hull.py:39`  
- Each generator file re-implements the same contract checks (registered, finite, [E*num_points] shape, closed loop, deterministic, diverse, NaN tail) by reading `gen._scratch.gen_centerline`, e.g. test_generate_polar.py:42-112, test_generate_checkpoint.py:56-102 and test_generate_repulsive.py:224-252 ('full per-generator contract'). - Only two tests iterate the registry. test_warp_graph.py:40-42 is CUDA-only, …  
→ *Fix:* - Ship `track_gen.testing` with a parametrizable `GeneratorConformance` class or pytest plugin that takes a spec name (default: every name in registry.available()). It should check: - shape and dtype of the raw output - finite output for valid envs; NaN padding past count[e] on the final Track - closed loop - …

**A40. Caller-side capture coverage gaps: gates-mode step, Course.reset, reseeding, gate generators, external and nested captures** — *testing* · `tests/test_course_cuda.py:80`, `tests/test_course_cuda.py:91`, `tests/test_gate_generator.py:370`  
Coverage today: - Caller capture is tested only for single utilities and for Course.step in track mode (test_course_cuda.py:80-122). Inside the capture, reset is called eagerly (line 91). Untested under capture: - Course.step in gates mode: TrackLocalizer plus FrameChecker or DiscChecker (course.py:592-604). - Course.reset(mask) at facade level. - PerEnvSeededRNG.set_seeds_warp (rng_utils.py:78-93). - …  
→ *Fix:* - Add these cases to a parametrized capture suite: {track, gates} x {step, reset} recorded together in one graph. - Add reseed-then-replay vs a fresh generator, through the public rng API only. - Capture all gate generators with poisoned replay. - Add a torch.cuda.graph capture using …

**A41. No test CI; CUDA capture tests never run automatically and silently skip when torch is a CPU-only build** — *testing* · `.github/workflows/docs.yml:1`, `tests/test_warp_graph.py:25`, `tests/test_collision_cuda_graph.py:15`  
- The only workflow builds and deploys docs (.github/workflows/docs.yml:14-35). No pytest job exists, for either the fast lane or CUDA. - Every CUDA test gates on `torch.cuda.is_available()`: as pytestmark skipif (test_warp_graph.py:25, test_collision_cuda_graph.py:15, test_course_cuda.py:10), as a module-level skip (test_warp_relax.py:6-7), or as device lists (`DEVS = ["cpu"] + (["cuda"] if …  
→ *Fix:* - Add a test workflow with a CPU fast lane on every PR and a GPU lane on a self-hosted or NVIDIA runner that runs `-m cuda`. - Fail that lane if CUDA tests were skipped (e.g. a conftest fixture that asserts `wp.is_cuda_available()` when an env var such as TRACKGEN_REQUIRE_CUDA=1 is set). - Replace …

**A42. No public 'centerline -> Track' ingestion entry point; mid-pipeline entry needs private stage functions with hidden preconditions** — *api-surface* · `track_gen/__init__.py:10`, `track_gen/_src/warp_pipeline.py:1066`, `track_gen/_src/warp_pipeline.py:395`  
Public exports (track_gen/__init__.py:10-38, docs/reference/api.rst) contain no stage function. Turning an external centerline into a Track takes four steps, all private: 1. arc_length_resample_inplace (warp_pipeline.py:801-829) with 5 caller-allocated scratch buffers, needed for variable-length input. 2. resample_constant_spacing (1066-1141). Its input must be fully real with a fixed N. _cs_scan_k sums all N slots, …  
→ *Fix:* Add a public ingestion API with two parts. - An offline function: track_gen.ingest.track_from_centerlines(points, lengths=None, *, spacing, half_width or per-point width, z=None, closed=True, relax=None, validity=ValidityConfig, n_max='auto', device) -> Track, plus GateSequence.from_gates(positions, yaw, half_size, …

**A43. Mixed-generator and per-env-config batches are unsupported: no caller-owned or env-slice output, and one generator per pipeline, graph and Course** — *extensibility* · `track_gen/_src/track_generator.py:94`, `track_gen/_src/gate_generator.py:92`, `track_gen/_src/warp_pipeline.py:922`  
TrackGenerator.__init__ always allocates its output with `self._track, self._scratch = warp_pipeline._inflate_warp_alloc(config, ...)` (track_generator.py:94-96) and its own `_seed_buf` (line 102). The constructor has no out=, view or env_offset parameter. generate() rejects env-id sequences (lines 155-168). GateGenerator does the same with `_gate_warp_alloc` (gate_generator.py:92-104). Inside the orchestrators, …  
→ *Fix:* 1. Accept caller-provided outputs: TrackGenerator(config, rng, out: Track | None = None) and GateGenerator(..., out: GateSequence | None = None). Validate shape, dtype and device, and explicitly allow contiguous slice views. Add public Track.alloc(E, n_max, device) and Track.view(e0, e1) helpers, with GateSequence …

**A44. Utilities infer batch layout from one field's length with only a divisibility check; a hand-assembled (mixed) Track is never validated, and mismatched strides silently mis-index** — *correctness* · `track_gen/_src/props.py:308`, `track_gen/_src/checkpoints.py:229`, `track_gen/_src/collision.py:318`  
Every Track consumer computes `E = int(track.count.shape[0])` and `stride = int(track.outer.shape[0])` (props.py:308-309, checkpoints.py:229-230, collision.py:318-319) or uses `track.center` (localize.py:255-256, heightfield.py:119-120). It then checks only `stride % E != 0` and sets `n_max = stride // E`. No other Track field (inner, center, tangent, arclen, length, valid, winding) is checked for shape, dtype or …  
→ *Fix:* Record the layout explicitly: add n_max to Track (max_gates to GateSequence), or a small layout object, set by the producer. Add Track.validate() / GateSequence.validate(), built on _check_arr, that checks every field's shape [E*n_max] or [E], dtype and device. Call it in every utility constructor and reject a …

**A45. Undocumented warm-up/record/replay model runs a plugin's generate 4x on the first CUDA call (1x on CPU); stateful generators diverge silently** — *graphability* · `track_gen/_src/track_generator.py:196`, `track_gen/_src/track_generator.py:200`, `track_gen/_src/track_generator.py:207`  
First CUDA generate(): `for _ in range(3): self._run()` (track_generator.py:196-197) runs the pipeline eagerly 3 times. `with wp.ScopedCapture(device=dev) as cap: self._run()` (200-201) records once without executing. `wp.capture_launch(self._graph)` (207) executes once. Net: the Python body of spec.generate runs 4 times and its device kernels execute 4 times on call 1. Later CUDA calls run Python 0 times and the …  
→ *Fix:* 1. Document the execution model in writing-a-generator.rst and the GeneratorSpec/GateGeneratorSpec docstrings: how many times generate runs per call by device, that the Python body never runs on replay, that scalar kernel arguments and Python state freeze at record, and that buffer contents persist across replays. 2. …

**A46. No supported per-call device input channel for generators besides seeds; per-env difficulty or target arrays must be smuggled through config attributes, global closures or private scratch** — *extensibility* · `track_gen/_src/warp_pipeline.py:889`, `track_gen/_src/warp_pipeline.py:923`, `track_gen/_src/track_generator.py:120`  
`TrackGenerator.generate(self, num_or_ids=None)` accepts no data (track_generator.py:120). `_run_pipeline(config, seed_buf_wp, out, scratch, generator_spec)` (warp_pipeline.py:889-895) calls `generate(seed_buf_wp, config, out_centerline=..., out_valid_wp=..., scratch=gen)` with a fixed keyword set (922-926). The gate pipeline calls `generator_spec.generate(seed_buf_wp, config, staging, gen_scratch)` positionally …  
→ *Fix:* 1. Add an explicit per-call input path. Either let GeneratorSpec declare named device inputs ([E]-shaped arrays allocated by the orchestrator outside capture, with stable pointers), or pass a typed `inputs` object to generate. 2. Expose these inputs on TrackGenerator, GateGenerator and Course: fill the persistent …

**A47. N_max-truncated tracks pass the validity gate: valid=1 ships a silently clipped track, and Track has no truncation flag** — *correctness* · `track_gen/_src/warp_pipeline.py:401`, `track_gen/_src/warp_pipeline.py:416`, `track_gen/_src/warp_relax.py:97`  
_cs_scan_k computes the full dense perimeter (warp_pipeline.py:389-395) and the unclamped k, then stores only `count[e] = wp.min(wp.max(k, 1), n_max)` (:401). The true perimeter and k are discarded. _cs_lookup_k places points at `target = wp.float32(k) * spacing` (:416) up to (n_max-1)*spacing, so one edge closes the loop with a chord across the dropped arc. XPBD then treats the truncated polygon as ground truth. …  
→ *Fix:* Keep the truncation decision on device and fail closed. In _cs_scan_k, write truncated[e] = int(k > n_max) and required_count[e] = k from the unclamped k. This is zero-cost and graph-safe. Pass truncated into _validity_k as a `trunc_ok` term, with a config policy on_truncate='invalidate' (default) or 'flag'. Add …

**A48. Invalid envs leave the pipeline with finite geometry and count>=3, so nearly every utility silently computes plausible garbage for them** — *correctness* · `track_gen/_src/warp_pipeline.py:709`, `track_gen/_src/warp_pipeline.py:1931`, `track_gen/_src/warp_generate.py:708`  
Every generator fills gen_valid=1 (warp_generate.py:708; hull:458; checkpoint:598; polar:216; voronoi:411; repulsive:860). _validity_k only writes out.valid ('out[e] = gv & turn_ok & ...', warp_pipeline.py:712). The lift kernels at :1931/:1940 still write outer/center/inner/tangent/normal for all envs, and count is set at resample before XPBD (:931-935, _cs_scan_k:376-399). The per-step kernels key their degenerate …  
→ *Fix:* Pick one device-side invalid-env contract and enforce it in the pipeline. Either (A) add a final 'poison' kernel to _run_pipeline/_run_gate_pipeline that writes count=0 and NaN geometry where valid==0, so every utility's existing count<3 path fires uniformly (this matches the original oracle semantics), or (B) keep …

**A49. Utilities disagree on invalid envs (ignore / NaN / no-hit), and NaN leaks into Course.step() frames, heightfields and gate rewards** — *correctness* · `track_gen/_src/heightfield.py:70`, `track_gen/_src/collision_frames.py:69`, `track_gen/_src/course_line.py:49`  
The conventions differ: HeightFieldBaker is the only Track consumer reading valid, and 'if m < 3 or valid[e] == 0: height[t] = wp.nan' (heightfield.py:70-72). FrameChecker returns hit=0, depth=0 for gate_valid==0 (collision_frames.py:66-70), which is permissive. CollisionChecker returns oob=1 only for count<3 (conservative). DiscChecker has no validity input, and Course constructs it without count …  
→ *Fix:* Define a single sentinel table for valid==0 (for example oob=1 and hit=0 with an explicit invalid flag, NaN geometry everywhere, progress frozen with dist_to_next=0 rather than NaN) and apply it in every kernel. Pass gate count and valid to the DiscChecker built by Course. Add a 'valid' alias field to StepResult and …

**A50. Empirically, every shipped generator edited 5-10 shared files, and the spec's 'collide only on the single registry line' model fails: a 4-month-old generator now conflicts in all 5 shared code/test files it touches** — *modularity* · `docs/superpowers/specs/2026-06-21-pluggable-generator-framework-design.md:200`, `track_gen/_src/generator_registry.py:57`, `track_gen/_src/types.py:553`  
The spec (:200-202) promises that 'a method = one new module + one GeneratorSpec + one registry line + its own tests, N agents collide only on the single registry line'. The git show --stat table in notes shows otherwise. types.py and generator_registry.py were edited by all six generators. hull: types, registry, param_explorer, test_types, test_param_explorer, plus framework fix 66a0401 touching …  
→ *Fix:* Move everything a generator currently adds to shared files into a per-generator descriptor carried by the spec: params dataclass and validation, explorer UI schema, fast/test config overrides, acceptance thresholds, gallery label, capturability. Discover descriptors via entry points (or an explicit `register_module` …


## Low (88)

**A51. Warmup x3 and the default force_module_load add first-call cost and touch unrelated modules** — *performance* · `track_gen/_src/track_generator.py:196`, `track_gen/_src/gate_generator.py:175`  
→ *Fix:* Warm up once (or use wp.load_module for the relevant modules), then capture with force_module_load=False. Make the warmup count internal and documented.

**A52. Utilities and Course have no warm-up API; modules load lazily, possibly inside the caller's capture** — *graphability* · `track_gen/_src/course.py:568`, `track_gen/_src/course.py:599`, `track_gen/_src/collision.py:459`  
→ *Fix:* Give every component (utilities, Course, generators) a warmup(device) that calls wp.load_module on its modules or does one eager launch. Have Course.__init__ or warmup() pre-load all component modules. Reduce generator …

**A53. CUDA-only capture gating via 'cuda' in str(device) blocks Warp CPU graph capture (APIC)** — *graphability* · `track_gen/_src/track_generator.py:172`, `track_gen/_src/gate_generator.py:158`, `track_gen/_src/course.py:325`  
→ *Fix:* Decide capture from capabilities (wp.get_device(dev).is_cuda or explicit opt-in) instead of string matching, and allow CPU capture where Warp supports it.

**A54. Whole-batch-only regeneration; no device-masked per-env regeneration** — *graphability* · `track_gen/_src/track_generator.py:155`, `track_gen/_src/gate_generator.py:144`, `track_gen/_src/course.py:18`  
→ *Fix:* Accept a bound device-side [E] int32 regeneration mask. Either every stage gates its writes on it, or the pipeline generates into scratch and conditionally commits per env. The launch topology stays fixed and …

**A55. "cuda" device alias not canonicalized in generators; config.device string re-resolved per call from the calling thread's CUDA context** — *correctness* · `track_gen/_src/track_generator.py:171`, `track_gen/_src/gate_generator.py:157`, `track_gen/_src/warp_zprofile.py:201`  
→ *Fix:* Resolve the device once at construction (`self._device = wp.get_device(config.device)`) and store the Device object in a frozen copy of the config. Derive every launch's device from the owned buffers or the explicit …

**A56. Four redundant lazy wp.init() guards called throughout the hot path** — *threading-state* · `track_gen/_src/runtime.py:37`, `track_gen/_src/warp_pipeline.py:34`, `track_gen/_src/warp_gate.py:19`  
→ *Fix:* Call wp.init() once in track_gen/__init__.py, or in constructors only. Delete the per-module flags.

**A57. GateGenConfig accepts min_point_distance > 0.5, giving an integer modulo by zero in the sampler kernel** — *correctness* · `track_gen/_src/types.py:1088`, `track_gen/_src/types.py:684`, `track_gen/_src/types.py:691`  
→ *Fix:* Validate the shared sampler params in one place, a shared params dataclass with a single validator, enforcing (0, 0.5]. Add a test that sweeps validation parity between the configs.

**A58. __post_init__ resolves spacing in place: dataclasses.replace() keeps a stale spacing and frozen configs are impossible** — *correctness* · `track_gen/_src/types.py:851`, `benchmarks/compare_generators.py:36`  
→ *Fix:* Keep `spacing: float | None` unresolved and expose a derived `effective_spacing` property (or resolve into a separate private frozen 'resolved config' at generator construction). Then mark the configs `frozen=True`.

**A59. Config ergonomics for a CUDA-first library: device defaults to cpu, configs are unhashable, version is duplicated** — *api-surface* · `track_gen/_src/types.py:539`, `track_gen/_src/types.py:1010`, `track_gen/_version.py:1`  
→ *Fix:* Default to `wp.get_preferred_device()` (or require the device explicitly), use frozen configs, and single-source the version (setuptools dynamic version or importlib.metadata).

**A60. PerEnvSeededRNG aliases the caller's seed array (and writes back into it), coupling independent instances; Course reseed allocates per call** — *threading-state* · `track_gen/_src/rng_utils.py:46`, `track_gen/_src/rng_kernels.py:30`, `track_gen/_src/course.py:328`  
→ *Fix:* Copy user seed arrays into an RNG-owned buffer at construction. Add a device-side, capture-safe reseed or advance API, for example a kernel `seeds[e] = hash(seeds[e], episode_counter[0])` reading a device counter, so …

**A61. set_seeds_warp leaves _new_states stale, so a later partial-ids sample silently reverts the reseed** — *correctness* · `track_gen/_src/rng_utils.py:78`, `track_gen/_src/rng_utils.py:58`, `track_gen/_src/rng_kernels.py:28`  
→ *Fix:* Write new_states[ids[tid]] in set_states as well, or (better) drop the whole-array copy and scatter only the touched ids back from new_states to states. Add a reseed-then-partial-sample test.

**A62. PerEnvSeededRNG is a required public constructor argument but the core library consumes only .seeds_warp (two parallel RNG systems)** — *api-surface* · `track_gen/__init__.py:13`, `track_gen/_src/track_generator.py:64`, `track_gen/_src/track_generator.py:188`  
→ *Fix:* Define a minimal SeedSource protocol (seeds_warp plus a capturable advance/reseed(mask)) that generators accept. Have PerEnvSeededRNG implement it, and move distribution sampling into a separate optional utility (or …

**A63. Carry-over RNG defects from the 2026-06-27 review still open** — *correctness* · `track_gen/_src/rng_kernels.py:132`, `track_gen/_src/rng_kernels.py:1127`, `track_gen/_src/rng_utils.py:41`  
→ *Fix:* Treat shape[0]==1 like any other shape, add the else/TypeError branch, validate seeds (shape [num_envs], int32) in the constructor, sample quaternions via Shoemake's method, and clamp band to (count-1)//2.

**A64. PerEnvSeededRNG is not thread-safe, and direct reseeds bypass the capture lock** — *threading-state* · `track_gen/_src/rng_kernels.py:136`, `track_gen/_src/course.py:474`, `track_gen/_src/runtime.py:35`  
→ *Fix:* Hold a per-instance lock around launch+copy, or fuse the state update into the sampling kernel (write states directly after reading into a register, one thread per env). Document that captures must not overlap direct …

**A65. Experimental modules have no path to the registry, ship in the wheel, and include unimportable or torch-dependent code** — *extensibility* · `track_gen/_experimental/fourier.py:14`, `track_gen/_experimental/fourier.py:26`, `track_gen/_experimental/fourier.py:72`  
→ *Fix:* Move the prototypes and benchmarks out of the package (to benchmarks/ or examples/), or into an optional `track_gen.contrib` extra that declares torch and matplotlib. Provide a documented HostGeneratorAdapter that wraps …

**A66. Asymmetric, untyped spec designs (capturable vs max_gates/orderings) and inconsistent call conventions** — *api-surface* · `track_gen/_src/generator_registry.py:18`, `track_gen/_src/gate_generator_registry.py:13`, `track_gen/_src/warp_pipeline.py:922`  
→ *Fix:* Define both specs as typing.Protocol classes with documented, typed signatures and shared optional hooks: `capture_mode`, `validate`, `capacity`, `params_cls`, `output_layout`. Validate specs in register(), and call …

**A67. Registry duplicate detection keys on callables' __module__; silent overwrite and skip cases** — *api-surface* · `track_gen/_src/gate_generator_registry.py:35`, `track_gen/_src/gate_generator_registry.py:63`  
→ *Fix:* Track an explicit `owner`/`source` string on the spec (or on the register() call), require `override=True` to replace, add unregister(), and raise if a listed module lacks register_specs.

**A68. Duck-typed config access with getattr defaults that diverge from the dataclass defaults** — *api-surface* · `track_gen/_src/warp_relax.py:391`, `track_gen/_src/warp_relax.py:413`, `track_gen/_src/warp_pipeline.py:1557`  
→ *Fix:* Have stages take their own typed params objects and validate the type at construction. Remove the getattr fallbacks, or source every default from the single dataclass definition.

**A69. Broadphase cache overflow is counted but never surfaced** — *correctness* · `track_gen/_src/warp_relax.py:213`, `track_gen/_src/warp_relax.py:439`, `track_gen/_src/warp_pipeline.py:1561`  
→ *Fix:* Record per-env overflow flags (sticky across refreshes) into a device diagnostics struct on Track, or fold them into validity. This stays capture-safe because it is device-side.

**A70. Device and batch size taken from config, and string device checks; a device-wide sync blocks foreign streams** — *graphability* · `track_gen/_src/warp_zprofile.py:199`, `track_gen/_src/warp_zprofile.py:354`, `track_gen/_src/warp_relax.py:399`  
→ *Fix:* Derive E and device from the arrays and validate them against each other. Use wp.get_device(dev).is_cuda. Remove the global syncs (see the first finding). Document ScopedStream usage for external streams.

**A71. Validity runs O(N^2) serial loops with one thread per env** — *performance* · `track_gen/_src/warp_pipeline.py:268`, `track_gen/_src/warp_pipeline.py:184`, `track_gen/_src/warp_pipeline.py:677`  
→ *Fix:* Parallelize per (env, i) with atomic_min for sep_min and kappa_max, reuse the by-i crossing kernel, and keep only the AND fusion serial.

**A72. Meaning of Track fields changes with z_profile; gate and track z profiles with the same name behave differently** — *docs* · `track_gen/_src/types.py:1266`, `track_gen/_src/types.py:1288`, `track_gen/_src/types.py:1251`  
→ *Fix:* Make the field semantics invariant (always 3D tangent and arclen; add an explicit plan_arclen if needed), unify the padding rule, and document the profile semantics per stage.

**A73. Bezier is privileged inside the shared orchestrator (GenScratch in warp_pipeline, lazy re-export, 'Bezier params')** — *modularity* · `track_gen/_src/warp_pipeline.py:1237`, `track_gen/_src/warp_pipeline.py:1961`, `track_gen/_src/warp_pipeline.py:1426`  
→ *Fix:* - Move GenScratch into warp_generate as BezierScratch, and type _Scratch.gen as an opaque protocol. - Delete the PEP 562 re-export and update tests to import warp_generate directly. - Rename the shared knobs generically …

**A74. _Scratch.__getattr__ recurses infinitely on copy or pickle** — *correctness* · `track_gen/_src/warp_pipeline.py:1486`  
→ *Fix:* Use object.__getattribute__ with try/except inside __getattr__, or drop the fall-through entirely (see the shadowing finding).

**A75. Inconsistent wrapper semantics, and docstrings that overstate capture safety** — *docs* · `track_gen/_src/warp_pipeline.py:5`, `track_gen/_src/warp_pipeline.py:900`, `track_gen/_src/warp_pipeline.py:801`  
→ *Fix:* Document the exact preconditions until the flag is removed, give every in-place op the same sync-free semantics, and choose one public name per op in a public module.

**A76. Allocate-if-None convenience paths inside pipeline primitives** — *graphability* · `track_gen/_src/warp_pipeline.py:1055`, `track_gen/_src/warp_pipeline.py:1108`, `track_gen/_src/warp_pipeline.py:1780`  
→ *Fix:* Separate the `alloc_*` helpers from strict `*_inplace` launchers that require all buffers. Raise when buffers are missing and `device.is_capturing`.

**A77. Public API inconsistencies: TrackGenerator skips the rng validation GateGenerator does, validates with assert, and the num_or_ids checks differ** — *api-surface* · `track_gen/_src/track_generator.py:72`, `track_gen/_src/track_generator.py:76`, `track_gen/_src/track_generator.py:158`  
→ *Fix:* Share one facade base class that does rng size, device and dtype validation and integer checks, and raises ValueError instead of using assert.

**A78. Shape contracts are inferred and unvalidated; owned-path contracts use assert** — *api-surface* · `track_gen/_src/warp_pipeline.py:982`, `track_gen/_src/warp_pipeline.py:1012`, `track_gen/_src/warp_pipeline.py:1209`  
→ *Fix:* Validate shape, dtype and device on the host (runtime._check_arr; metadata only, so capture-safe) and raise ValueError instead of using assert.

**A79. No input or shape validation; batch size E is inferred from different arrays in each wrapper** — *api-surface* · `track_gen/_src/warp_generate.py:511`, `track_gen/_src/warp_generate.py:557`, `track_gen/_src/warp_generate.py:643`  
→ *Fix:* - Validate once per (spec, scratch) at construction and at public block entry points, using runtime._check_arr. - Derive E from a single source (the scratch's recorded dims).

**A80. capturable=False is a binary escape hatch that holds the global lock for seconds and can never be graphed** — *graphability* · `track_gen/_src/track_generator.py:88`, `track_gen/_src/track_generator.py:209`, `track_gen/_src/warp_generate_repulsive.py:845`  
→ *Fix:* Replace `capturable: bool` with a capture-mode enum, e.g. {static, conditional (uses wp.capture_while/capture_if; needs a recent driver), host}. Port repulsive's stall-stop to a device-side flag driving wp.capture_while …

**A81. Repulsive per-env results depend on batch composition (frozen envs keep getting resampled until the global exit)** — *correctness* · `track_gen/_src/warp_generate_repulsive.py:835`, `track_gen/_src/warp_generate_repulsive.py:845`, `track_gen/_src/warp_generate_repulsive.py:857`  
→ *Fix:* Gate periodic and final resampling on the frozen mask (a selected-resample variant). Add an env-independence test (the same seed in batches of different sizes). Either support per-env-mask regeneration or fix the docs.

**A82. Serial per-env kernels and O(N^2) single-thread clip limit GPU use; eager launch overhead dominates small batches** — *performance* · `track_gen/_src/warp_generate_checkpoint.py:337`, `track_gen/_src/warp_generate_repulsive.py:325`, `track_gen/_src/warp_generate_repulsive.py:460`  
→ *Fix:* Use `break` once the first crossing is found, or reuse the parallel by-i crossing kernel to find the first (i,j) via atomic_min. Use Warp tile reductions (wp.tile_sum/tile_max) for per-env reductions, and consider tile …

**A83. Stale or incorrect docs about capture and the generator contract** — *docs* · `track_gen/_src/warp_generate_repulsive.py:10`, `track_gen/_src/warp_generate_repulsive.py:732`, `track_gen/_src/generator_registry.py:6`  
→ *Fix:* Correct the blocker list (readback, early exit, the syncing public resample, trailing sync), write the generator contract doc, generate the generator list from the registry, and document capture behaviour per spec.

**A84. Positional 6-tuple pipeline scratch indexed by position (scratch[5])** — *modularity* · `track_gen/_src/warp_gate.py:817`, `track_gen/_src/warp_gate.py:839`, `track_gen/_src/gate_generator.py:100`  
→ *Fix:* Replace it with a named dataclass (_GatePipelineScratch) with per-stage sub-scratch fields, or a dict keyed by stage name for pluggable stages.

**A85. gate_width documented as 'half-width' but used as full width** — *docs* · `track_gen/_src/types.py:896`, `track_gen/_src/warp_gate.py:407`  
→ *Fix:* Fix the docstring to 'full gate opening width'.

**A86. Gate collision is hard-wired to one gate geometry and to ProgressTracker internals** — *extensibility* · `track_gen/_src/collision_frames.py:85`, `track_gen/_src/collision_frames.py:75`, `track_gen/_src/collision_frames.py:149`  
→ *Fix:* Add per-gate shape parameters to GateSequence (shape id plus extents) or a gate-geometry protocol consumed by a generic frame checker. Make the window source an explicit [E] int32 'focus index' input rather than tracker …

**A87. RNG stream separation via `rand_init(seed*K + c)` collapses all streams at seed 0; salts are ad hoc with no registry** — *correctness* · `track_gen/_src/warp_generate.py:51`, `track_gen/_src/warp_generate.py:77`, `track_gen/_src/warp_generate.py:111`  
→ *Fix:* - Separate streams with a hash, not a multiply: use Warp's two-argument wp.rand_init(seed, offset) with a distinct stream id per (generator, stage), or a hash_combine(seed, stream_id) wp.func. - Publish a small …

**A88. Style sampling is bezier-only, hard-codes three knobs, needs duplicated kernels, and is silently ignored by hull/polar** — *extensibility* · `track_gen/_src/warp_generate.py:57`, `track_gen/_src/warp_generate.py:518`, `track_gen/_src/warp_generate.py:320`  
→ *Fix:* - Generalize into a per-env parameter sampler. Each generator declares sampleable params (name, default, range, salt); one kernel fills an [E, k] float array, and generators always read per-env arrays (a constant range …

**A89. Dead kernels, an inert min_angle knob, and inaccurate comments and docs** — *docs* · `track_gen/_src/warp_generate.py:404`, `track_gen/_src/warp_generate.py:484`, `track_gen/_src/types.py:95`  
→ *Fix:* - Move the oracle-parity kernels into the test tree, delete the unused multi-attempt kernels and the `attempt` parameter, and deprecate min_angle (or wire it up). - Rename the sort to angle_sort with an explicit winding …

**A90. Polar and hull kernels have low parallelism and generation issues many tiny launches** — *performance* · `track_gen/_src/warp_generate_polar.py:76`, `track_gen/_src/warp_generate_polar.py:102`, `track_gen/_src/warp_generate.py:622`  
→ *Fix:* - Evaluate the polar dense spline one thread per sample, as hull does. - Use a block-reduction or two-kernel bbox for normalization. - Cache the device object rather than its string. - Fuse sample, sort and assemble per …

**A91. Prototype extension points (closure, scoring) and closed layout enum: algorithm variants require editing kernels** — *extensibility* · `track_gen/_experimental/checkpoint_proto.py:223`, `track_gen/_experimental/checkpoint_proto.py:271`, `track_gen/_src/warp_generate_checkpoint.py:558`  
→ *Fix:* Pass variants as Warp @wp.func overloads selected at module-build time (generic kernels parametrised by a wp.func, e.g. via wp.Kernel with closures or wp.overload), or as separate small stage kernels composed by the …

**A92. Silent-divergence hazards under capture: rebind, unstable reset masks and replaceable sub-tools are not detected** — *correctness* · `track_gen/_src/course.py:408`, `track_gen/_src/course.py:606`, `track_gen/_src/course.py:335`  
→ *Fix:* Record the bound pointers (or a capture epoch) when a capture is observed (wp.get_device().is_capturing during step()), and raise on rebind afterwards, or invalidate Graph B. Expose a Course-owned, stable reset-mask …

**A93. CourseConfig is a monolithic, hard-wired config with a hand-maintained cross-validation matrix** — *extensibility* · `track_gen/_src/course.py:126`, `track_gen/_src/course.py:143`, `track_gen/_src/course.py:191`  
→ *Fix:* Use nested per-component configs that validate themselves, e.g. `components=[CheckpointCfg(spacing=...), SegmentsCollision(max_boxes=...), HeightFieldCfg(res=...)]`. Course should validate only the cross-component …

**A94. bind() hard-codes one agent model (planar box with quatf orientation and vec2f half-extents)** — *api-surface* · `track_gen/_src/course.py:377`, `track_gen/_src/course.py:392`, `track_gen/_src/course.py:136`  
→ *Fix:* Bind a named input dict or AgentState object. Each component declares its required inputs (name -> shape/dtype), and Course validates them generically with _check_arr.

**A95. Representation limits baked into the utilities: one shared count, index-aligned boundaries, planar collision, fixed fields** — *extensibility* · `track_gen/_src/types.py:1305`, `track_gen/_src/collision.py:152`, `track_gen/_src/collision.py:114`  
→ *Fix:* Document index alignment and level cross-sections as explicit contract items now. Longer term, allow per-polyline counts (or a cross-section correspondence), an optional per-point attribute dict (`extras: dict[str, …

**A96. Utilities bind to a Track instance and cannot be rebound; producers must overwrite buffers in place forever** — *extensibility* · `track_gen/_src/collision.py:323`, `track_gen/_src/props.py:320`, `track_gen/_src/checkpoints.py:85`  
→ *Fix:* State the 'persistent in-place result' contract in the generator-author docs and enforce it (e.g. a Track owner/epoch). Add `rebind(track)` that validates the layout and swaps references outside capture, raising if a …

**A97. Generator winding leaks into utility semantics (TrackFrame.n sign, checkpoint left/right naming)** — *api-surface* · `track_gen/_src/localize.py:64`, `track_gen/_src/checkpoints.py:177`, `track_gen/_src/types.py:1309`  
→ *Fix:* Expose a winding-normalized lateral coordinate (e.g. `n_outer`, positive toward outer) using Track.winding, and name the checkpoint endpoints `inner`/`outer` (or orient left/right by travel direction).

**A98. PropSampler placement modes and outputs are closed enums inside one kernel** — *extensibility* · `track_gen/_src/props.py:299`, `track_gen/_src/props.py:212`, `track_gen/_src/props.py:76`  
→ *Fix:* Split the sampler into a reusable arc-length resampler (public) plus pluggable placement functions or kernels, and allow extra per-prop attribute arrays.

**A99. Public helpers allocate per call with no out= variant** — *api-surface* · `track_gen/_src/localize.py:476`, `track_gen/_src/localize.py:632`, `track_gen/_src/rng_kernels.py:136`  
→ *Fix:* Add out= or preallocated class variants: a CurvatureProfiler/SpeedProfiler with bake() into owned buffers, rng.sample_uniform_warp(..., out=), Track.clone_into(dst). Keep the allocating forms as thin wrappers.

**A100. Sampler capacities sized by host readback of the first batch; a static bound exists but Track has no metadata** — *graphability* · `track_gen/_src/checkpoints.py:258`, `track_gen/_src/props.py:340`, `track_gen/_src/course.py:527`  
→ *Fix:* Attach metadata to Track (n_max, spacing, half_width, closed, generator) and derive capacities statically, e.g. ceil((N_max*spacing + 2π·hw)/cp_spacing). Keep max_checkpoints/max_props overrides.

**A101. No public SDK for utility authors; helpers are private and boilerplate is copy-pasted** — *modularity* · `track_gen/_src/runtime.py:71`, `track_gen/_src/collision_geom.py:1`, `track_gen/_src/checkpoints.py:33`  
→ *Fix:* Publish a small `track_gen.ext` (or `track_gen.runtime`) module containing `check_array`, a `BatchLayout.from_track(track)` helper, a `BoundInputs` mixin, a `PreallocatedResult` base with a generic dataclass-field …

**A102. Heightfield and CourseLine: private coupling, hard-coded choices, not public, brute-force bake** — *modularity* · `track_gen/_src/heightfield.py:19`, `track_gen/_src/heightfield.py:129`, `track_gen/_src/heightfield.py:84`  
→ *Fix:* Move the AABB kernel into a public geometry module, expose padding through Course, and export CourseLine publicly with a pluggable interpolant. Consider an accelerated bake (a hash grid or wp.Mesh query) for large res …

**A103. No mechanical check for host syncs or allocations on capture paths** — *graphability* · `docs/contributing/writing-a-generator.rst:51`, `tests/test_gate_generator.py:91`, `tests/test_collision_cuda_graph.py:38`  
→ *Fix:* - Provide `track_gen.testing.assert_capture_safe(fn, device)` with two modes. - CPU mode: monkeypatch wp.synchronize, wp.synchronize_device, wp.synchronize_stream, wp.array.numpy/list, …

**A104. Contributing docs describe only in-tree edits; stale and incorrect references** — *docs* · `docs/contributing/writing-a-generator.rst:54`, `track_gen/_src/generator_registry.py:5`, `docs/how-it-works/cuda-graph.rst:14`  
→ *Fix:* Rewrite the guide around an out-of-tree example package (pyproject entry point, params dataclass, conformance test) plus a gate-generator section. Fix the signatures and the replay claim, remove the dead reference, and …

**A105. Doc inaccuracies and tutorial patterns that conflict with graph use** — *docs* · `docs/tutorials/batch-of-tracks.rst:88`, `docs/tutorials/batch-of-tracks.rst:124`, `docs/how-it-works/cuda-graph.rst:14`  
→ *Fix:* - Fix the shapes and signatures. - Rewrite the RL-loop example around an on-device reseed kernel and a persistent buffer. - Reconcile the performance numbers. - Add `sphinx.ext.doctest`, or a pytest job that runs …

**A106. Docs misstate the validation and lock contracts** — *docs* · `docs/utilities/overview.rst:38`, `track_gen/_src/runtime.py:30`, `track_gen/_src/collision.py:446`  
→ *Fix:* Correct the docs and either lock set_capturing or (preferably) remove it per the global-flag finding.

**A107. Documentation overstates capture safety and misdescribes the threading model** — *docs* · `track_gen/_src/course_line.py:9`, `track_gen/_src/heightfield.py:12`, `track_gen/_src/runtime.py:30`  
→ *Fix:* Document capture status for each entrypoint (self-capturing, caller-capturable, conditions), matching the graph_status table, and fix the threading claims.

**A108. param_explorer hard-codes every generator's knobs, UI sections and gate counts, and uses private kernels** — *extensibility* · `viz/param_explorer.py:58`, `viz/param_explorer.py:70`, `viz/param_explorer.py:139`  
→ *Fix:* - Give each spec a params schema: a dataclass whose fields carry metadata for range, step, label and group, and for gates a `count_bounds(params)` callable. - Generate the UI sections from the schema. - Expose public …

**A109. test_shape_variety reshapes vec3f output as vec2f, defeating the only registry-wide degenerate-shape gate** — *correctness* · `tests/test_shape_variety.py:46`, `tests/test_shape_variety.py:38`, `track_gen/_src/types.py:1255`  
→ *Fix:* - Use `.reshape(E, -1, 3)[..., :2]`. - Add a self-test that runs the gate on a synthetic circular Track and asserts it fails. - Move the check into the shipped conformance kit.

**A110. Other hard-coded generator lists, and inconsistent registry semantics that tests rely on** — *extensibility* · `tests/test_gate_generator_registry.py:99`, `tests/test_generator_registry.py:18`, `tests/test_generator_registry.py:61`  
→ *Fix:* - Replace exact-equality checks with subset checks. - Drive assets, goldens and doc tables from the registry, or tag built-ins (`spec.builtin=True`). - Give both registries one policy: raise on duplicates, with …

**A111. Benchmarks and visualization reach into private pipeline scratch for intermediate stages; no public introspection hook** — *api-surface* · `benchmarks/compare_generators.py:47`, `viz/render_readme_assets.py:177`, `viz/render_readme_assets.py:207`  
→ *Fix:* - Add a public, opt-in stage-tap API, e.g. `TrackGenerator(..., keep_intermediates=('raw','resampled','relaxed'))` returning a typed `Intermediates` object, or per-stage outputs on a pipeline object. - Port …

**A112. No eager mode on CUDA: benchmark 'eager' rows actually measure graph replay, and memory is measured with the wrong allocator** — *performance* · `benchmarks/benchmark_pipeline.py:39`, `benchmarks/benchmark_pipeline.py:50`, `benchmarks/benchmark_yield_sweep.py:69`  
→ *Fix:* - Add a public `capture: bool | 'auto'` option to TrackGenerator, GateGenerator and Course, which also helps caller capture. - Measure memory with Warp's mempool stats (`wp.get_mempool_used_mem_current` / high-water …

**A113. warp-lang>=1.14 floor admits versions missing capture fixes the remediation relies on; driver/toolkit requirements and the unbounded ceiling are undocumented** — *graphability* · `pyproject.toml:13`, `README.md:10`, `docs/getting-started/installation.rst:85`  
→ *Fix:* Raise the floor to warp-lang>=1.16, with 1.15 as the hard minimum. Document driver R550+ (CUDA 12.4) for conditional-node paths and R545+ for in-capture module loading. Mention the +cu12 GitHub wheels for drivers older …

**A114. A load_module/force_module_load-based warm-up is unreliable at Warp 1.14 once the same kernels ran on CPU, so modules compile inside the caller's capture** — *graphability* · `track_gen/_src/track_generator.py:196`, `track_gen/_src/gate_generator.py:175`, `track_gen/_src/course.py:510`  
→ *Fix:* Implement warmup(device) for generators, Course and utilities as one real eager pass on the target device, outside any capture. Alternatively, require Warp 1.15 or later before relying on load_module or …

**A115. No lifetime anchor for buffers baked into a caller's graph: wp.Graph retains only modules, and the library's scratch, seed and RNG buffers live on objects users need not hold** — *graphability* · `track_gen/_src/track_generator.py:94`, `track_gen/_src/track_generator.py:102`, `track_gen/_src/rng_utils.py:44`  
→ *Fix:* Have the record API return a binding or handle that owns every buffer it references (generator, RNG, scratch, subtools), and document that the caller must keep it alive as long as the graph. As an alternative, have …

**A116. No validate-only or from-host constructor for Track/GateSequence; library validators need vec2f+config or overwrite the data; stage kernels do not clamp count** — *correctness* · `track_gen/_src/types.py:1318`, `track_gen/_src/warp_pipeline.py:1144`, `track_gen/_src/warp_gate.py:713`  
→ *Fix:* Add Track.validate()/GateSequence.validate(): a host-side invariant check covering stride, count <= stride, NaN padding, arclen/length consistency, winding vs signed area, unit tangents and quaternion frames. Add device …

**A117. No persistence, metadata or suite operations on result types; immutable suites must be hand-serialized with hard-coded dtype tables** — *api-surface* · `track_gen/_src/types.py:1202`, `track_gen/_src/types.py:1318`, `track_gen/_src/checkpoints.py:69`  
→ *Fix:* Define one field registry per result type (name -> dtype and per-env or per-point kind) and generate the following from it: - to_numpy/from_numpy and save/load (npz or safetensors), with a schema version and a metadata …

**A118. `scale` is not a unit system: generators read it differently (extent, origin, perimeter); absolute knobs are not derived from it; no transform is recorded** — *extensibility* · `track_gen/_src/warp_generate.py:131`, `track_gen/_src/warp_generate_polar.py:21`, `track_gen/_src/warp_generate_polar.py:202`  
→ *Fix:* Define a library-wide units contract: - One world length scale with documented per-generator meaning (for example a target perimeter or extent distribution), plus a common frame convention (centred, start at index 0). - …

**A119. Area computations in float32 depend on absolute position, and there is no re-centring step for ingested coordinates** — *correctness* · `track_gen/_src/warp_pipeline.py:88`, `track_gen/_src/warp_pipeline.py:757`, `track_gen/_src/warp_pipeline.py:124`  
→ *Fix:* Accumulate these areas relative to a per-env reference point (the first point or the bbox centre). Have the ingestion API re-centre each course and record the origin in metadata. Document the float32 precision limits …

**A120. The only working mixed-batch path (K generators + wp.copy into a combined Track) is half-capturable, doubles output memory and multiplies device-wide syncs** — *graphability* · `track_gen/_src/track_generator.py:207`, `track_gen/_src/gate_generator.py:186`, `track_gen/_src/types.py:1318`  
→ *Fix:* Remove the need for copies by letting generators write into caller-owned slice views (finding above). Until then, ship a documented helper, e.g. track_gen.compose_tracks(sources, out), that records the per-field copies …

**A121. Seed namespace is not partitioned across sub-batches: shared per-stage salts give int-seeded sub-generators identical elevation profiles** — *correctness* · `track_gen/_src/rng_utils.py:44`, `track_gen/_src/warp_zprofile.py:86`, `track_gen/_src/warp_zprofile.py:104`  
→ *Fix:* Make the seed namespace global. Options: - Add PerEnvSeededRNG(seeds, num_envs, device, env_offset=0) so an int seed expands to seed + env_offset + arange(E_k). - Or derive streams from a hash or counter-based RNG of …

**A122. Generators cannot observe the previous batch's final outcome; only their raw pre-relax output persists (track), the gate staging is mutated in place, and first-call contents differ between CPU and CUDA** — *extensibility* · `track_gen/_src/warp_pipeline.py:1553`, `track_gen/_src/warp_pipeline.py:929`, `track_gen/_src/warp_pipeline.py:1922`  
→ *Fix:* 1. Define and document what a generator may read from the previous batch. 2. Pass read-only handles to the previous final outputs: Track or GateSequence, plus final valid. Or provide an opt-in history buffer that the …

**A123. No generator lifecycle hooks: no reset, state access, capture-awareness or checkpointing of generator state** — *api-surface* · `track_gen/_src/generator_registry.py:38`, `track_gen/_src/gate_generator_registry.py:17`, `track_gen/_src/track_generator.py:105`  
→ *Fix:* Add optional spec hooks: `reset(scratch, mask_wp)` as device-side, capturable or eager outside the graph; `state(scratch)` and `load_state(scratch, ...)` for checkpointing; and a warm-up/record context flag. Expose …

**A124. Statelessness is implied by the contract but never stated or tested, while the roadmap proposes stateful, learned and codebook generators** — *docs* · `docs/appendix/future-generators.rst:21`, `docs/contributing/writing-a-generator.rst:63`, `docs/appendix/future-generators.rst:344`  
→ *Fix:* State explicitly that generators must be stateless across calls, or define the supported stateful pattern once lifecycle hooks exist. Add a per-family note in future-generators.rst. Ship a conformance test that runs a …

**A125. README, quickstart and tutorials (including the CUDA-graph tutorial) use the documented truncating regime: hw=0.03 with default N_max=384** — *docs* · `README.md:63`, `docs/getting-started/quickstart.rst:17`, `docs/tutorials/cuda-graph-in-a-sim.rst:60`  
→ *Fix:* Make N_max derived by default (N_max=None, resolved in __post_init__ from scale, spacing and a perimeter bound, like spacing), or set N_max=512 in every hw=0.03 example. Validate N_max >= 3. Re-measure the published …

**A126. Yield-sweep benchmark and report measure heavily truncated tracks: placeholder spacing/N_max left behind when 'fixed' mode was removed** — *testing* · `benchmarks/benchmark_yield_sweep.py:42`, `benchmarks/benchmark_yield_sweep.py:49`, `viz/make_report.py:123`  
→ *Fix:* Size n_max from spacing in both harnesses (or use the auto-derived N_max). Assert or report the truncated fraction per row (from the proposed device flag, or count==N_max as a proxy). Regenerate the sweep and report …

**A127. No test covers Track N_max truncation; the only test that truncates does so by accident and asserts nothing about it** — *testing* · `tests/test_gate_track_compat.py:18`, `tests/test_warp_pipeline_e2e.py:36`, `tests/test_end_to_end_relaxation.py:30`  
→ *Fix:* Add deterministic tests through the stage functions. Feed a synthetic near-circle with P ≈ 1.1*N_max*spacing and assert truncated==1 and valid==0 on CPU and under CUDA graph replay. Feed P with floor(P/s)+1 == N_max …

**A128. The validity gate's w_floor check is a tautology: w is a constant fill of half_width** — *correctness* · `track_gen/_src/warp_pipeline.py:1869`, `track_gen/_src/warp_pipeline.py:1854`, `track_gen/_src/warp_pipeline.py:661`  
→ *Fix:* Remove the term, or validate half_width > w_floor in __post_init__ and document the check as config-level. Reinstate a real per-point width-floor check only once a per-point width channel exists.

**A129. Oversample-and-compact and fallback substitution are possible only as user code: no candidate axis, no gather/commit primitive, and no hook between Course generation and refresh** — *extensibility* · `track_gen/_src/warp_pipeline.py:1528`, `track_gen/_src/warp_pipeline.py:962`, `track_gen/_src/course.py:486`  
→ *Fix:* Provide public Track/GateSequence device primitives: gather(src, src_env_index) -> dst, a masked commit, and a capacity-sized allocator. Add an optional candidates_per_env to the generator configs with an in-graph …

**A130. Per-env determinism and batch independence are what every reseed/retry repair depends on, but they are unstated and untested on the Warp path** — *correctness* · `tests/test_generators.py:137`, `track_gen/_src/warp_pipeline.py:91`, `track_gen/_src/generator_registry.py:27`  
→ *Fix:* State the contract for plugin authors: outputs for env e are a deterministic function of (seed[e], config) only, with no cross-env reductions affecting results. Add a Warp-path conformance test that permutes or replaces …

**A131. The invalid-env path is undocumented, untested, and contradicted by the tutorials and vestigial oracle semantics** — *testing* · `docs/utilities/overview.rst:43`, `docs/tutorials/batch-of-tracks.rst:256`, `docs/tutorials/gate-sequences.rst:260`  
→ *Fix:* Add a 'Handling invalid envs inside a captured step' section with a capturable masking kernel example. Gate the tutorial rewards on valid. Add contract tests that force valid=0 with count>=3 (and the NaN-anchor gate …

**A132. The recommended wp.capture_while port is not a drop-in: loop cadences are tied to the host counter `it` and the shared primitives have no device gate** — *graphability* · `track_gen/_src/warp_generate_repulsive.py:835`, `track_gen/_src/warp_generate_repulsive.py:841`, `track_gen/_src/warp_generate_repulsive.py:845`  
→ *Fix:* Treat full unrolling (option i, ~4.1k nodes at defaults) as the baseline capture path. If a compact graph is wanted: add `it_dev`/`cond_dev` int32[1] to RepulsiveScratch; hoist the `count` fill out of the loop; add …

**A133. capture_while/capture_if remediation fails inside non-Warp (torch/simulator) captures and needs CUDA 12.4+; GeneratorSpec.capturable cannot express this** — *graphability* · `track_gen/_src/warp_generate_repulsive.py:845`, `track_gen/_src/warp_generate_repulsive.py:869`, `pyproject.toml:13`  
→ *Fix:* Make the conditional-node path optional: use it only when `wp.is_conditional_graph_supported()` holds and the active capture is Warp-registered (`stream in device.captures`). Otherwise fall back to the unrolled graph. …

**A134. The global early exit (the sole capture blocker) saves only parked iterations; its first readback can never fire** — *performance* · `track_gen/_src/warp_generate_repulsive.py:207`, `track_gen/_src/warp_generate_repulsive.py:597`, `track_gen/_src/warp_generate_repulsive.py:845`  
→ *Fix:* Drop the exit on the capture path; it costs little once freeze is absorbing. Also add `frozen` early-returns to the O(E*N) bookkeeping kernels so parked iterations are pure launch latency, and skip the readback at the …

**A135. The documented capture plan (per-stage graphs with host seams) would not make repulsive caller-capturable** — *docs* · `docs/superpowers/specs/2026-07-05-repulsive-generator-design.md:223`, `docs/superpowers/spikes/2026-07-05-repulsive-growth-phase1/README.md:429`, `docs/generators/repulsive.rst:32`  
→ *Fix:* Replace the plan in the spec and repulsive.rst with: absorbing freeze, then an unrolled single-graph body (or a stage-3 capture_while/capture_if when conditional nodes are available), with no host seams. Add a …

**A136. Registry-wide enumerators build a base TrackGenConfig for every registered name, so a plugin whose knobs live on a TrackGenConfig subclass crashes them** — *extensibility* · `tests/test_shape_variety.py:43`, `tests/test_warp_graph.py:41`, `tests/test_warp_graph.py:46`  
→ *Fix:* Let each spec own its parameter type: GeneratorSpec gains `params_type` (a defaulted dataclass) and a `default_config(**common)` factory, or TrackGenConfig gets a `generator_params` namespace that the spec resolves and …

**A137. The generator-facing private ABI stayed stable by accident: nothing defines or pins it, nearby helpers broke, and three breaking commits shipped without a version bump** — *api-surface* · `track_gen/_src/warp_pipeline.py:1153`, `track_gen/_src/warp_pipeline.py:1703`, `track_gen/_src/warp_pipeline.py:1383`  
→ *Fix:* Publish a versioned plugin surface (e.g. track_gen.plugin / track_gen.kernels) that re-exports the generator building blocks: sync-free, keyword-only, documented. Snapshot-test their `inspect.signature` and kernel …

**A138. Acceptance thresholds and slow-generator budgets are global constants in shared tests, and shared-stage default changes silently shift every generator's calibration** — *testing* · `tests/test_shape_variety.py:25`, `tests/test_shape_variety.py:33`, `track_gen/_src/types.py:621`  
→ *Fix:* Carry acceptance metadata (thresholds, fast config, expected metric bands) on the generator descriptor and have a shipped conformance and quality kit read it. Version pipeline behavior (goldens or a behavior-version …


## Dismissed by verifiers (12)

- Iteration loops are unrolled into the graph (~120 nodes at defaults, linear in relax_iters) — *impact: not_an_issue*
- random_walk leaves stale z for count==0 envs, violating the 'padding gets z = 0' contract — *impact: not_an_issue*
- Validity gate ignores the relax_band override used by the solver — *impact: not_an_issue*
- Replay path launches the generator's own graph into the caller's stream; removing the trailing sync does not make generate() caller-capturable at any Warp version — *duplicate*
- External (torch/simulator) captures are invisible to Warp unless the caller routes the stream and registers external=True; the library defines no contract, and an unregistered sync invalidates the capture instead of raising — *duplicate*
- A dataset/codebook generator can be captured, but the pipeline always reprocesses its output; there is no Track-level source extension point — *duplicate*
- A uniform stride forces the most demanding sub-batch's N_max and max_gates onto every generator in a mixed batch — *impact: not_an_issue*
- No capturable masked regeneration: the draw salt is a baked host int fixed at 0, stages overwrite every env in place, and the only per-call input is a private seed buffer refreshed outside the graph — *duplicate*
- Frozen envs are not absorbing, so unrolling for capture changes results, and a mixed eager-exit/graph-unrolled setup makes CUDA replay disagree with eager — *duplicate*
- Unrolled graph size scales with n_iters, which has no upper bound in config validation — *impact: not_an_issue*
- generate() keyword names are load-bearing but undocumented as such; the design spec's names would crash — *duplicate*
- Knob squatting: polar shipped reading the experimental Fourier fields, and still falls back to them, contradicting the config docs — *impact: not_an_issue*