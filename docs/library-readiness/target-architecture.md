# TrackGen 1.0 — target architecture

> Companion to [`README.md`](README.md). This design was synthesized from three independently drafted proposals (capture-first, plugin-first, incremental), scored by three judges (CUDA-graph/Warp expert, plugin-ecosystem maintainer, delivery lead), with every judge-raised flaw resolved. Warp claims were checked against Warp `v1.14.0` and `1.19.0.dev1` sources; **nothing was run on a GPU** — see *Open questions* at the end.

Contents: §0–13 architecture · [Key design decisions](#key-design-decisions) · [Roadmap](#roadmap-17-pr-sized-steps-in-7-phases) · [Open questions](#open-questions)

## 0. Invariants

1. **One emission path.** All device work, built-in or plugin, goes through a `LaunchContext`: stream-explicit `launch`/`copy`/`fill`, plus two structured helpers, `skip_unless` and `bounded_loop`. Nothing else is allowed on the hot path: no host sync, readback, allocation, lock, nested capture, `capture_launch`, `ScopedStream`, or `arr.zero_()`/`fill_()`.
2. **One executable unit.** A `Program` (a compiled pipeline), or anything implementing `Launchable` (for example the Course regenerate body). It is built and allocated at construction, warmed up once by a real eager pass, and after that only enqueued.
3. **One policy function.** `exec.run()` chooses Record, Eager or SelfGraph on each call, from `Stream.is_capturing` of the stream the work will go to. There are no module flags and no `Device.is_capturing`.
4. **One extension surface.** `track_gen.plugin` (Stage, Params, Registry, entry points) plus `track_gen.ops`. Built-ins register through the same `register(reg)` signature a third party writes.
5. **One mechanical guarantee.** `track_gen.testing` runs every registered stage, generator, preset and component through the same checks: no sync, no allocation, launch trace, poisoned replay against an eager twin. This runs in TrackGen's CI and in plugin authors' CI.

**Kept from today:** preallocated, pointer-stable results returned as the same instance; fixed-shape, count-aware, NaN-padded kernels; device-resident seeds; bind once, query with no arguments; zero per-call allocation.

## 1. Layers and module layout

| Layer | Public module | Contents | Replaces |
|---|---|---|---|
| L0 | `track_gen.exec` | `LaunchContext`, `resolve_stream`, `CaptureState`, `run`, `prepare`, errors. Private `_compat` is the ONLY code that touches Warp internals. | `runtime._sync/_CAPTURING/_CAPTURE_LOCK/set_capturing` (runtime.py:22-68); `warp_pipeline._sync/_CAPTURING` (31-48); `warp_gate._sync/_CAPTURING` (15-31); `warp_relax` `capturing=` (337, 510, 556); the three warm-up/capture/replay copies (track_generator.py:170-217, gate_generator.py:157-190, course.py:451-521) |
| L1 | `track_gen.results`, `track_gen.seeds` | `Layout`, `Track`, `GateSequence`, capability protocols, `SeedState` | types.py:1148-1347 (additive); `_seed_buf` plus the host `wp.copy` (track_generator.py:102, 188) |
| L2 | `track_gen.ops` (+ `ops.funcs`, `ops.rng`) | Sync-free, allocation-free primitives that take `ctx` first | private `_pipe._init/_sync/_fill_i32_k/_arc_resample_*/self_intersections_inplace/GenScratch/_ccw`; the duplicated samplers, sorts, Catmull-Rom, fallbacks and normalizers |
| L3 | `track_gen.plugin` | `Stage`, `StageBuilder`, `Params`/`static`/`dynamic`, `CaptureClass`, `Registry`, `Recipe`, `API_VERSION` | generator_registry.py, gate_generator_registry.py; closed enums at types.py:24-29, 616, 1078, 1141 |
| L4 | `track_gen.pipeline` | `StageRef`, `PipelineSpec`, `presets`, `build()`, `Program`, `legacy.to_pipeline_spec` | `_run_pipeline` (warp_pipeline.py:889-968), `inflate_warp` (1699-1950), `_run_gate_pipeline` (warp_gate.py:820-864), `_Scratch` (1418-1495), `_inflate_warp_alloc` (1511-1612), `_gate_warp_alloc` (793-817) |
| L4b | `track_gen.stages` | Built-in stages, registered by `track_gen.stages.register(reg)` | — |
| L5 | `track_gen` | `TrackGenerator`, `GateGenerator` (thin facades) | track_generator.py, gate_generator.py |
| L6 | `track_gen.tools`, `track_gen.course` | `Component` protocol, built-in components, `Course` | course.py isinstance and private-attribute dispatch (431-447, 524-604); the per-class method zoo |
| L7 | `track_gen.interop.torch`, `track_gen.testing` | torch capture registration and retention; conformance kit plus pytest plugin | tests that read `gen._scratch` |

During migration the kernels stay in `track_gen/_src`. Stage modules import them, and `_src` shrinks as code moves. The existing public modules (`track_gen.collision`, `track_gen.progress`, …) remain as aliases.

## 2. L0 `track_gen.exec`: the launch/capture kernel

### 2.1 Stream resolution and capture state

```python
# track_gen/exec/context.py
StreamLike = "wp.Stream | Literal['torch'] | None"

def resolve_stream(device: wp.Device, stream: StreamLike = None,
                   default: wp.Stream | None = None) -> wp.Stream | None:
    """Exactly one stream per public call. Priority:
       explicit stream > Program default (constructor stream=) > torch-capture adoption
       > wp.get_stream(device) evaluated NOW (never cached). Returns None on CPU."""
    if device.is_cpu:
        return None
    if isinstance(stream, wp.Stream):
        return stream
    if stream == "torch":
        return _compat.torch_current_stream(device)
    if default is not None:
        return default
    adopted = _compat.torch_capturing_stream(device)   # torch imported AND capturing on its current stream for THIS device
    return adopted if adopted is not None else wp.get_stream(device)

@dataclass(frozen=True, slots=True)
class CaptureState:
    capturing: bool                 # cudaStreamIsCapturing(handle): sees Warp, external, torch and simulator captures on THIS stream
    registered: wp.Stream | None    # the exact key object Warp registered for this handle (needed by capture_if); None if unregistered
    conditional_ok: bool            # registered and conditional nodes supported (toolkit+driver >= 12.4) and Warp >= 1.16

    @staticmethod
    def probe(device: wp.Device, stream: wp.Stream | None) -> "CaptureState":
        if stream is None or not stream.is_capturing:   # CPU is eager-only in 0.x
            return CaptureState(False, None, False)
        reg = _compat.registered_stream(stream)
        return CaptureState(True, reg, reg is not None and _compat.conditional_nodes_ok(device))
```

Torch adoption fixes the silent-omission case that all three proposals had. `torch.cuda.graph(g)` enters `torch.cuda.stream(capture_stream)`, so torch's current stream is the capturing one. A call with `stream=None` made inside the block therefore records into torch's graph. Before this change it ran eagerly on Warp's stream and was missing from the graph.

`Device.is_capturing` is never consulted. It is true for any Warp-registered capture on the device, from any thread, including our own self-capture (context.py:4011-4020). Using it would raise errors that are not real.

### 2.2 LaunchContext

```python
class LaunchContext:
    __slots__ = ("device", "stream", "_cap", "_host_ok")

    @classmethod
    def for_call(cls, device, stream: StreamLike = None, *, default=None) -> "LaunchContext":
        dev = wp.get_device(device)                      # canonical: 'cuda' -> 'cuda:0'
        s = resolve_stream(dev, stream, default)
        cap = CaptureState.probe(dev, s)
        return cls(dev, s, cap, host_ok=not cap.capturing)

    @property
    def capturing(self) -> bool: return self._cap.capturing   # informational: stage code must NOT branch on it (G1)

    def launch(self, kernel, dim, inputs=(), outputs=(), *, block_dim: int = 256) -> None:
        wp.launch(kernel, dim=dim, inputs=inputs, outputs=outputs,
                  device=self.device, stream=self.stream, block_dim=block_dim)

    def copy(self, dst, src, *, count=0, dst_offset=0, src_offset=0) -> None:
        wp.copy(dst, src, dest_offset=dst_offset, src_offset=src_offset, count=count, stream=self.stream)

    def fill(self, arr: wp.array, value) -> None:
        ops.fill(self, arr, value)    # typed fill kernel on self.stream. Never arr.zero_()/fill_(): their memset
                                      # uses the WP_CURRENT_STREAM sentinel (v1.14 context.py:3959), i.e. the wrong stream

    def skip_unless(self, flag: wp.array, body: Callable[["LaunchContext"], None]) -> None:
        """Branch used only as an optimisation. Contract: body is an exact no-op when flag[0] == 0,
        so running it unconditionally is always correct. Lowers to a conditional node only inside a
        Warp-registered capture, and never to Warp's host-poll fallback, which would call
        synchronize_stream inside a foreign capture (context.py:10188-10196)."""
        if self._cap.conditional_ok:
            wp.capture_if(flag, on_true=lambda: body(self), stream=self._cap.registered)  # registered KEY object
        else:
            body(self)

    def bounded_loop(self, n_iters: int, body: Callable[["LaunchContext", int], None], *,
                     active: wp.array | None = None, first_check: int = 0, check_every: int = 1) -> None:
        """body(ctx, it) for it in range(n_iters). `it` is a static Python int, so coprime cadences
        (repulsive: resample every 25 iterations, stall check every 16) stay expressible.
        `active` is a [1] int32 that the body refreshes on the iteration just before each window starts.
        REQUIRED (checked by the kit): freeze is absorbing, so once active[0] == 0 every remaining
        iteration is an exact no-op. All three lowerings are then bit-identical."""
        def window(c, lo, hi):
            for it in range(lo, hi):
                body(c, it)
        window(self, 0, min(first_check, n_iters))
        if active is None:
            return window(self, first_check, n_iters)
        for lo in range(first_check, n_iters, check_every):
            hi = min(lo + check_every, n_iters)
            if self._host_ok and not _compat.any_registered_capture(self.device):
                if self.host().read_i32(active) == 0:          # (a) eager: host poll, then break
                    return
                window(self, lo, hi)
            elif self._cap.conditional_ok:                     # (b) registered capture with CUDA >= 12.4
                self.skip_unless(active, lambda c, lo=lo, hi=hi: window(c, lo, hi))
            else:                                              # (c) any other capture (torch, sim, old driver): unrolled
                window(self, lo, hi)

    def host(self) -> "HostOps":
        """read_i32 / readback / upload / sync on self.stream through pinned buffers allocated at build.
        Raises NotCapturableError unless the call is eager and outside every capture."""
```

### 2.3 `_compat`: the only module that touches Warp internals

This module is tested at the version floor and at latest.

```python
# track_gen/exec/_compat.py
from warp._src import context as _wctx

def registered_graph(stream: wp.Stream) -> "wp.Graph | None":
    """Find the capture the same way wp.launch does: by capture id (context.py:8569-8573)."""
    rt = _wctx.runtime
    if not rt.captures or not stream.is_capturing:
        return None
    return rt.captures.get(rt.core.wp_cuda_stream_get_capture_id(stream.cuda_stream))

def registered_stream(stream: wp.Stream) -> "wp.Stream | None":
    """Return the exact wp.Stream object Warp registered.

    device.captures is keyed by object identity: wp.Stream defines no __eq__/__hash__, and
    stream_from_torch() returns a new wrapper on every call. capture_if, capture_while and
    capture_end must be passed THIS object, otherwise they take the unregistered host-poll path."""
    for key in stream.device.captures:
        if key.cuda_stream == stream.cuda_stream:
            return key
    return None

def any_registered_capture(device) -> bool: return bool(device.is_cuda and device.captures)  # synchronize_stream would raise
def conditional_nodes_ok(device) -> bool: return device.is_cuda and wp.is_conditional_graph_supported() and WARP >= (1, 16)

def anchor(stream: wp.Stream, *owners) -> None:
    g = registered_graph(stream)
    if g is not None:                                     # wp.Graph has no __slots__
        g.__dict__.setdefault("_track_gen_owners", []).extend(owners)

def torch_capturing_stream(device) -> "wp.Stream | None":
    torch = sys.modules.get("torch")                      # no import, no dependency
    if torch is None or not torch.cuda.is_initialized():
        return None
    if torch.cuda.current_device() != device.ordinal or not torch.cuda.is_current_stream_capturing():
        return None
    return wp.stream_from_torch(torch.cuda.current_stream(device.ordinal))
```

### 2.4 `prepare()`: real-launch warm-up

```python
def prepare(target: "Launchable", stream: StreamLike = None) -> None:
    """ONE real eager pass, done outside any capture (at construction by default).

    Launching for real loads every module variant the hot path uses: generic-kernel overloads
    (added at launch time; add_overload calls mark_modified, context.py:855-896), block_dim
    variants, and Warp-internal copy/fill kernels. This works at Warp 1.14, where GH-564 makes
    load_module/force_module_load unreliable after any CPU launch.

    Stage state buffers (b.state) are snapshotted before the pass and restored after it, so
    stateful plugins see the same history on CPU and CUDA. The pass does not advance seeds or
    epoch. No host sync is needed: module loading is synchronous on the host at launch, and
    execution stays stream-ordered."""
    ctx = LaunchContext.for_call(target.device, stream, default=target.default_stream)
    if ctx.capturing:
        raise CaptureError("prepare()/construction must run outside any capture")
    snap = target._snapshot_state(ctx)
    target.enqueue(ctx)                          # also leaves the initial batch (epoch 0) in the result
    target._restore_state(ctx, snap)
    target._prepared = True
```

**Execution model, documented for plugin authors.** This replaces today's undocumented 3 + 1 + replay pattern (track_generator.py:196-207), under which a plugin body ran four times.

| Mode | Python body runs | Kernels execute |
|---|---|---|
| `prepare()` | once | once |
| Each self-graph key (recording) | once | not at record time |
| Each replay | 0 times | once |
| Eager call | once | once |
| Caller's recording | once | per caller replay |

Buffer contents persist across replays. Host scalars and Python state freeze at record time.

### 2.5 `run()`: the single execution policy

```python
Policy = Literal["auto", "eager"]

def run(target: "Launchable", *, stream: StreamLike = None, policy: Policy = "auto", **kw):
    ctx = LaunchContext.for_call(target.device, stream, default=target.default_stream)
    if ctx.capturing:                                    # RECORD: the caller owns the graph
        if target.capture_class is CaptureClass.HOST:
            raise NotCapturableError(target.host_stage_ids)   # raised before ANY launch
        target._require_stream_compat(ctx)               # legacy-ABI plugins: see G2
        out = target.enqueue(ctx, **kw)                  # never capture_launch: graph launch is not capturable (error 900)
        _compat.anchor(ctx.stream, target)
        return out
    _debug.check_foreign_capture(ctx)                    # warns once if a non-library stream on this device is capturing
    if policy == "eager" or ctx.device.is_cpu:
        return target.enqueue(ctx, **kw)
    graphs = target._self_graphs(kw)                     # lazy, under a per-target RLock; one graph per non-HOST segment;
                                                         # ScopedCapture(stream=target._capture_stream, force_module_load=False)
    for i, g in enumerate(graphs):
        if g is None:
            target.enqueue_segment(i, ctx, **kw)         # HOST segment: eager on the caller's stream
        else:
            wp.capture_launch(g, stream=ctx.stream)      # recorded on a private stream, launched on the caller's
    ctx.stream.record_event(target.done_event)           # outside capture only
    return target.result
```

- The self-graph cache key is `(advance, mask.ptr if mask else 0)`. A new pointer triggers a new capture; bound inputs never do.
- A Program is single-writer. Running the same Program concurrently on two streams is a documented race; use one Program per concurrent producer.
- The process-wide `_CAPTURE_LOCK` is deleted.

### 2.6 Errors

`CaptureError`, `NotCapturableError` (names the HOST stages), `NotPreparedError`, `LegacyPluginStreamError`, and `CaptureStreamWarning`. The warning becomes an error under `TRACKGEN_STRICT=1`.

## 3. L1: results, layout, seeds

```python
@dataclass(frozen=True, slots=True)
class Layout:
    num_envs: int
    stride: int            # n_max for tracks, max_gates for gates
    device: str            # canonical str(wp.get_device(d))
    closed: bool = True    # batch-static topology

@dataclass(eq=False)
class Track:
    # the 10 existing fields are unchanged: outer/center/inner/tangent/normal [E*n_max] vec3f,
    # arclen [E*n_max] f32, length [E] f32, valid [E] i32, count [E] i32, winding [E] f32
    layout: Layout | None = None                # 0.x: optional, so hand-built Tracks still construct
    half_width: wp.array | None = None          # [E*n_max] f32 per point
    fail_mask: wp.array | None = None           # [E] int32 bitmask; valid == (fail_mask == 0)
    fail_bits: Mapping[str, int] = field(default_factory=dict)
    truncated: wp.array | None = None           # [E] i32
    required_count: wp.array | None = None      # [E] i32, the unclamped k
    generation: wp.array | None = None          # [E] i32 = epoch at commit
    attributes: Mapping[str, wp.array] = field(default_factory=dict)   # exported channels, e.g. 'acme.surface_id'
    def capabilities(self) -> frozenset[str]: ...                      # {'centerline','band','elevation',...}
    def validate(self, need: Iterable[str] = ("center", "count")) -> Layout: ...  # every field: shape, dtype, device
    def view(self, e0: int, e1: int) -> "Track": ...                   # contiguous env-slice views (kernels are view-safe)
    def copy_from(self, src: "Track", ctx: LaunchContext) -> None: ... # capturable snapshot
    def clone(self) -> "Track": ...                                    # cold; iterates dataclass fields plus attributes

class GateSequence: ...   # same additions: layout, fail_mask, fail_bits, generation, attributes

@runtime_checkable
class HasCenterline(Protocol): layout: Layout; center: wp.array; tangent: wp.array; arclen: wp.array; length: wp.array; count: wp.array; valid: wp.array; generation: wp.array
@runtime_checkable
class HasBand(HasCenterline, Protocol): outer: wp.array; inner: wp.array; normal: wp.array; half_width: wp.array
@runtime_checkable
class HasGates(Protocol): layout: Layout; position: wp.array; forward: wp.array; left: wp.array; right: wp.array; half_size: wp.array; count: wp.array; valid: wp.array; generation: wp.array
```

```python
# track_gen/seeds.py
class SeedState:
    seeds: wp.array   # [E] int32: an owned copy (default), or a caller array bound by pointer (validated once).
                      # The library NEVER writes caller seed arrays.
    epoch: wp.array   # [E] int32, owned, advanced only on device
    def set(self, ctx, seeds: wp.array, ids: wp.array | None = None) -> None   # enqueue-only scatter; epoch[ids] = 0
    def advance(self, ctx, mask: wp.array | None = None) -> None              # epoch[e] += 1 where mask[e] != 0 (one launch)
    def reseed_range(self, base: int) -> None                                  # eager only: seeds[e] = base + e, no allocation

@wp.kernel   # FIRST launch of every Program, recorded in the graph
def _effective_seed_k(seeds: wp.array(dtype=wp.int32), epoch: wp.array(dtype=wp.int32), out: wp.array(dtype=wp.int32)):
    e = wp.tid()
    s = seeds[e]
    if epoch[e] != 0:   # epoch 0 leaves the seed unchanged, so legacy keying and goldens are byte-identical
        s = wp.int32(funcs.hash32(wp.uint32(s) ^ funcs.hash32(wp.uint32(epoch[e]) * wp.uint32(0x9E3779B9))) & wp.uint32(0x7FFFFFFF))
    out[e] = s
```

The prologue replaces the host-issued `wp.copy(self._seed_buf, rng.seeds_warp)` (track_generator.py:188, gate_generator.py:166). Device-side writes to seeds or epoch therefore propagate on every replay of a caller graph. `PerEnvSeededRNG` stays as a general sampler: it gains `out=` variants and the `_new_states` mirror fix in `set_seeds_warp` (rng_utils.py:78-93). Passing `rng=` binds `rng.seeds_warp` by pointer.

## 4. L2 `track_gen.ops`

Every function takes `ctx` first, never syncs, never allocates, and has no config parameter. Every one that needs scratch owns a small `Scratch.alloc(E, ...)` type that knows its own shapes.

```python
fill(ctx, arr, value)                                      # typed kernels for i32/u32/f32/vec2f/vec3f
ArcScratch.alloc(E, M, device)
arc_resample(ctx, src, M, num, s: ArcScratch, out, count_out, *, active: wp.array | None = None)   # NaN-aware; `active` = per-env gate
resample_constant_spacing(ctx, src, src_count, spacing: wp.array, n_max, s, out, count_out, *,
                          truncated, required, tables: "ResampleTables")   # NaN- and length-aware, no readback
resample_channels(ctx, tables, pairs: Sequence[tuple[wp.array, wp.array]], interp: Sequence[str])   # same (idx, frac) lerp
self_intersections(ctx, poly, count, stride, out); turning(ctx, ...); thickness(ctx, ...); winding(ctx, ...)
frames(ctx, xy, count, stride, tangent, normal, kappa, *, closed: bool)
offset(ctx, xy, normal, half_width: wp.array, outer, inner, count, s, *, closed: bool)   # per-point width
normalize_extent(ctx, pts, count, stride, extent); angle_sort(ctx, pts, count, P, keys, out, *, winding="ccw")
sample_grid_points(ctx, seed, salt, P, cell, scale: wp.array, out, used); sample_counts(ctx, seed, salt, lo, hi, out)
closed_catmull_rom(ctx, ctrl, count, K, per_seg, out); polygon_fallback_selected(ctx, ...)
any_set(ctx, mask, flag_out); all_set(ctx, mask, flag_out)     # [E] -> [1] int32 for skip_unless / bounded_loop
curvature(ctx, track, *, out); speed_profile(ctx, track, ..., *, out)   # out= variants of localize.py:419/561
check_array(name, arr, shape, dtype, device)                   # public runtime._check_arr
# ops.funcs (@wp.func): safe_normalize2/3, cross2, ccw, nan0, seg_seg_intersect, catmull_rom, cubic_bezier,
#                       pchip_tangent, frame_quat, hash32, fail(mask, e, bit)
# ops.rng: @wp.func init(seed: int32, salt: uint32) -> uint32
#   = wp.rand_init(int32(hash32(uint32(seed) ^ salt) & 0x7FFFFFFF)), with salt a StageBuilder.salt() value
#   (already hashed: hash32_host(fnv1a_32('<stage_id>/<stream>'))). This is exactly the per-env keying PR8's
#   generator guide and b.salt docs recommend (frozen for plugin API 1), so plugin streams written against PR8
#   reproduce bit for bit once ops.rng ships: one keying, not two. It avoids the seed*K+c collapse at seed 0.
```

## 5. L3 `track_gen.plugin`: the extension surface

```python
API_VERSION = 1

class CaptureClass(enum.IntEnum):
    STATIC = 0    # fixed launch sequence: capturable by anything
    BOUNDED = 1   # data-dependent convergence with a static bound, via ctx.bounded_loop: still capturable by anything
    HOST = 2      # needs host readback/control: eager only; under caller capture -> NotCapturableError

def static(default, *, doc="", ui=None):                 # baked into the plan/graph (shapes, iteration counts, enums)
    return field(default=default, metadata={"tg": "static", "doc": doc, "ui": ui})
def dynamic(default, *, lo=None, hi=None, doc="", ui=None):   # becomes an [E] device array, read by pointer
    return field(default=default, metadata={"tg": "dynamic", "lo": lo, "hi": hi, "doc": doc, "ui": ui})

@dataclass(frozen=True, kw_only=True)
class Params:
    def validate(self, layout: Layout) -> None: ...      # validates ONLY itself
    @classmethod
    def from_legacy(cls, cfg) -> "Params": ...           # built-ins map the flat TrackGenConfig fields (0.x)

class Stage(abc.ABC):
    kind: ClassVar[str]                  # open set: 'generator','resample','width','relax','frames','offset','elevation',
                                         # 'validity','order','gate_relax','gate_frames','lift','commit', or any new string
    name: ClassVar[str]                  # third parties: dotted 'pkg.name'
    Params: ClassVar[type[Params]] = NoParams
    capture: ClassVar[CaptureClass] = CaptureClass.STATIC
    reads: ClassVar[Mapping[str, str]] = {}     # local port -> default channel (rewired by StageRef.ports)
    writes: ClassVar[Mapping[str, str]] = {}
    provides: ClassVar[frozenset[str]] = frozenset()   # e.g. {'z','tangent','half_width'}: presets skip the defaults
    topology: ClassVar[Literal["closed", "open", "any"]] = "closed"
    output: ClassVar[Literal["centerline", "anchors", None]] = None   # generators only
    api_version: ClassVar[int] = 1
    @classmethod
    def capture_class(cls, params) -> CaptureClass: return cls.capture   # may depend on params
    def capacity(self, params, layout) -> Mapping[str, int]: return {}   # e.g. {'raw': N}, {'gates': G, 'min_gates': 3}
    @abc.abstractmethod
    def build(self, b: "StageBuilder", params) -> Callable[[LaunchContext], None]: ...

class StageBuilder:                      # handed to Stage.build; everything is allocated ONCE, here
    E: int; layout: Layout; device: wp.Device; stage_id: str; scale: float
    seed: wp.array                       # [E] int32 effective seed (epoch folded in)
    def read(self, port: str) -> wp.array
    def write(self, port: str, dtype, *, per: str = "env", interp: str | None = None,
              export: str | None = None) -> wp.array            # per: 'env' | 'raw' | 'cl' | 'gates' | 'scalar'
    def temp_like(self, port: str) -> wp.array                  # ping-pong partner for in-place stages
    def carried(self, src_domain: str, dst_domain: str) -> list[tuple[wp.array, wp.array, str]]   # for resamplers
    def scratch(self, name: str, dtype, shape) -> wp.array      # namespaced 'scratch:<stage_id>/<name>'; invisible to others
    def state(self, name: str, dtype, shape, init=0) -> wp.array   # persists across runs; snapshot/restored by prepare()
    def param(self, field: str) -> wp.array                     # dynamic Params field -> [E] array
    def input(self, name: str, dtype, default=0) -> wp.array    # exposed as program.inputs[name]; writable or bindable
    def salt(self, stream: str) -> int                          # hash32('<stage_id>/<stream>'); collisions rejected per Program
    def fail_bit(self) -> int                                   # validity stages only
    def host_buffer(self, dtype, shape) -> wp.array             # pinned; HOST stages only

@dataclass(frozen=True)
class Recipe:                            # multi-stage generator, e.g. anchors + curve
    name: str; stages: tuple["StageRef", ...]; output: Literal["centerline", "anchors"]

class Registry:
    def add(self, spec: "type[Stage] | Recipe | ComponentSpec", *, replace: bool = False) -> None
        # DuplicateSpecError unless replace; replace=True -> PluginOverrideWarning naming both origins
    def get(self, kind: str, name: str): ...          # error lists available names AND quarantined plugins with reasons
    def available(self, kind: str | None = None) -> list[str]
    def describe(self, kind: str, name: str) -> "SpecInfo"   # origin dist/version, capture class, reads/writes/provides,
                                                             # Params JSON schema plus ui/doc metadata (drives the explorer)
    def unregister(self, kind: str, name: str) -> None
    @contextlib.contextmanager
    def scoped(self) -> Iterator["Registry"]: ...     # copy-on-enter, for tests

registry: Registry
# Lazy, locked and idempotent: track_gen.stages.register(reg) first, then
# importlib.metadata.entry_points(group="track_gen.plugins"), sorted by distribution name.
# Policy: TRACK_GEN_PLUGINS=warn (default; broken plugins are quarantined) | strict (CI) | off.
# A config that explicitly names a quarantined plugin raises the original error.
```

**Commitments of the PR8 generator subset** (PR10 and later extend them; none may be broken):
- `StageBuilder.raw` (the `"raw.xy"` points per env: `capacity()["raw"]`, else the raw default) is a permanent attribute of generator stages, next to the planned `b.stride_of(port)`.
- When `capacity()` omits `"raw"`, a generator's raw capacity is `TrackGenConfig.num_points`. `BatchSpec` has no `num_points`, so PR10 carries that default: `legacy.to_pipeline_spec(cfg)` passes `cfg.num_points` as the generator stage's raw default, and the presets take one (256). `capacity()` keys other than `"raw"` are rejected until a release defines them (PR11 adds `gates` / `min_gates`).
- `capture_class`, `capacity` and `build` receive a static view of the Params: reading a dynamic field raises `AttributeError`. Host scalars come from static fields only (G1).
- The arrays `b.param()` and `b.input()` return are read-only for the stage. PR8 latches live -> work at the start of each generation (D-f); a stage never observes the difference, so PR10's masked commit can drop the latch. The PR9 kit snapshots those arrays around `enqueue` and fails on any write.
- `Params` reserves the field names `validate`, `from_legacy`, `to_dict`, `from_dict`, `replace`, `schema`, `fields`, `describe` and every `_`-prefixed name, so later Params helpers (PipelineSpec serialisation) cannot clash with a plugin's field.
- `generator_params` mappings are converted by annotation (lists -> tuples, enum values -> members), so `PipelineSpec.from_dict` round-trips Params through JSON/YAML.
- Only `kind="generator"` Stages register (other kinds arrive with PR10, which validates their ports, channels and topology), and `capture=BOUNDED` is rejected until PR12 defines its contract.
- `describe()` reports `capture` (for the default Params), `output`, `writes` and `provides`, besides `params_schema` (`{}` for no parameters).

## 6. L4 `track_gen.pipeline`

```python
@dataclass(frozen=True)
class StageRef:
    kind: str; name: str; params: "Params | Mapping | None" = None
    id: str | None = None                       # default: kind
    ports: Mapping[str, str] = MappingProxyType({})   # local port -> channel rewiring

@dataclass(frozen=True)
class BatchSpec:
    num_envs: int; device: str; n_max: int; closed: bool = True; scale: float = 1.0
    on_invalid: Literal["keep", "poison"] = "keep"          # 'poison' becomes the default at 1.0

@dataclass(frozen=True)
class PipelineSpec:
    batch: BatchSpec
    stages: tuple[StageRef, ...]
    require_caller_capturable: bool = False      # reject HOST stages at build time, naming them
    def replace(self, id: str, ref: StageRef) -> "PipelineSpec": ...
    def insert_after(self, id: str, ref: StageRef) -> "PipelineSpec": ...
    def insert_before(self, id: str, ref: StageRef) -> "PipelineSpec": ...
    def remove(self, id: str) -> "PipelineSpec": ...
    def to_dict(self) -> dict: ...
    @classmethod
    def from_dict(cls, d: Mapping) -> "PipelineSpec": ...

# track_gen.pipeline.presets
def track(*, num_envs, device, n_max=384, generator="bezier", generator_params=None, width=("constant", {"half_width": 0.1}),
          spacing=None, relax="xpbd", relax_params=None, elevation="flat", elevation_params=None,
          checks=("default",), randomize=None, closed=True, on_invalid="keep") -> PipelineSpec
    # generator -> resample.constant_spacing -> width.* (or width.from_channel if the generator provides half_width)
    # -> relax.* -> frames -> offset -> elevation.* (skipped if provided) -> validity.* -> commit
def gates(*, num_envs, device, gates=12, max_gates=None, generator="bezier", ordering="ccw",
          source: Literal["anchors", "stations", "relaxed_stations"] = "anchors", elevation="flat",
          checks=("gate_default",), spacing_metric: Literal["xy", "xyz"] = "xy") -> PipelineSpec
    # anchors generator, or centerline generator + stations -> order.* -> gate_relax -> gate_arclen
    # -> elevation.* (ports rewired to gates.*; skipped if provided) -> lift -> tangents (skipped if provided)
    # -> gate_frames -> validity.* -> commit

def build(spec: PipelineSpec, *, registry: Registry | None = None, seeds: "wp.array | int | None" = None,
          out: "Track | GateSequence | None" = None, inputs: Mapping[str, wp.array] | None = None,
          stream: wp.Stream | None = None, prepare: bool = True) -> "Program": ...
```

**Planner rules** (inside `build`, all at construction):
1. Resolve each `StageRef` through the registry once and snapshot the spec objects. Later registry edits never change a live Program.
2. Coerce dict params with `Params(**d)`; unknown keys raise and list the valid fields. Run `Params.validate(layout)`.
3. Check cross-stage constraints from the declarations:
   - every read has an earlier writer or is a bound input;
   - no two stages write the same channel unless the stage declares an in-place rewrite;
   - topology agrees;
   - capacities fit (anchors capacity ≤ max_gates and min_count ≥ min_gates, replacing the overloaded `max_gates` at gate_generator.py:71-83);
   - `provides` conflicts are resolved by the preset.
4. Allocate every channel exactly once and namespace scratch. There is no attribute fall-through, so the `_Scratch.__getattr__` hijack (warp_pipeline.py:1480-1495) cannot happen.
5. Assign validity bits (built-ins 0-15; plugins 16-30; more than 15 plugin checks is a build error) and salts, rejecting collisions.
6. Call every `Stage.build`, group the results into segments of equal capture class, and run `prepare()`.

```python
class Program:                                   # implements Launchable
    device: wp.Device; layout: Layout; spec: PipelineSpec   # spec is fully resolved (concrete params, bits, plugin versions): log it
    result: "Track | GateSequence"               # same instance forever; result._owner = self
    seeds: SeedState
    inputs: Mapping[str, wp.array]               # stable [E] arrays: 'regen_mask' plus stage-declared inputs
    params: "ParamTable"                         # params['generator']['m'] -> [E] f32; .set(stage_id, field, value|array, ctx=None)
    capture_class: CaptureClass                  # max over segments
    segments: tuple["Segment", ...]
    fail_bits: Mapping[str, int]
    default_stream: wp.Stream | None
    done_event: wp.Event | None

    def enqueue(self, ctx: LaunchContext, *, advance: bool = False, mask: wp.array | None = None):
        for i in range(len(self.segments)):
            self.enqueue_segment(i, ctx, advance=advance, mask=mask)
        return self.result

    def enqueue_segment(self, i: int, ctx: LaunchContext, *, advance=False, mask=None) -> None:
        if ctx.capturing and not self._prepared:
            raise NotPreparedError(self)               # never compile inside the caller's capture
        with testing.contract.scope(ctx):              # contextvar; the kit's guards fire only inside it
            if i == 0:
                if advance:
                    self.seeds.advance(ctx, mask)
                ctx.launch(_effective_seed_k, self.layout.num_envs,
                           [self.seeds.seeds, self.seeds.epoch, self._bag["seed"]])
            for stage_enqueue in self.segments[i].enqueues:
                stage_enqueue(ctx)
            if i == len(self.segments) - 1:
                self._commit(ctx, mask)    # SOLE writer of result fields: masked copy, valid = (fail == 0),
                                           # generation <- epoch, poison policy

    def prepare(self, stream: StreamLike = None) -> None: ...
    def tap(self, channel: str) -> wp.array: ...         # read-only intermediate channel (viz, benchmarks)
    def memory_report(self) -> dict[str, int]: ...
```

**Masked commit** makes masked regeneration exact for unmasked envs. It does not depend on bitwise determinism, which the float `atomic_add` in `offset` (warp_pipeline.py:91-92) does not guarantee, and it needs no second result buffer, because the bag already holds the intermediates.

## 7. L5 facades

```python
class TrackGenerator:                    # GateGenerator has the identical surface over presets.gates / GateGenConfig
    def __init__(self, config: "TrackGenConfig | PipelineSpec", rng: "PerEnvSeededRNG | None" = None, *,
                 seeds: "wp.array | int | None" = None, stream: wp.Stream | None = None,
                 policy: Policy = "auto", out: Track | None = None, registry: Registry | None = None) -> None:
        spec = config if isinstance(config, PipelineSpec) else legacy.to_pipeline_spec(copy.deepcopy(config))
        self.config = config_snapshot                          # read-only deep copy
        self.program = pipeline.build(spec, registry=registry, out=out, stream=stream,
                                      seeds=rng.seeds_warp if rng is not None else seeds)   # prepare() runs here
    result: Track; inputs; params; seeds: SeedState; done_event: wp.Event
    @property
    def graph(self) -> "wp.Graph | None": ...                 # public self-graph for the default key
    def generate(self, num_or_ids=None, *, stream: StreamLike = None, advance: bool = False,
                 mask: wp.array | None = None, sync: bool = False) -> Track:
        out = exec.run(self.program, stream=stream, policy=self._policy, advance=advance, mask=mask)
        if sync:
            wp.synchronize_stream(...)                         # opt-in; raises if any registered capture is live on the device
        return out
    def enqueue(self, *, stream: StreamLike = None, advance=False, mask=None) -> Track:
        """Records into the active capture, or runs eagerly. Never self-graphs."""
```

## 8. L6 components and Course

```python
class Component(Protocol):
    name: str
    requires: tuple[type, ...]               # capability protocols, e.g. (HasBand,)
    inputs: Mapping[str, "InputSpec"]        # routed by Course.bind(**arrays)
    after: tuple[str, ...]                   # ordering dependencies (topologically sorted)
    def attach(self, source: "Track | GateSequence", cc: "ComponentContext") -> None   # cold: allocate from layout capacities, validate
    def bind(self, **arrays: wp.array) -> None                                          # cold: validate once, pin pointers
    def on_generate(self, ctx: LaunchContext, mask: wp.array | None) -> None            # hot: bake/sample/refresh
    def step(self, ctx: LaunchContext) -> object                                        # hot: returns its stable result instance
    def reset(self, ctx: LaunchContext, mask: wp.array) -> None                         # hot
    # result: object  (stable instance)

class ComponentContext:
    layout: Layout
    capacity: Mapping[str, int]              # static bounds from the plan, e.g. max_perimeter = stride * spacing
    def component(self, name: str) -> Component: ...
```

Built-in components are adapters over the existing classes:
- CollisionChecker (with a public `.method`), DiscChecker (now given count and valid), FrameChecker, PropSampler, CheckpointSampler, ProgressTracker, TrackLocalizer, HeightFieldBaker;
- `centerline_from_gates` (formerly CourseLine; returns a `Centerline` view, not a NaN-faked Track).

The legacy method names (`query`, `update`, `sample`, `bake`, `refresh`, `bind_inputs`) stay as aliases. Capacities come from the layout, which removes the readback sizing at checkpoints.py:258-266 and props.py:348-350. ProgressTracker and TrackLocalizer store `seen_generation[e]` and reset themselves inside their step kernels.

```python
class Course:                                   # composition root; its regenerate body implements Launchable
    def __init__(self, generator: "TrackGenerator | GateGenerator | Program", components: Sequence[Component] = (), *,
                 stream: wp.Stream | None = None, prepare: bool = True): ...
    @classmethod
    def track(cls, spec_or_cfg, *, collision: str | None = "segments", checkpoint_spacing: float, extra=()) -> "Course": ...
    @classmethod
    def gates(cls, spec_or_cfg, *, posts: float = 0.0, frames: bool = False, extra=()) -> "Course": ...
    @classmethod
    def from_config(cls, cfg: "CourseConfig") -> "Course": ...       # legacy presets
    inputs: Mapping[str, wp.array]              # 'regen_mask' and 'reset_mask' [E] int32, stable pointers
    def bind(self, **arrays: wp.array) -> None: ...
    def regenerate(self, *, mask: wp.array | None = None, advance: bool = True, stream: StreamLike = None):
        """Body: [any_set(mask) -> skip_unless(any, ...)] advance(mask) -> pipeline -> masked commit
        -> on_generate(mask) for each component -> reset(mask) for each component.
        Self-graphed as ONE graph (today's Graph A + B), or recorded into the caller's capture.
        With an all-zero mask the body is an exact identity, so the unconditional lowering is correct."""
    def step(self, *, stream: StreamLike = None) -> "StepResult": ...   # Mapping[str, object] + .valid + legacy .events/.contacts/.frame
    def reset(self, mask: wp.array | None = None, *, stream: StreamLike = None) -> None: ...
    def generate(self, seeds=None):                 # legacy wrapper: optional eager reseed + regenerate(advance=False)
        ...
```

## 9. L7 interop and testing

```python
# track_gen/interop/torch.py
@contextlib.contextmanager
def registered(*, capture_mode=wp.CaptureMode.GLOBAL) -> Iterator[wp.Stream]:
    """Use INSIDE torch.cuda.graph(g) to enable conditional lowerings and lifetime anchors.
    Wraps torch's current stream once, opens ScopedCapture(stream=s, external=True, capture_mode=...)
    ONLY if _compat.registered_stream(s) is None (avoids the double-registration overwrite at
    context.py:9946), and closes it with the SAME Stream object (capture_end looks the stream up by identity)."""
def retain(torch_graph, *owners) -> None: ...   # keeps Programs alive as long as torch_graph lives

# track_gen/testing
def check_stage(kind: str, name: str, params=None, *, devices=("cpu", "cuda:0"), upstream: PipelineSpec | None = None) -> "Report": ...
def check_generator(name: str, params=None, *, devices=("cpu", "cuda:0")) -> "Report": ...
def check_component(factory: Callable[["Track | GateSequence"], Component], *, source="track", steps=64) -> "Report": ...
def assert_caller_capturable(target: "Launchable", *, modes=("warp", "external", "torch", "torch_registered")) -> None: ...
class contract:      # contextvar-scoped guards, active inside any enqueue: wp.synchronize*, array.numpy/list,
    ...              # wp.empty/zeros/full/ones/clone/array/*_like, array.zero_/fill_, ScopedCapture/capture_launch, ScopedStream
class LaunchTrace:   # records (kernel key, dim, stream handle, array ptrs, scalars) for wp.launch/wp.copy
    ...              # asserts: every op on ctx.stream; two enqueues identical; ptrs within the Program's bag, bound inputs and result
# pytest: `-p track_gen.testing.pytest_plugin` -> fixtures parametrized over registry.available(), presets and components
```

## 10. Graph-capture contract

**G1. Enqueue purity.** This applies to every `Stage.build` closure, every `Component.on_generate/step/reset`, `Program.enqueue`, `SeedState.set/advance`, `ParamTable.set` and every `ops` call.
- Allowed: `ctx.launch`, `ctx.copy`, `ctx.fill`, `ctx.skip_unless`, `ctx.bounded_loop`, `ops.*`, and other enqueues.
- Forbidden:
  - `wp.synchronize*`, `.numpy()`/`.list()`/`int(device_value)`;
  - any allocation, including allocate-if-None conveniences (warp_pipeline.py:1055-1058);
  - `arr.zero_()`/`fill_()`;
  - direct `wp.launch`/`wp.copy` without `stream=ctx.stream`;
  - `ScopedStream`/`set_stream`, `ScopedCapture`/`capture_begin`/`capture_launch`, locks;
  - Python branches on device data or on `ctx.capturing`;
  - Python state changes that alter what a later call emits.
- Host scalars come from static Params. Anything that varies between replays is device data: seeds and epoch, inputs, dynamic params.
- HOST stages are the only exception, and they reach host operations only through `ctx.host()`.

**G2. Streams.**
- Each public call resolves exactly one stream (§2.1), and every op passes it. The library never changes Warp's current stream and never host-syncs by default; `generate(sync=True)` and `TRACKGEN_DEBUG_SYNC=1` are opt-in.
- Ordering for consumers:
  - Same-stream consumers are ordered automatically.
  - Torch work on the legacy default stream and `.numpy()` are ordered because Warp streams are created blocking (`CU_STREAM_DEFAULT`, warp.cu:3055 at v1.14 and HEAD).
  - Side-stream consumers wait on `done_event`, which is recorded outside capture only.
- **Legacy-ABI plugins** (5-arg `generate`, `_pipe._*` helpers) launch on Warp's current stream. For Programs containing them:
  - self-capture happens on `wp.get_stream(device)`;
  - Record requires the resolved stream's handle to equal the current stream's handle, otherwise `LegacyPluginStreamError`. The error carries the recipe: enter `wp.ScopedStream(wp.stream_from_torch(s))` before `torch.cuda.graph`.
  - The library never wraps calls in `ScopedStream` itself, because `ScopedStream` is process-global.

**G3. Capture detection.**
- `CaptureState.probe(resolved_stream)`, as in §2.1.
- Registration is matched by capture id or handle, never by `wp.Stream` identity.
- `_debug.check_foreign_capture` warns once per (target, stream) when a non-library stream on this device is capturing while ours is not. Library-owned private capture streams are excluded. The warning catches "captured stream A, resolved stream B" without the false positives of the incremental proposal's OTHER_STREAM rule.
- CPU reports not capturing in 0.x. CPU APIC self-graphs are opt-in from Warp 1.15, because an active APIC capture blocks every other capture process-wide.

**G4. Dispatch.**
- `run()` as in §2.5.
- Record never calls `capture_launch`; HOST raises before launching anything.
- SelfGraph captures on a private per-target stream with `force_module_load=False`, so drivers below 12.3 do not force-load every module. It launches on the caller's stream.
- HOST segments run eagerly between graph launches (Segmented). This fixes the all-or-nothing `capturable` flag at track_generator.py:88/213.

**G5. Warm-up.**
- `prepare()` is one real pass at construction (§2.4).
- Record or SelfGraph on an unprepared target raises `NotPreparedError`.
- Construction inside a capture raises.

**G6. Lifetime.** `wp.Graph` retains only module execs (context.py:4351-4381). The Program owns its bag, seeds, inputs, params and result. `result._owner` keeps the Program alive. Record anchors the target on a registered Warp graph. `interop.torch.retain` covers torch graphs. Course anchors its components with the generator. Rebuilding means new pointers: re-bind and re-capture.

**G7. Per-call device inputs.**
- `program.inputs[...]` (regen and reset masks, stage inputs), `program.params[stage][field]` (dynamic params as [E] arrays) and `seeds.seeds/epoch` are stable arrays.
- They can be written in place by the caller's own kernels inside its graph (in-graph curriculum), through `wp.to_torch` views, or by enqueue-only setters. A host scalar passed to `params.set` is baked if recorded.
- Arrays passed per call (`mask=`) are baked by the recording; for SelfGraph they are part of the cache key.

**G8. Regeneration.** `advance(mask)`, then the full fixed-shape pipeline, then a masked commit, then `generation[e] = epoch[e]`.
- Unmasked envs are untouched bit-for-bit.
- Every replay of a caller graph that contains `advance=True` produces new tracks with no host involvement.
- To reproduce an env, write back its (seed, epoch).

**G9. Iterative algorithms.**
- BOUNDED stages use `ctx.bounded_loop`, with three lowerings: (a) eager host poll, (b) windowed `capture_if` in registered captures (CUDA 12.4+, Warp 1.16+), (c) full unroll everywhere else.
- They are bit-identical because freeze is absorbing, and the kit checks (a) = (b) = (c).
- Repulsive: `n_iters=269`, `first_check=144`, `check_every=16`, giving about 4,098 kernel nodes unrolled or 8 IF nodes windowed. The readback and break at warp_generate_repulsive.py:845-846 are gone.
- Reject-and-retry becomes recompute: `advance(mask = 1 - valid)` followed by regenerate.

**G10. Non-capturable tier.**
- HOST stages run eagerly in Segmented self-graph mode, raise `NotCapturableError` under caller capture, and are rejected at build when `require_caller_capturable=True`.
- `capture_class(params)` may make a stage HOST only for some parameter settings.

**G11. Invalid envs.**
- `commit` always writes `fail_mask` and `valid`.
- With `on_invalid='poison'` (the default from 1.0), invalid envs leave with `count=0` and NaN geometry, so every utility's existing `count<3` path fires uniformly.
- `StepResult.valid` carries the mask into the caller's graph.

**G12. Versions.**
- Floor `warp-lang>=1.14,<2` through 0.x. Recording into caller captures works at 1.14 given the real-launch warm-up.
- Conditional lowerings are feature-gated (registered capture, `wp.is_conditional_graph_supported()`, Warp ≥ 1.16 for GH-1641).
- Floor `>=1.16` at 1.0.
- Document that 1.18+ PyPI wheels need driver R580+ and sm_75+.

**G13. Mechanical non-regression.** The kit runs over every registered stage, generator, preset and component:
- **CPU, every PR:** contract guards, LaunchTrace, poisoning of undeclared writes, batch slicing and permutation independence, NaN tail, closure, extent and winding.
- **GPU lane:**
  - outer capture with the mempool disabled;
  - poison, replay, compare with an eager twin;
  - `advance` + replay against the twin;
  - double replay (stateful components);
  - a nested self-graphing call must Record;
  - torch default and torch registered captures;
  - lowering equivalence (a) = (b) = (c);
  - masked-regen exactness;
  - two threads on two streams equal a serial run.

## 11. Extension model

**Packaging.**

```toml
[project.entry-points."track_gen.plugins"]
acme = "acme_tracks:register"
```

The plugin imports only `track_gen.plugin`, `track_gen.ops`, `track_gen.exec`, `track_gen.results`, `track_gen.tools` and `track_gen.testing`. Adding a plugin touches zero shared files, and CI tracks that as a metric.

**(a) A capturable track generator with per-env randomizable params**

```python
SALT_DOC = "shape"

@wp.kernel
def _superformula_k(seed: wp.array(dtype=wp.int32), salt: wp.uint32, m: wp.array(dtype=wp.float32),
                    n1: wp.array(dtype=wp.float32), n2: float, n_pts: int, extent: float,
                    out: wp.array(dtype=wp.vec2f)):
    t = wp.tid()
    e = t // n_pts
    i = t - e * n_pts
    st = rng.init(seed[e], salt)                          # same per-env stream for every i
    phase = wp.randf(st) * 6.2831853
    th = 6.2831853 * float(i) / float(n_pts)
    r = superformula_r(th + phase, m[e], n1[e], n2)       # the plugin's own @wp.func
    out[t] = wp.vec2f(r * wp.cos(th), r * wp.sin(th)) * extent

@dataclass(frozen=True, kw_only=True)
class SuperformulaParams(plugin.Params):
    points: int = plugin.static(256, doc="raw points per env")
    m: float = plugin.dynamic(5.0, lo=2.0, hi=12.0, ui="slider")
    n1: float = plugin.dynamic(0.3, lo=0.1, hi=2.0)
    n2: float = plugin.static(1.7)
    def validate(self, layout):
        if self.points < 16:
            raise ValueError("points must be >= 16")

class Superformula(plugin.Stage):
    kind = "generator"; name = "acme.superformula"; output = "centerline"
    Params = SuperformulaParams
    writes = {"xy": "raw.xy", "valid": "gen.valid"}
    def capacity(self, p, layout): return {"raw": p.points}
    def build(self, b, p):
        xy = b.write("xy", wp.vec2f, per="raw")
        ok = b.write("valid", wp.int32)
        m, n1, salt = b.param("m"), b.param("n1"), b.salt(SALT_DOC)
        def enqueue(ctx):
            ctx.launch(_superformula_k, b.E * p.points, [b.seed, salt, m, n1, p.n2, p.points, 1.44 * b.scale, xy])
            ctx.fill(ok, 1)
        return enqueue

def register(reg: plugin.Registry) -> None:
    reg.add(Superformula)

tg = TrackGenerator(presets.track(num_envs=4096, device="cuda:0", n_max=512,
                                  generator=StageRef("generator", "acme.superformula", SuperformulaParams(m=7.0))))
tg.params["generator"]["m"]     # [E] array, writable inside the caller's own graph (curriculum)
testing.check_generator("acme.superformula")
```

**(b) Gate generators.**
- **For free:** `GateGenerator(presets.gates(..., generator="acme.superformula", gates=12))`. The generator's output is `"centerline"`, so the preset inserts the built-in `stations` stage, which places arc-length-spaced anchors on `raw.xy` (or on relaxed `cl.xy` with `source="relaxed_stations"`). The shared order, gate_relax, elevation, lift, tangents, frames and validity stages then run. This gives a gate variant to every track generator, including repulsive and third-party ones, with no private kernels; today `warp_generate_polar_gates.py:6-11` and its siblings import private kernels.
- **Native:** a stage with `output="anchors"` writes `anchors.xy` and `anchors.count`, plus optional `anchors.z`, `anchors.tangent`, `anchors.half_size` and `anchors.roll`. Declaring `provides={"z","tangent"}` makes the preset drop the elevation and central-difference tangent stages, fixing the unconditional overwrites at warp_gate.py:853-862. Ordering is a pipeline stage from the `order` registry; generators never call `finish_ordered_gates`. `capacity()` returns `{'gates': G, 'min_gates': k}`.
- **Composite:** `reg.add(Recipe("acme.super_cr", stages=(StageRef("anchors", "acme.super_anchors"), StageRef("curve", "catmull_rom_closed")), output="centerline"))`.

**(c) Replacing relaxation**

```python
class Taubin(plugin.Stage):
    kind = "relax"; name = "acme.taubin"; Params = TaubinParams          # passes: static; lam, mu: dynamic
    reads = {"xy": "cl.xy", "count": "cl.count"}; writes = {"xy": "cl.xy"}  # in-place rewrite, declared
    def build(self, b, p):
        x, tmp, cnt = b.read("xy"), b.temp_like("xy"), b.read("count")
        lam, mu, S, closed = b.param("lam"), b.param("mu"), b.layout.stride, int(b.layout.closed)
        def enqueue(ctx):
            a, c = x, tmp
            for _ in range(p.passes):                                    # static trip count -> STATIC
                for f in (lam, mu):
                    ctx.launch(_laplace_k, b.E * S, [a, cnt, f, S, closed, c])
                    a, c = c, a
            if a is not x:
                ctx.copy(x, a)                                           # output location invariant (cf. warp_relax.py:502-503)
        return enqueue
```

> **As shipped in PR10b:** the in-place model is a versioned rewrite with no aliasing form. A stage rewrites a channel by declaring it in `reads` and `writes` through ONE port; `b.read(port)` is the previous version and `b.write(port)` a NEW buffer every later stage reads; the stage writes its whole output there and never writes a read buffer (`check_stage`'s `stage.isolation` catches a violation). This example therefore ends in the write buffer instead of copying back into `x`, with no copy:
>
> ```python
> def build(self, b, p):
>     x, cnt = b.read("xy"), b.read("count")
>     out, tmp = b.write("xy", wp.vec2f, per="cl"), b.temp_like("xy")
>     lam, mu, S = b.param("lam"), b.param("mu"), b.layout.stride
>     steps = [f for _ in range(p.passes) for f in (lam, mu)]          # static trip count -> STATIC
>     def enqueue(ctx):
>         src = x
>         for k, f in enumerate(steps):          # x -> tmp -> out -> tmp -> ... -> out (even count)
>             dst = tmp if k % 2 == 0 else out
>             ctx.launch(_laplace_k, b.E * S, [src, cnt, f, S, dst])
>             src = dst
>     return enqueue
> ```
>
> See docs/contributing/writing-a-stage.rst. `preferred=` (generator hints) is PR11, and the thickness check reading `cl.half_width` is PR13.

Select it with `presets.track(..., relax=("acme.taubin", {"passes": 30}))` or `relax=None`. A generator can declare `preferred={"relax": "none"}`; presets, Course and the explorer honour it. The thickness check reads `cl.half_width` with its own tolerance, so it no longer depends on XPBD's separation target.

**(d) Variable width.** Either the generator writes `raw.half_width` with `interp="linear"` and `provides={"half_width"}`, and the preset picks `width.from_channel`, or a `width` stage writes `cl.half_width` (for example `acme.curvature_scaled`). Downstream:
- `resample.constant_spacing` carries every `raw.*` channel that has an interp to `cl.*` using the same `(src_idx, frac)` tables;
- XPBD uses `target_ij = (w_i + w_j)(1 + margin)`; the constant path stays bit-identical because `hw + hw == 2*hw` in fp32;
- `offset`, `thickness` and `w_floor` (no longer a tautology, warp_pipeline.py:1869) read the per-point array;
- `Track.half_width` is exported.

**(e) Elevation, one stage for tracks and gates**

```python
class SinusoidHills(plugin.Stage):
    kind = "elevation"; name = "acme.sinusoid_hills"; Params = HillsParams
    parameterization = "per_point"         # 'knots': the planner wraps it with knot tables + periodic PCHIP
    reads = {"cum": "cl.arclen2", "perim": "cl.length2", "count": "cl.count"}; writes = {"z": "cl.z"}
    def build(self, b, p):
        cum, perim, cnt = b.read("cum"), b.read("perim"), b.read("count")
        z = b.write("z", wp.float32, per=b.domain_of("cum"))
        amp, salt, S = b.param("amplitude"), b.salt("phase"), b.stride_of("cum")
        def enqueue(ctx):
            ctx.launch(_hills_k, b.E * S, [b.seed, salt, cnt, S, int(b.layout.closed), cum, perim, amp, p.waves, z])
        return enqueue
```

The gates preset rewires the ports automatically: `{cum: gates.cum, perim: gates.perim, count: gates.count, z: gates.z}`. There is no monkeypatch of `apply_z_profile`, no name tuples (warp_pipeline.py:1505, 1899), and no fall-through to noise (warp_zprofile.py:222).

**(f) Validity check**

```python
class MaxCurvature(plugin.Stage):
    kind = "validity"; name = "acme.max_curvature"; Params = KappaParams
    reads = {"kappa": "cl.kappa", "count": "cl.count", "fail": "valid.fail"}; writes = {"fail": "valid.fail"}
    def build(self, b, p):
        bit, k, cnt, fail = b.fail_bit(), b.read("kappa"), b.read("count"), b.read("fail")
        def enqueue(ctx):   # one thread per env: fail[e] = fail[e] | bit (sequential checks, single writer)
            ctx.launch(_kmax_k, b.E, [k, cnt, b.layout.stride, b.param("kappa_max"), bit, fail])
        return enqueue
```

> **As shipped in PR10b:** a chained check rewrites `valid.fail` in one kernel: `fail_in, fail_out = b.read("fail"), b.write("fail", wp.int32)` and `fail_out[e] = fail_in[e] | ((kmax > k_max[e]) << bit)` for every env (no copy, no aliasing). `b.fail_bit(name=None)` hands out bits 16-30 in pipeline order (a 16th plugin bit, a taken name, or a validity stage owning no bit is a `PlanError`). `validity.default` keeps the fused `_validity_k` (same launch, byte-identical `valid`) and writes bits 0-7 (generator, turning, finite, thickness, w_floor, border, grade, truncation) into `valid.fail`; the split checks are NOT registered in PR10b (a later PR). The commit is not a registered kind but the Program's epilogue, and PR10b ships the "hold / restore" form: the `Track` arrays stay the final channel versions (`Track.fail_mask` is the final `valid.fail`), a masked generation holds the unmasked envs' rows before the stages and the commit restores them after (exact without bitwise determinism), the commit derives `Track.valid = (fail_mask == 0)` when a check follows `validity.default`, and it applies `on_invalid="poison"`. Cost: no launch for an unmasked default generation (the PR9 launch sequence is unchanged), one for a derived `valid` or poison, two for a masked generation; one held copy of the `Track` (64 B per point + 24 B per env). The sole-writer commit (copying work buffers into the `Track`) remains an option for 1.0.

Enable it with `presets.track(..., checks=("default", "acme.max_curvature"))`. `program.fail_bits` and `Track.fail_bits` name every bit. The built-in `validity.default` keeps the fused `_validity_k` for speed and goldens and owns bits 0-6. The split checks (`validity.turning`, `.finite`, `.thickness`, `.w_floor`, `.border`, `.grade`, `.truncation`) are registered as well, so the default can be replaced.

**(g) Output attribute.** `b.write("surface", wp.int32, per="raw", interp="nearest", export="acme.surface_id")` produces `track.attributes["acme.surface_id"]`: [E*n_max], pointer-stable, padded with -1 (NaN for floats), carried through the resamplers, and covered by `clone`/`copy_from`.

**(h) Runtime utility, standalone or in Course**

```python
class CurvatureLookahead:                       # satisfies tools.Component
    name = "acme.lookahead"; requires = (results.HasCenterline,); inputs = {}; after = ("localizer",)
    def __init__(self, horizon: int = 8): self.horizon = horizon
    def attach(self, source, cc):
        L = source.layout
        self._src, self._frame = source, cc.component("localizer").result
        self._kappa = wp.empty(L.num_envs * L.stride, dtype=wp.float32, device=L.device)
        self.result = wp.empty(L.num_envs * self.horizon, dtype=wp.float32, device=L.device)
    def bind(self, **_): pass
    def on_generate(self, ctx, mask): ops.curvature(ctx, self._src, out=self._kappa)
    def step(self, ctx):
        ctx.launch(_lookahead_k, self.result.shape[0],
                   [self._kappa, self._src.count, self._src.layout.stride, self._frame.segment, self.horizon, self.result])
        return self.result
    def reset(self, ctx, mask): pass

course = Course.track(spec, collision="segments", checkpoint_spacing=0.5, extra=[CurvatureLookahead(8)])
course.step()["acme.lookahead"]
```

**(i) Iterative (BOUNDED) and host-driven (HOST) stages**

```python
class Repulsive(plugin.Stage):          # built-in port of warp_generate_repulsive.py
    kind = "generator"; name = "repulsive"; output = "centerline"; capture = CaptureClass.BOUNDED
    def build(self, b, p):
        s, sched = _alloc(b, p), _schedule(p)          # closed-form host schedule (lines 618-630, 681-683)
        def body(ctx, it):
            sched.maybe_upsample(ctx, it, s)            # static in `it`
            _iteration(ctx, s, sched.ncur(it))          # the 15 launches; O(N^2) kernels gated by s.frozen
            if (it + 1) % p.resample_every == 0:
                ops.arc_resample(ctx, ..., active=s.not_frozen)   # absorbing freeze (was 835-838)
            if sched.is_check(it):
                ctx.launch(_freeze_update_k, b.E, [...])
                ops.any_set(ctx, s.not_frozen, s.active)
        def enqueue(ctx):
            ctx.launch(_seed_lfinal_k, b.E, [...])
            ctx.bounded_loop(sched.n_iters, body, active=s.active, first_check=sched.first_window, check_every=p.stall_window)
            _final_upsample_and_resample(ctx, s)        # unconditional
        return enqueue

class LearnedDecoder(plugin.Stage):     # host-driven
    kind = "generator"; name = "acme.learned"; output = "centerline"; capture = CaptureClass.HOST
    def build(self, b, p):
        xy, host_xy = b.write("xy", wp.vec2f, per="raw"), b.host_buffer(wp.vec2f, (b.E * p.points,))
        def enqueue(ctx):
            h = ctx.host()                              # NotCapturableError under any capture
            h.upload(xy, self.model.sample(h.readback(b.seed)), via=host_xy)
        return enqueue
```

**(j) Per-env inputs, curriculum and ingestion.**
- `b.input("difficulty", wp.float32, default=0.5)` appears as `tg.inputs["difficulty"]`: a stable [E] array, writable in place or bound at build with `build(..., inputs={"difficulty": arr})`.
- The built-in `randomize.uniform` stage writes dynamic params from (seed, salt) ranges, generalizing the bezier-only style sampling at warp_generate.py:650-663.
- The built-in `generator/external` stage reads a bound `raw.xy`/`raw.count` input, giving a capture-safe ingestion path for external centerlines.

## 12. Data model

**Storage.** Storage stays flat: `[E*stride]` per point plus `[E]` per env, with `count[e]` real points and NaN padding beyond them (-1 for integer attributes). Kernels are view-safe because they index relative to element 0, so `Track.view(e0, e1)` and caller-owned `out=` slices work without kernel changes.

**Channel catalogue** (versioned). Plugin channels must be namespaced, for example `cl.acme.surface_id`.

| Channel | Shape | dtype | Default producer |
|---|---|---|---|
| `seed` | [E] | i32 | prologue (effective seed) |
| `raw.xy` | [E*n_raw] | vec2f | generator (output=centerline) |
| `raw.count` (opt), `raw.<attr>` | [E], [E*n_raw] | i32, f32/i32 | generator |
| `gen.valid` | [E] | i32 | generator |
| `anchors.xy/.count/.z/.tangent/.half_size/.roll` | [E*G] / [E] | vec2f/i32/f32/vec3f/f32/f32 | generator (output=anchors) or `stations` |
| `cl.xy`, `cl.count` | [E*n_max], [E] | vec2f, i32 | `resample.constant_spacing` |
| `cl.src_idx`, `cl.src_frac` | [E*n_max] | i32, f32 | resample (tables reused by `resample_channels`) |
| `cl.truncated`, `cl.required` | [E] | i32 | resample |
| `cl.half_width`, `cl.z`, `cl.<attr>` | [E*n_max] | f32 / carried | width.*, elevation.*, resamplers |
| `cl.tangent2`, `cl.normal2`, `cl.kappa`, `cl.arclen2`, `cl.length2`, `cl.winding` | | | `frames` |
| `band.outer2`, `band.inner2` | [E*n_max] | vec2f | `offset` |
| `gates.xy/.count/.cum/.perim/.z/.tangent/.half_size` | [E*G] / [E] | | gate stages |
| `valid.fail` | [E] | i32 bitmask | validity.* |

**Results.** Track and GateSequence gain the fields in §3. All are additive and optional through 0.x and required at 1.0. `commit` is their only writer.

**Validity bits:**

| Bits | Meaning |
|---|---|
| 0 | generator |
| 1 | turning |
| 2 | finite |
| 3 | thickness |
| 4 | w_floor |
| 5 | border |
| 6 | grade |
| 7 | truncation |
| 8-15 | reserved |
| 16-30 | plugin checks |

**Invalid-env sentinel table.** It applies to every component, and every StepResult exposes `valid`.

| Component | Value for an invalid env |
|---|---|
| collision | `oob=1` |
| discs, frames | `hit=0` |
| progress | frozen, `dist_to_next=0` |
| localizer | NaN s/n, `segment=-1` |
| heightfield | NaN |

**Truncation.** The constant-spacing scan writes `truncated` and `required_count` from the unclamped `k` (warp_pipeline.py:401). With `on_truncate='invalidate'` (the default) the truncation bit fails the env.

**Topology.** `Layout.closed` is batch-static and passed to kernels as an int. Stages and components declare the topologies they support and reject others loudly at build or attach time. Open topology rolls out per stage after 1.0.

**Seeds.** Each env is keyed by (seed, epoch), folded into `seed` by the prologue. New code uses `ops.rng.init(seed, salt)` (§4) with salts hashed from `'<stage_id>/<stream>'` and collision-checked. Built-ins keep their legacy `seed*K+c` keying until 1.0, so goldens stay byte-identical at epoch 0.

## 13. Config model

**Three tiers, all frozen, copied and hashable:**
1. `BatchSpec(num_envs, device, n_max, closed, scale, on_invalid)`. The device is canonicalized with `wp.get_device`, which fixes the `'cuda'` vs `'cuda:0'` mismatch and the substring checks at track_generator.py:171-172.
2. Per-stage `Params`, owned by each stage. The built-in generators' Params keep the flat prefixed field names (`polar_num_knots`, `voronoi_*`, `repulsive_*`; PR8 decision D-b) as their permanent schema through 1.0, so `generator_params` mappings and YAML written against PR8 stay valid; stage-local names would need aliases and are not planned. `static` fields (shapes, iteration counts, enums, Python-level branches) are baked. `dynamic` fields become [E] device arrays: generator shape knobs, z ranges and amplitudes, relax margin and rates, and `half_width` within the spacing-derived capacity. Field metadata (`doc`, `ui`, `lo`/`hi`, test overrides, acceptance thresholds) drives `describe()`, the parameter explorer (no more hard-coding at viz/param_explorer.py:58) and the shared tests. Cross-stage values are derived by the planner, for example `spacing = spacing or 0.6 * nominal_half_width`; nothing is mutated in `__post_init__` (types.py:851).
3. `PipelineSpec`, built by presets. `to_dict()`/`from_dict()` make it work with Hydra or YAML and make it loggable for reproducibility.

**Legacy.**
- `TrackGenConfig` and `GateGenConfig` remain valid inputs for all of 0.x. `legacy.to_pipeline_spec(cfg)` maps the 107 flat fields through each built-in's `Params.from_legacy` mapper.
- Generator-specific validation moves out of `TrackGenConfig.__post_init__` (types.py:697-791, including the repulsive branch at :776) into each stage.
- The roughly 21 dead or oracle-only fields (`energy_*`, `tp_*`, `smooth_finish*`, `relax_chunk_size`, `relax_use_warp`, `max_regen_iters`, `min_angle`, `output_mode`, the Fourier fields) warn when set, and are removed at 1.0.
- The asserts at track_generator.py:76-83 become planner errors that name the registry.
- `CourseConfig` becomes a preset factory (`Course.from_config`).

**Mutation semantics.**
- Facades deep-copy their config at construction and expose a read-only `.config`. Configs become `frozen=True` at 1.0.
- To change a dynamic param, write `program.params[...]` in place, by host setter or inside the caller's graph. It is visible on the next replay with no re-capture.
- To change a static field, call `program.rebuild(**changes)`. It returns a new Program with new pointers, so consumers must re-bind and callers must re-capture.

---

# Key design decisions

## D1. Warp version floor

**Decision (updated 2026-10-22, owner decision):** the floor is `warp-lang>=1.18,<2` from 0.2.0.dev3 on; Warp 1.14 is no longer supported and no compatibility code for older Warp is kept. GPU CI runs a {1.18.0, latest} matrix. Documentation states that 1.18+ PyPI wheels need driver R580+ and sm_75+. (Original plan: keep `>=1.14` for all of 0.x and raise to `>=1.16` at 1.0; the Warp-version gates below are therefore always satisfied.)

Feature gates:
- Conditional lowerings (`skip_unless`, windowed `bounded_loop`) are used only when all of these hold: the capture is Warp-registered, `wp.is_conditional_graph_supported()` is true (toolkit and driver ≥ 12.4), and Warp ≥ 1.16 (the GH-1641 fix).
- CPU APIC self-graphs are opt-in from Warp 1.15, because an active APIC capture blocks every other capture process-wide (context.py:9828-9830 region).

**Alternatives considered:**
- 1.16 now (capture-first M0, plugin-first step 1). Rejected: it strands simulator stacks pinned to older Warp. Its only hard justification was the DryRun warm-up's dependence on GH-564, and that warm-up is dropped (D3).
- 1.14 forever. Rejected: it gives up conditional nodes.

**Why it works:** recording into a caller capture needs nothing newer than 1.14, given explicit streams, no sync/allocation/nested capture, and a real-launch warm-up.

## D2. Stream policy

**Decision:**
- Each public call resolves exactly one stream, in priority order: explicit `stream=` > Program default > torch-capture adoption > `wp.get_stream(device)` evaluated at call time.
- Every launch, copy and fill passes it. Fills use kernels, because `array.zero_()/fill_()` memset uses the WP_CURRENT_STREAM sentinel (v1.14 context.py:3959) and so targets the wrong stream once streams are explicit (this already affects warp_pipeline.py:983-984).
- The library never calls `ScopedStream`/`set_stream`, which are process-global.
- Self-graphs are captured on a private per-Program stream (`ScopedCapture(stream=s)` does not switch the current stream) and launched on the caller's stream with `capture_launch(graph, stream=ctx.stream)`.
- No host sync by default; `generate(sync=True)` and `TRACKGEN_DEBUG_SYNC=1` are opt-in. `done_event` is recorded outside capture only.

**Alternatives considered:**
- An owned stream with `wait_stream` handoff on every call (plugin-first). Adds two event operations per call for no benefit.
- `ScopedStream` around legacy plugins (incremental). Process-global.
- Keeping `generate()` blocking by default (incremental). It hides ordering bugs and raises whenever another thread holds a registered capture on the device, because `synchronize_stream` checks `device.is_capturing` (context.py:9198).

**Why async is safe:** Warp streams are created `CU_STREAM_DEFAULT` (blocking; warp.cu:3055 at v1.14 and HEAD). Torch on the legacy default stream and `.numpy()` (null-stream copy) remain ordered. Side-stream consumers use `done_event`.

**Legacy-ABI plugins** launch on the current stream. They are self-captured on that stream, and Record raises `LegacyPluginStreamError` unless the resolved handle is the current one.

## D3. Warm-up

**Decision:** `prepare()` is ONE real eager pass at construction. Declared stage state is snapshotted and restored around it, and seeds and epoch are untouched. Record or SelfGraph on an unprepared target raises.

**Alternatives considered:**
- DryRun + `load_module` (capture-first). Rejected: generic kernels add overloads at launch time and `add_overload` calls `module.mark_modified()` (v1.14 context.py:855-896), so the first real launch would recompile inside the caller's capture. DryRun also misses block_dim variants and Warp-internal kernels, and `load_module` is wrong after CPU launches at 1.14 (GH-564).
- 3x warm-up (today). Executes plugins 4x on the first call and is undocumented.

**Cost:** one extra execution at construction, which also leaves the epoch-0 batch in the result.

## D4. Capture detection

**Decision:**
- Probe only the resolved stream: `Stream.is_capturing` (cudaStreamIsCapturing).
- Determine registration in `_compat`:
  - for module retention and anchors, by capture id, as `wp.launch` does (context.py:8569-8573);
  - for the key that `capture_if` needs, by cuda handle match.
- When `stream` is unspecified and torch (already in `sys.modules`) is capturing on its current stream for this device, adopt that stream.
- Warn once when a non-library stream on the device is capturing while ours is not. This becomes an error under `TRACKGEN_STRICT=1`. Library-owned private capture streams are excluded.

**Rejected:**
- `Device.is_capturing`. It is `captures or stream.is_capturing` (context.py:4011-4020), so it fires for other threads' captures, including our own self-capture. This is incremental's fatal flaw, and plugin-first's OR has the same problem.
- Lookup by `wp.Stream` identity. `device.captures` is keyed by object identity, `Stream` has no `__eq__`/`__hash__`, and `stream_from_torch` builds a new wrapper on every call (torch.py:391). This was the fatal flaw in both capture-first and plugin-first.
- Debug-only mismatch detection. It allows silent omission, which was capture-first's fatal flaw.
- Raising on a foreign torch capture. Adoption is strictly more useful: `torch.cuda.graph` makes its capture stream current, so the user's intent is unambiguous.

## D5. Execution policy

**Decision:** one `run()` function with three branches.
- **Record:** the resolved stream is capturing. Enqueue only, then anchor.
- **Eager:** `policy='eager'` or CPU.
- **SelfGraph:** lazily captured per `(advance, mask ptr)` key. HOST segments run eagerly between graph launches.

Record never calls `capture_launch`: a graph launch is not capturable (error 900), and Warp has no child-graph path outside conditional bodies. There is no Record-splice through `capture_if(ONE, graph)`, which judges flagged as a source of complexity and stale-pointer risk.

**Rejected:** four Executor classes (capture-first), because the behaviour fits in about 30 lines; a per-segment GraphRunner with foreign-capture fall-through (plugin-first), which was a bug.

## D6. Non-capturable tier

**Decision:** `CaptureClass` has three values.
- **STATIC.**
- **BOUNDED:** `ctx.bounded_loop` with a static Python `it` and an absorbing-freeze invariant. It has three lowerings — eager host poll, windowed `capture_if` in registered captures, and full unroll elsewhere — which are bit-identical by construction and checked by the kit. It is always caller-capturable.
- **HOST:** `ctx.host()`. Eager only. Segmented self-graph keeps the tail captured. Under caller capture it raises `NotCapturableError` before any launch. `require_caller_capturable=True` rejects it at build time.

`capture_class(params)` can depend on params.

**Rejected:**
- plugin-first's CONDITIONAL class, which fails in torch captures. Every conditional use here has an unrolled fallback.
- `capture_while`-per-stage for repulsive. It is not a drop-in (coprime cadences) and Warp's unregistered fallback host-polls inside foreign captures (context.py:10188-10196).

## D7. Regeneration and seeds

**Decision:**
- `SeedState(seeds, epoch)`. The library never writes seed arrays.
- A recorded prologue `_effective_seed_k` folds epoch into an effective `seed` channel. Epoch 0 leaves the seed unchanged, so goldens stay byte-identical.
- `advance(mask)` increments epoch on device.
- A masked `commit` is the only writer of result fields, and it sets `generation <- epoch`.

**Why:**
- Unmasked envs are preserved bit-for-bit without relying on determinism. `offset` uses float `atomic_add` (warp_pipeline.py:91-92), so bitwise determinism is not guaranteed.
- No extra result-sized memory.

**Rejected:**
- `mix32` seed mutation (incremental). It writes into caller-aliased arrays and loses (seed, epoch) reproducibility.
- `staged_commit` (capture-first). Doubles memory.
- Recompute-and-trust-determinism.

**RNG scheme:** built-ins keep `seed*K+c` in 0.x; new code uses `ops.rng.init(seed, salt)` (§4, the PR8 keying) with hashed, collision-checked salts; built-ins switch at 1.0 in one deliberate golden re-baseline.

## D8. Config split

**Decision:** three frozen tiers.
1. `BatchSpec`/`Layout`: num_envs, canonical device, n_max, closed, scale, on_invalid.
2. Per-stage `Params`: `static` fields are baked; `dynamic` fields become [E] device arrays read by pointer.
3. `PipelineSpec`: built by presets, with `to_dict`/`from_dict`.

`TrackGenConfig`/`GateGenConfig` stay valid inputs for all of 0.x through `legacy.to_pipeline_spec`. Facades deep-copy their config at construction; configs become frozen at 1.0.

**Rejected:**
- Generated `wp.struct` `Params.Device` classes (capture-first). Codegen complexity; per-field arrays are simpler and transparent.
- [1]-vs-[E] tunables (plugin-first). Always [E]: E floats per field is negligible and keeps kernels uniform.
- A one-minor-release compat window (plugin-first). Too aggressive for RL users.

## D9. Registry and entry points

**Decision:**
- One `Registry`, keyed by kind (an open set of strings).
- Built-ins are registered explicitly by `track_gen.stages.register(reg)`, which works from a source checkout. Third parties are then loaded from `entry_points(group="track_gen.plugins")`, sorted by distribution.
- Duplicates raise unless `replace=True`, which emits `PluginOverrideWarning` naming both origins.
- Quarantine policy: `warn` (default), `strict` for CI, `off`. A config that explicitly names a quarantined plugin raises.
- `api_version` is checked. Third parties use dotted names.
- `build()` snapshots specs, so later registration never alters a live Program.

**Rejected:**
- Eight per-kind registries (incremental).
- `PluginHost` with 14 sub-registries and a global ChannelCatalog (plugin-first).
- Dogfooding through TrackGen's own entry point (capture-first). It breaks in editable/source runs without metadata.

## D10. Pipeline model

**Decision:**
- An open, ordered stage list declared as data.
- Each stage declares read and write ports, which `StageRef.ports` can rewire; this is how one elevation stage serves both tracks and gates.
- A small planner allocates channels once and namespaces scratch.
- Interp channels are carried by the resamplers on shared `(idx, frac)` tables.
- Validity checks set bits. The built-in fused `_validity_k` is kept as `validity.default` for performance and goldens, and individual checks are also registered.
- Third parties can add new kinds (banking, segmentation) with `insert_after`.

**Rejected:**
- Fixed slots (incremental): third parties cannot add a new kind.
- Roles, symbolic dims and frozen results (plugin-first).

## D11. Gates

**Decision:** gate generators are `output='anchors'` stages. Any `output='centerline'` generator gets gates through the `stations` adapter. Ordering, elevation and tangents are pipeline stages, skipped when the generator declares them in `provides`. `Recipe` allows multi-stage generators.

**Rejected for now:** decomposing every built-in into anchors+curve (plugin-first step 8). Large output-change risk for a deduplication benefit; listed as an open question.

## D12. Invalid envs and truncation

**Decision:** `fail_mask` bits are always present. `on_invalid='keep'` is the default in 0.x and `'poison'` the default at 1.0. One sentinel table covers every component. Truncation fails closed in PR5, with `on_truncate='flag'` as the escape hatch.

## D13. Lifetime

**Decision:**
- The Program owns every buffer, and `result._owner` points back to it.
- Record anchors the target on registered Warp graphs through `Graph.__dict__` (`wp.Graph` has no `__slots__`).
- `interop.torch.retain()` covers torch graphs.
- Course anchors its components.

**Rejected:** relying only on `_owner` (incremental's weakness).

## D14. Testing as the guarantee

**Decision:** the shipped `track_gen.testing` kit includes:
- contract guards and LaunchTrace with a pointer allowlist (CPU);
- mempool-disabled outer capture, poisoned replay against an eager twin, and advance/replay checks (GPU);
- torch default and registered captures;
- BOUNDED lowering equivalence;
- masked-regen exactness;
- undeclared-write poisoning;
- batch slicing and permutation independence.

It runs over everything registered, so a capture regression fails CI mechanically for built-ins and third-party plugins alike.

---

# Roadmap: 17 PR-sized steps in 7 phases

**Golden rules for every PR:**
- Goldens stay byte-identical except for four deliberate, release-noted changes:
  - PR5: truncation now fails validity;
  - PR12: repulsive uses an absorbing freeze;
  - PR16: hashed RNG salts become the default;
  - PR16: `on_invalid='poison'` becomes the default.
- Every PR adds or extends a test gate on both CI lanes.
- Phases 1-2 (capture) and Phase 3 (plugin surface) can run in parallel with two engineers. Phase 3 needs only PR0 and PR2.

## Phase 1: unblock caller capture of generation (PR0-PR3)

### PR0: Guardrails: CI lanes, CUDA detection, capture probe skeleton
- **Scope:**
  - `.github/workflows/tests.yml`:
    - CPU lane on every PR: `pytest -m "not cuda and not slow"`;
    - GPU self-hosted lane: `-m cuda` with `TRACKGEN_REQUIRE_CUDA=1`, over a matrix of warp-lang {1.14.0, latest}.
  - `tests/conftest.py`: a shared `cuda` marker and fixture based on `wp.is_cuda_available()`. It replaces 33 `torch.cuda.is_available()` gates, e.g. test_warp_graph.py:25, test_collision_cuda_graph.py:15, test_course_cuda.py:10, and the DEVS lists such as test_warp_pipeline_e2e.py:24. With REQUIRE_CUDA set, the session fails if any cuda test was skipped.
  - `pyproject.toml`: exclude `track_gen._experimental` from the wheel and pin `warp-lang>=1.14,<2`.
  - Fix the vec3f-as-vec2f reshape at `tests/test_shape_variety.py:46`.
  - `track_gen/testing/_probe.py`:
    - a CPU monkeypatch probe for sync, readback and allocation;
    - a CUDA helper: outer `wp.ScopedCapture` with `wp.set_mempool_enabled(dev, False)`, poison, replay, compare to an eager twin (generalized from tests/test_course_cuda_graph.py).
  - Strict-xfail target tests: TrackGenerator, GateGenerator and Course generation recorded inside ScopedCapture and torch.cuda.graph; gates-mode step+reset in one graph; reseed-then-replay; repulsive capture.
- **Fixes:** No test CI; CUDA tests silently skip on CPU-only torch; test_shape_variety broken gate; experimental modules ship in the wheel; no mechanical check for host syncs/allocations (skeleton).
- **Test gate:** both lanes green; the GPU lane reports more than 0 cuda tests executed; the xfails are strict.
- **Risk:** low. Needs a GPU runner.
- **Depends on:** nothing.

### PR1: Delete host syncs and capture flags
- **Scope:**
  - `_src/runtime.py`: `_sync` becomes an alias that acts only with `TRACKGEN_DEBUG_SYNC=1` and when the stream is not capturing. `set_capturing` becomes a no-op with DeprecationWarning; the `_CAPTURING` names stay defined.
  - `warp_pipeline._sync` (42-48) and `warp_gate._sync` (27-31) get the same alias. The names are kept because the out-of-tree grammar plugin calls `_pipe._sync`.
  - `warp_relax`: `capturing=` is accepted and ignored; the syncs at 510-511 and 556-557 are deleted.
  - All 48 `_sync` sites and the 17 `wp.synchronize` calls in utilities become no-ops: progress.py:346,359; collision.py:468; collision_frames.py:164; localize.py:335,353; props.py:386; checkpoints.py:299; heightfield.py:152; course_line.py:156. The facades keep a blocking sync until PR3.
  - The `count_wp.numpy()` readback in `resample_constant_spacing` (warp_pipeline.py:1127-1139) runs only in allocating mode.
  - Repulsive: line 784 calls `_arc_resample_inplace`; the sync at 861 is deleted.
  - `"cuda" in str(device)` becomes `wp.get_device(d).is_cuda`. The four `wp.init` guards are merged.
- **Fixes:** three private process-global _CAPTURING flags; _sync global wp.synchronize (~48 sites); utilities need set_capturing for caller capture (every safe_with_flag row of the matrix becomes safe); "cuda" substring and alias checks; redundant wp.init guards; repulsive blockers (2) and (3).
- **Test gate:**
  - CPU: `tests/test_no_host_sync.py` patches `wp.synchronize*` and `array.numpy` to raise, then drives every wrapper, every generator `_run`, every utility's query/update/reset/sample/bake/refresh, and Course.step/reset. There must be zero calls.
  - GPU: every existing poisoned-replay utility test passes without `set_capturing`.
  - Goldens byte-identical.
- **Risk:** low-medium.
  - Utilities stop blocking the host. Ordering stays correct for `.numpy()` (a null-stream copy) and for torch on its legacy default stream, because Warp streams are created `CU_STREAM_DEFAULT` (blocking; warp.cu:3055 at v1.14 and HEAD).
  - Torch consumers on side streams must order explicitly (release note).
  - Benchmarks must sync before reading timers.
- **Depends on:** PR0.

### PR2: LaunchContext and explicit streams
- **Scope:**
  - New `track_gen/exec/{__init__,context,_compat,errors}.py`: LaunchContext, resolve_stream with torch adoption, CaptureState, and `registered_stream`/`registered_graph` (matched by capture id and handle).
  - New `track_gen/ops/fill.py` with typed fill kernels.
  - Thread `ctx: LaunchContext | None = None` through every wrapper in warp_pipeline, warp_gate, warp_relax, warp_zprofile, the six generator modules, the four gate generator modules and all utilities: 187 `wp.launch` and 23 `wp.copy` sites, none of which pass `stream=` today. `None` resolves to `LaunchContext.for_call(device)`; this shim is removed at 1.0.
  - `area_a.zero_()`/`area_b.zero_()` (warp_pipeline.py:983-984) become `ctx.fill`. Warp's memset uses the WP_CURRENT_STREAM sentinel, so this was a latent wrong-stream bug.
  - GeneratorSpec gains `accepts_ctx` (True for built-ins). Legacy 5-arg plugins are flagged `stream_explicit=False`.
- **Fixes:** No stream parameter (the library relied on the process-global current stream); external captures silently omit work (stream half); latent wrong-stream memset; ScopedStream is process-global (never used).
- **Test gate:**
  - CPU LaunchTrace probe: every launch or copy reached from a hot path passes `stream == ctx.stream`.
  - GPU: an enqueue with `stream=s` (s not current) records into a capture on s.
  - Goldens byte-identical.
- **Risk:** medium. The diff is wide but mechanical, and the probe catches missed sites.
- **Depends on:** PR1.

### PR3: Program, run() and capture-aware facades (caller capture of generation lands)
- **Scope:**
  - New `track_gen/pipeline/program.py`: a Program holding ONE `legacy` stage that wraps the ctx-threaded `_run_pipeline` or `_run_gate_pipeline`.
  - New `track_gen/exec/run.py`: Record, Eager and SelfGraph, plus `prepare()` as a real-launch warm-up.
  - TrackGenerator (track_generator.py:59-219) and GateGenerator (gate_generator.py:32-192) become facades:
    - API: `generate(num_or_ids=None, *, stream=None, sync=False)`, `enqueue()`, public `graph`, `done_event`, `stream=`, and a deep-copied read-only `config`.
    - Deleted: `_CAPTURE_LOCK` (runtime.py:34), the 3x warm-up loops, the nested ScopedCapture and the trailing `wp.synchronize`.
    - Self-capture uses a private per-Program stream with `force_module_load=False`, then `capture_launch(graph, stream=ctx.stream)`.
  - GateGeneratorSpec gains `capturable`; False maps to HOST.
  - Course (course.py:317-604):
    - `prepare()` at construction runs the first eager generation and `_build_subtools()`, so its readback sizing stays cold.
    - `regenerate()`/`enqueue_generate()` = generator enqueue + `_refresh()` as ONE body (Graph A and B merged). `generate(seeds=)` becomes a legacy wrapper.
    - The int reseed no longer allocates (course.py:481-482) and raises under capture.
  - `result._owner` and `_compat.anchor`.
  - The legacy-ABI stream rule (G2).
- **Fixes:** CRITICAL generation APIs cannot be recorded by a caller; CRITICAL Course.generate cannot be recorded; HIGH mutable config held by reference (snapshot); MEDIUM _CAPTURE_LOCK process-wide; MEDIUM undocumented 4x warm-up execution; MEDIUM GateGeneratorSpec has no capturable opt-out; LOW warmup x3 + force_module_load; LOW no warm-up API; LOW load_module warm-up unreliable at 1.14; LOW no lifetime anchor (part); LOW the "cuda" alias not canonicalized.
- **Test gate (GPU, `tests/test_caller_capture.py`):**
  - Every generator × {`wp.ScopedCapture`, `ScopedCapture(external=True)`, default `torch.cuda.graph` (adoption), torch + `interop.torch.registered`}: poisoned replay equals the eager twin with the mempool disabled.
  - Course track{segments, sdf} and gates{posts, frames}: step+reset+regenerate in one graph.
  - Negative and contract checks:
    - `generate()` inside a capture never calls `capture_launch` (monkeypatched);
    - a HOST Program under capture raises NotCapturableError before any launch;
    - 2 threads × 2 streams × 100 iterations equal a serial run;
    - the Python body runs twice on the first CUDA call and once per CPU call.
- **Risk:** medium.
  - First-call execution count goes from 4 to 2.
  - `generate()` is no longer host-blocking; `sync=True` is opt-in.
  - Code peeking at `_graph`/`_refresh_graph` breaks.
  - The blocking private capture stream can be invalidated if another thread uses the legacy default stream during the one-time capture; retry once and document.
- **Depends on:** PR2.

## Phase 2: replay correctness and device inputs

### PR4: SeedState, epoch, advance(mask), generation counter, torch interop
- **Scope:**
  - `track_gen/seeds.py`: an owned or pointer-bound seed array, an epoch array, and `set`/`advance`/`reseed_range`.
  - `_effective_seed_k` as the first recorded launch of every Program. It replaces the host `wp.copy(_seed_buf, rng.seeds_warp)` at track_generator.py:188 and gate_generator.py:166.
  - `Track/GateSequence.generation` and `Program.inputs['regen_mask']`.
  - PerEnvSeededRNG:
    - `set_seeds_warp` mirrors `_new_states` (rng_utils.py:78-93);
    - `out=` on every `sample_*` (rng_kernels.py:136, 274, 489, 539, 753, 804, 1018);
    - the library never writes into caller seed arrays.
  - `track_gen/interop/torch.py`: `registered()` and `retain()`.
- **Fixes:** no device-side reseed/advance (replaying a caller graph regenerates identical tracks); no regeneration signal (counter); set_seeds_warp _new_states desync; RNG sampling allocates per call; RNG aliasing and writing into caller arrays; whole-batch-only regeneration (partial).
- **Test gate:**
  - Goldens byte-identical at epoch 0.
  - A caller graph [advance(mask), enqueue] replayed 5x gives 5 distinct batches, each equal to an eager twin advanced identically.
  - Unmasked envs compared bitwise. This is expected to pass but is not guaranteed, because `offset` uses float atomics (warp_pipeline.py:91-92); if it fails, it stays xfail until PR10's masked commit.
- **Risk:** low.
- **Depends on:** PR3.

### PR5: Truncation fails closed
- **Scope:**
  - `_cs_scan_k` (warp_pipeline.py:377-401) writes `truncated`/`required_count` before clamping.
  - `_validity_k` (625-712) ANDs `trunc_ok`.
  - New `on_truncate={'invalidate','flag'}`.
  - New fields `Track.truncated` and `Track.required_count`; the remaining readback is deleted.
  - Doc fixes: types.py:470, docs/how-it-works/resample.rst:80, tuning.rst:66-71, and the README/quickstart defaults that truncate (hw=0.03 with N_max=384).
  - `tests/test_gate_track_compat.py:18` gets an explicit N_max.
- **Fixes:** N_max truncation silent on CUDA; truncated tracks pass validity; README in the truncating regime; no truncation test; yield sweep measures truncated tracks.
- **Test gate (CPU and GPU replay):**
  - A forced-truncation config gives `valid=0, truncated=1`.
  - `flag` keeps `valid` and sets `truncated`.
  - Goldens byte-identical for non-truncating configs.
- **Risk:** medium. Deliberate output change #1: yield drops for configs that were shipping clipped tracks. Escape hatch: `on_truncate='flag'`.
- **Depends on:** PR1.

## Phase 3: public plugin surface v1 (parallel track)

### PR6: One Registry with entry points
- **Scope:**
  - New `track_gen/plugin/__init__.py`:
    - Registry with typed kinds, `replace=False`, `unregister` and `scoped()`;
    - `describe()` stub and `API_VERSION`;
    - lazy, locked, idempotent loading: `track_gen.stages.register(reg)`, then `entry_points(group="track_gen.plugins")`;
    - quarantine policy `TRACK_GEN_PLUGINS=warn|strict|off`.
  - `generator_registry.py` and `gate_generator_registry.py` become shims. The old `register` maps to `replace=True` plus a DeprecationWarning when it replaces a built-in.
  - Built-in modules move from import-time `register(...)` to `register(reg)` functions.
  - Tests: `test_gate_generator_registry.py:99` changes from an exact list to a superset; `test_generator_registry.py:62-69` uses `scoped()`.
- **Fixes:** HIGH registries private, hard-coded, no entry points; MEDIUM registry duplicate/ordering semantics; LOW duplicate detection keyed on `__module__`; part of MEDIUM every shipped generator edited 5-10 shared files.
- **Test gate (CPU):**
  - A fake distribution (monkeypatched `entry_points`) registers `toy.circle`, and TrackGenerator runs it.
  - A duplicate registration raises.
  - A broken plugin is quarantined and named in lookup errors; strict mode raises.
  - In a fresh interpreter, `import track_gen` still loads neither torch nor scipy (test_public_api.py:39-67).
- **Risk:** low.
- **Depends on:** PR0.

### PR7: Public `track_gen.ops`; port and deduplicate the built-ins; scratch hygiene
- **Scope:**
  - `track_gen/ops/{fill,resample,geometry,sampling,curves,reduce,rng,funcs,check}.py`, all taking ctx first, plus Scratch types. `resample_constant_spacing` becomes NaN- and length-aware.
  - Built-in generators and gate generators import ops instead of `_pipe._*` and each other's underscore kernels (warp_generate_polar_gates.py:6-11, voronoi_gates:6-11, checkpoint_gates:6-10). The GateGenConfig-as-TrackGenConfig call at warp_generate_gates.py:38 is removed.
  - Deduplicate the grid sampler, count sampler, angle sort, Catmull-Rom (3 copies), polygon fallback (3) and extent normalization (3).
  - Delete `_Scratch.__getattr__` (warp_pipeline.py:1480-1495); `inflate_warp` reads `scratch.inflate.*` explicitly.
  - Move `GenScratch` into warp_generate. Keep the alias and the PEP 562 re-export (1953-1968) with a DeprecationWarning.
  - `tests/test_legacy_plugin_abi.py` pins the names and `inspect.signature` of `_pipe._init/_sync/_fill_i32_k/_arc_resample_inplace/_arc_resample_selected_inplace/self_intersections_inplace` for 0.x.
- **Fixes:** MEDIUM generators import private warp_pipeline internals with an undocumented sync rule; MEDIUM reusable building blocks private, duplicated; MEDIUM _Scratch.__getattr__ hijack; LOW bezier privileged in the orchestrator; LOW _Scratch recursion on copy/pickle; LOW allocate-if-None paths; LOW private ABI stable by accident.
- **Test gate:**
  - Goldens byte-identical. Merges that would change draw order are deferred to PR16.
  - Every op passes the no-sync and LaunchTrace probes.
  - A plugin scratch with fields `w`/`kappa`/`z` no longer shadows pipeline buffers.
  - The ABI pin passes.
- **Risk:** low-medium (pure refactor).
- **Depends on:** PR2.

### PR8: Stage protocol for generators, Params (static/dynamic), legacy mapping, describe()
- **Scope:**
  - New `track_gen/plugin/{stage,params}.py`: Stage, StageBuilder (generator subset: `write`, `scratch`, `state`, `param`, `input`, `salt`, `seed`), CaptureClass, `static`/`dynamic`, and Recipe.
  - A built-in generator is now a stage that wraps its kernels, with a `Params.from_legacy` mapper. The legacy 5-arg adapter stays.
  - `legacy.to_pipeline_spec`, at generator granularity.
  - Generator-specific validation moves out of `TrackGenConfig.__post_init__` (types.py:697-791, including :776) into the stages. The getattr duck-typing goes (warp_generate_polar.py:156,199-201; warp_relax.py:391).
  - `describe()` returns the Params JSON schema plus ui metadata. `viz/param_explorer.py` and `benchmarks/compare_generators.py` enumerate `describe()`.
  - Dead and oracle-only fields warn when set.
- **Fixes:** MEDIUM monolithic TrackGenConfig, no per-generator namespace; MEDIUM generator output contract fixed; MEDIUM no per-call device input channel for generators; LOW duck-typed getattr defaults; LOW registry enumerators crash on subclass configs; LOW param_explorer hard-codes generators; LOW acceptance thresholds global.
- **Test gate (CPU):**
  - Each built-in generator with legacy flat fields is byte-identical to the same generator via StageRef params.
  - A toy plugin's params are validated only when it is selected.
  - The explorer renders the toy's controls from metadata.
- **Risk:** low-medium. Validation timing moves to construction.
- **Depends on:** PR3, PR6, PR7.

### PR9: `track_gen.testing` conformance kit and plugin-author docs
- **Scope:**
  - `track_gen/testing/{contract,trace,capture,conformance,pytest_plugin}.py`: `check_generator`, `check_stage`, `check_component`, `assert_caller_capturable`, contract guards, LaunchTrace with a pointer allowlist, mempool-off capture, poisoned replay, batch-independence checks, undeclared-write detection.
  - Register a pytest11 plugin.
  - Stage checks from PR8: snapshot each `b.param()` / `b.input()` working array before `enqueue` and fail on any write.
  - Collapse the duplicated per-generator contract tests (test_generate_polar.py:42-112, test_generate_checkpoint.py:56-102, test_generate_repulsive.py:224-252, …) into a suite parametrized over `registry.available()` that reads public outputs only.
  - Rewrite `docs/contributing/writing-a-generator.rst` against the public API. Add guides for writing a gate generator, a stage and a component, each documenting the execution model.
  - Rewrite `docs/how-it-works/cuda-graph.rst` and `docs/tutorials/cuda-graph-in-a-sim.rst` around G1-G13, with Warp, sim and torch recipes.
- **Fixes:** MEDIUM no shipped conformance suite; MEDIUM output contract undocumented; LOW no mechanical check for syncs/allocations; LOW per-env determinism and statelessness untested; LOW contributing docs describe in-tree edits only; LOW docs misstate capture/lock contracts; LOW the CUDA-graph tutorial only shows self-capture.
- **Test gate:**
  - The kit passes for every built-in on both lanes.
  - Five deliberately broken toys are each rejected with a precise message: one syncs, one allocates, one emits NaN, one has data-dependent topology, one has a batch-dependent result.
- **Risk:** low.
- **Depends on:** PR8.

## Phase 4: open pipeline

### PR10: Planner and track-pipeline decomposition; masked commit
- **Scope:**
  - New `track_gen/pipeline/{spec,planner,presets,legacy}.py`: PipelineSpec, StageRef with ports, and the full StageBuilder (read/write/temp_like/carried/fail_bit).
  - Split the legacy track stage into registered stages under `track_gen/stages/`:
    - `resample.constant_spacing`;
    - `relax.{xpbd,none}` (XPBD scratch via `b.scratch`; the `relax_solver` assert at track_generator.py:76-83 becomes a registry lookup);
    - `frames`, `width.constant`, `offset`;
    - `elevation.{flat,uniform,random_walk,noise}`, each with its `parameterization` (deletes the if/elif at warp_zprofile.py:202-222 and the name tuples at warp_pipeline.py:1505, 1899);
    - `validity.default` (fused `_validity_k`) plus the split checks;
    - `commit`: the only writer of the result; masked; generation; `on_invalid` with keep/poison. *Shipped in PR10b as the "hold / restore" form instead (byte-identical PR9 launch sequence): the stages write the `Track`, a masked generation holds the unmasked rows and the commit restores them (they are transiently overwritten on the generation's stream); the sole-writer commit remains an option for 1.0.*
  - `_run_pipeline`/`inflate_warp`/`_Scratch` are deleted once parity holds.
  - `Program.tap()` for viz and benchmarks.
  - `legacy.to_pipeline_spec` carries `cfg.num_points` as the generator stage's raw default (see the PR8 commitments under §5).
  - Remove the PR8 live -> work latch of dynamic params and inputs (the masked commit makes it unnecessary), or record that PR13's `randomize.uniform` writes the WORK arrays: the latch runs before every stage, so a randomize stage writing the LIVE arrays would reach the generator one generation late. One of the two must hold before PR13's `randomize.uniform` lands.
- **Fixes:** HIGH stage sequence hard-coded (relax/inflate/validity not pluggable); HIGH z-profile set closed; MEDIUM validity monolithic single bit; MEDIUM invalid envs leave finite geometry (policy available); LOW benchmarks/viz reach private scratch; LOW no eager CUDA benchmark mode (`policy='eager'`); whole-batch-only regeneration (masked commit makes it exact).
- **Test gate:**
  - Goldens byte-identical per stage, diffing every Track field against the PR9 baseline on CPU and CUDA.
  - Toy relax, elevation and validity plugins installed via entry points run under caller capture.
  - Masked regeneration leaves unmasked envs bitwise unchanged.
  - The launch-count and time delta in benchmark_pipeline is recorded.
- **Risk:** medium-high (core refactor). Contained by the per-stage diffs and the PR9 kit.
- **Depends on:** PR4, PR9.

### PR11: Gate pipeline stages, ordering registry, Stations adapter
- **Scope:**
  - Existing gate generators become `output='anchors'` stages.
  - New stages: `order.{raw,ccw,random_pairs}`, applied by the pipeline; `gate_relax`; `gate_arclen`; elevation reused through port rewiring; `lift`; `tangents`; `gate_frames`; `validity.gate_default`.
  - `provides` (z/tangent/half_size/roll) skips stages; `capacity`/`min_gates` replace the overloaded `max_gates` (gate_generator.py:71-83).
  - New `stations` adapter (and `relaxed_stations`).
  - `spacing_metric` xy/xyz.
  - Deleted: `_GateStaging`, `finish_ordered_gates`, `order_points` dispatch and the positional 6-tuple (warp_gate.py:599-648, 777-864).
- **Fixes:** HIGH gate contract private and coupled to track internals; HIGH gate contract is 2D anchors only and the pipeline overwrites z/tangent/orientation/size; MEDIUM gate stage sequence and closed-set dispatch; MEDIUM ordering delegated and closed; MEDIUM GateGeneratorSpec undocumented, max_gates overloaded; LOW positional 6-tuple scratch; LOW gate_width doc.
- **Test gate:**
  - Goldens for all five gate generators byte-identical; today only bezier and hull are capture-tested.
  - Superformula gates via `stations` work under caller capture.
  - A toy helix that provides z and tangent skips those stages.
- **Risk:** medium. The preset must reproduce today's call order (ordering, then normalization).
- **Depends on:** PR10.

### PR12: BOUNDED tier, repulsive capturable, Segmented HOST
- **Scope:**
  - `ctx.bounded_loop`, `ctx.skip_unless`, `ops.any_set/all_set`, `HostOps` (pinned buffers from `b.host_buffer`).
  - Port repulsive to a Stage (warp_generate_repulsive.py:725-869):
    - drop the readback and break (845-846);
    - make freeze absorbing: the periodic and final resample/copy (835-838) gated via `ops.arc_resample(active=not_frozen)`;
    - stage upsample via ops;
    - `capture=BOUNDED`, `first_check=144`, `check_every=16`.
  - Segmented execution of HOST segments in `run()`.
  - `require_caller_capturable`.
  - Fix the docstring at :10-16.
- **Fixes:** CRITICAL repulsive cannot be graph-captured; MEDIUM capturable is all-or-nothing; LOW capturable=False holds the global lock for seconds; LOW repulsive results depend on batch composition; LOW capture_while port not a drop-in (sidestepped); LOW capture_while fails inside non-Warp captures (unroll fallback); LOW stale capture docs.
- **Test gate:**
  - Lowerings (a), (b) and (c) are bitwise equal on CUDA 12.4+.
  - Repulsive replay equals eager under ScopedCapture and torch default capture.
  - Env e is identical at E=8 and inside E=64.
  - Fewer than 5k nodes at defaults.
  - A HOST toy runs eagerly with a captured tail, and raises under caller capture.
- **Risk:** medium. Deliberate output change #2: frozen envs are no longer re-resampled. About 4.1k nodes: tens of ms to instantiate once, a few MB.
- **Depends on:** PR10.

## Phase 5: channels and results

### PR13: Per-point channels, width registry, attributes, Layout/validate/view, protocols
- **Scope:**
  - `raw.*` to `cl.*` carry via `ops.resample_channels` on the resample tables, re-interpolated at the uniform resample.
  - Width stages:
    - registry: `width.{constant,from_channel}`;
    - XPBD (warp_relax.py:384-387, 551), offset (warp_pipeline.py:52-131) and the thickness/w_floor checks read `cl.half_width`;
    - the `w_floor` tautology at warp_pipeline.py:1869 becomes a real check.
  - Track and results:
    - `Track.half_width`, `attributes`, `layout`, `validate`, `view`, `copy_from`; field-iterating `clone`; `out=` on facades.
    - Capability protocols (HasCenterline, HasBand, HasGates). CourseLine produces a Centerline (course_line.py:132-143).
    - Utilities call `validate()` at construction instead of inferring the stride: props.py:308, checkpoints.py:229, collision.py:318, localize.py:255, heightfield.py:119.
  - New stages:
    - the `randomize.uniform` stage (requires the PR8 latch to be gone, or writes the work arrays: see PR10);
    - the `generator/external` ingestion stage;
    - the `closed` flag plumbed through relax, elevation and utility kernels; only `closed=True` is accepted in 0.x.
- **Fixes:** HIGH constant half-width and 2D-only output with no attribute channel; MEDIUM Track/GateSequence fixed schemas; MEDIUM utilities infer layout from one field; MEDIUM Track used as universal representation; MEDIUM all knobs are host scalars (dynamic params plus randomize); MEDIUM no public centerline-to-Track ingestion; MEDIUM mixed batches/caller-owned outputs (out=/view); LOW w_floor tautology; LOW clone allocates; LOW style sampling bezier-only.
- **Test gate:**
  - Goldens byte-identical at constant width.
  - A variable-width toy has its overlap detected by validity.
  - A hand-built Track with a mismatched field length is rejected.
  - `out=track.view(...)` writes into a shared buffer under caller capture.
- **Risk:** medium-high. The relax kernels are numerically sensitive; constant width stays bit-exact.
- **Depends on:** PR10.

## Phase 6: components and Course

### PR14: Component protocol and adapters
- **Scope:**
  - `track_gen/tools/__init__.py`: Component, ComponentContext, InputSpec.
  - Adapters for CollisionChecker (public `.method`), DiscChecker (built with count and valid; fixes course.py:551), FrameChecker, PropSampler, CheckpointSampler, ProgressTracker, TrackLocalizer, HeightFieldBaker and `centerline_from_gates`. Every hot method takes ctx; the legacy names stay.
  - Static capacities from layout and plan, replacing the readbacks at checkpoints.py:258-266 and props.py:348-350.
  - Generation-aware self-reset in progress and localize kernels.
  - One sentinel table.
  - `out=` for `curvature()`/`speed_profile()` (localize.py:476, 485, 632).
- **Fixes:** MEDIUM no common utility protocol; MEDIUM utilities disagree on invalid envs; MEDIUM no regeneration signal (self-reset); LOW sampler capacities by host readback; LOW public helpers allocate per call; LOW rebind/stale-mask hazards (bind_epoch warning).
- **Test gate:** `check_component` passes for every adapter; the eager-vs-captured trajectory twin test passes in track and gates modes; the invalid-env sentinel tests pass.
- **Risk:** medium.
- **Depends on:** PR13.

### PR15: Course as composition root; in-graph regeneration
- **Scope:**
  - `Course(generator, components)`; presets `Course.track/gates/from_config`; `StepResult` as a Mapping with `.valid` and the legacy properties; `bind(**arrays)` routed by declared inputs.
  - `regenerate(mask)` uses `skip_unless(any_set(mask))` and has a stable `inputs.regen_mask`.
  - Course anchors and retains its components.
  - The explorer and benchmarks consume `describe()`/`tap()`.
- **Fixes:** MEDIUM Course hard-codes two families and a fixed tool set; MEDIUM Course cannot be given a generator or RNG; LOW whole-batch-only regeneration (in-graph auto reset); LOW CourseConfig monolithic; LOW bind hard-codes one agent model; LOW gate collision hard-wired.
- **Test gate (GPU):** an RL-style graph [physics stub, `course.step()`, done mask, `course.regenerate(mask=done)`] replayed 200 steps equals an eager twin in both modes, under ScopedCapture and torch capture. A toy CurvatureLookahead plugs in with no private access.
- **Risk:** medium (Course API). `from_config` and `generate(seeds=)` are kept through 0.x.
- **Depends on:** PR12, PR14.

### PR15b: Documentation restructure and migration guide
- **Scope:**
  - Restructure the doc site around tasks and extension instead of the prototype's internals:
    - **Getting started:** install (Warp ≥ 1.18; driver R580+ and sm_75+ for the PyPI wheels) and a presets-based quickstart.
    - **User guide:** generating tracks and gates (presets, `PipelineSpec`, `Params`); `Course` and the runtime utilities; CUDA graphs in a sim loop (caller capture, streams, torch); RL auto-reset with masked regeneration; choosing a generator; collision backends with their accuracy and cost.
    - **Extending:** generators, gate generators, stages (relax/width/elevation/validity), components, and testing a plugin with `track_gen.testing`.
    - **Reference:** autodoc of the public modules and the configuration reference.
    - **Internals:** the current how-it-works pages (resample, XPBD, inflation, validity), kept and updated.
  - A migration guide from 0.1 to 0.2 covering:
    - flat config → `Params` / presets;
    - `set_capturing` deprecated;
    - the `generate()` semantics (`stream=` / `sync=` / `enqueue()`, warm-up at construction);
    - truncation now invalidating tracks;
    - the repulsive output change;
    - the SDF collision change (option C) and the retraining it implies;
    - Warp ≥ 1.18;
    - the private-registry shims.
  - Promote `pipeline-preview.rst` into the user guide once the pipeline API leaves preview, and fold the per-PR notes into the new structure.
  - Regenerate stale figures and benchmark tables: the repulsive metrics row, the README batch figure, the collision benchmarks.
  - Check code snippets in CI (sphinx doctest or executed examples), so examples cannot go stale. Keep `sphinx -W` in CI.
  - Keep internal planning pages (`docs/superpowers`, `docs/library-readiness`) out of the published site.
- **Fixes:** docs organized around the prototype's internals; tutorials built on the flat 107-field config; preview-page sprawl; no migration guide; stale figures and benchmarks; unchecked code snippets.
- **Test gate:** `sphinx -W`, the snippet checks and a link check pass in CI; every public symbol is documented.
- **Risk:** low (docs only). PR16's 1.0 default changes get a final migration-guide pass in PR16.
- **Depends on:** PR15, so the public API is stable. It must land before `v2` merges into `main`, which is when its docs get published.

## Phase 7: 1.0

### PR16: 1.0 cleanup
- **Scope:**
  - Built-ins switch to `ops.rng.init` with hashed salts (deliberate change #3; goldens re-baselined once; `legacy_rng=True` kept for one minor release).
  - `on_invalid='poison'` becomes the default (change #4).
  - Configs become `frozen=True`; Track/GateSequence new fields become required.
  - Remove the shims: `set_capturing`, the `_CAPTURING` names, the `ctx=None` shims, the PEP 562 re-export, the legacy registries and 5-arg adapter, the dead config fields.
  - Floor raised to `warp-lang>=1.16`; conditional lowerings on by default where supported; CPU APIC self-graph opt-in.
  - MixtureProgram and open topology are scheduled after 1.0.
- **Fixes:** HIGH mutable configs (complete); LOW RNG salt collapse at seed 0 and ad-hoc salts; LOW floor admits versions missing capture fixes; LOW CPU APIC gating; leftover deprecated surface.
- **Test gate:** the full suite at the new floor and at latest on both lanes; goldens re-baselined in one commit.
- **Risk:** breaking (major). Every removal was preceded by a warning release.
- **Depends on:** all.

## Dependency summary

- **Capture track:** PR0 → PR1 → PR2 → PR3 → PR4 → PR10 → {PR11, PR12, PR13} → PR14 → PR15 → PR15b (docs) → PR16.
- **Plugin track:** PR0 → PR6, then PR2 → PR7, then {PR3, PR6, PR7} → PR8 → PR9 → PR10.
- **Independent:** PR5 after PR1.
- **First user-visible milestone:** caller capture of every generation API (except repulsive) and of every utility after PR3. Out-of-tree generators without private imports after PR8.

---

# Open questions

1. Several claims need hardware verification; no GPU was available for this analysis:
(a) adopting the torch capture stream via `stream_from_torch(torch.cuda.current_stream())` records library launches into torch's graph in torch's default 'global' error mode;
(b) `capture_launch(graph, stream=caller)` of a graph captured on a private blocking stream performs as expected;
(c) disabling the mempool during a capture reliably turns allocations into errors at Warp 1.14;
(d) the windowed `capture_if` lowering of repulsive (8 IF nodes) runs correctly at the floor;
(e) instantiation time and memory of the roughly 4.1k-node repulsive graph at E=8192.
2. Which Warp versions do Isaac Sim, Isaac Lab and Newton currently bundle? The answer decides whether the 1.16 floor can land at 1.0 or must wait, and whether conditional lowerings can be on by default.
3. Should we ask Warp upstream for non-blocking stream creation (`wp.Stream(..., blocking=False)`) or a RELAXED-mode recommendation? This would remove the risk that another thread's use of the legacy default stream invalidates the one-time self-capture on our private CU_STREAM_DEFAULT stream.
4. Is setting an attribute on a `torch.cuda.CUDAGraph` instance a robust lifetime anchor for `interop.torch.retain()` across torch versions, or must users hold a returned handle?
5. Which built-in params can safely become dynamic? `half_width` and `spacing` drive the constant-spacing count and N_max capacity. Should dynamic fields that affect capacity require declared `hi` bounds, so capacities are computed from the worst case at build?
6. Topology granularity: is a batch-static `Layout.closed` enough (mixed batches via env-slice views or a future MixtureProgram), or are per-env closed flags needed? What is the post-1.0 order for rolling open topology through relax, elevation and utility kernels?
7. Attribute semantics through relaxation: keep channels attached to bead index across XPBD (proposed), or re-interpolate by arc fraction afterwards? Segment labels and continuous attributes (bank, speed limit) may need different policies.
8. Low-yield generators (checkpoint has about 2.3% invalid at E=512): is regenerate-with-`advance(mask=1-valid)` enough, or is a candidate axis (E*K front half plus `select_first_valid` gather) worth its memory?
9. Invalid-env defaults at 1.0: confirm poison by default, and the progress sentinel (frozen with dist_to_next=0, or NaN). Should StepResult expose per-component invalid flags or only the source `valid` alias?
10. Should built-ins eventually be decomposed into anchors+curve Recipes to deduplicate samplers (plugin-first's model), given the golden churn and call-order reproduction risk?
11. Plugin ABI for `ops.funcs` `@wp.func` signatures that plugin kernels compile against: freeze them within a major version with hash-pinned ABI tests, or version the namespace (`ops.v1`)?
12. Third-party naming: require dotted `pkg.name` (error) or only warn? Should built-ins also get namespaced aliases (`track_gen.bezier`)?
13. Multi-GPU: one Course/Program per device composed by the caller (assumed), or a device-sharded Layout?
14. Per-step performance: once components are composable, should Course offer fused multi-component step kernels (progress + collision + localize) to cut launches inside sim graphs, even though they cross the plugin boundary?
15. MixtureProgram after 1.0: which tail params must agree across parts (n_max, relax solver, width kind), and can per-slice values ride on dynamic [E] params?
