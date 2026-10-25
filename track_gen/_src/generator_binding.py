"""Resolve ``config.generator`` + ``config.generator_params`` into what a facade runs.

:func:`bind_generator` is the one place that looks a generator up, resolves its Params
(``generator_params`` or ``Params.from_legacy(config)``), validates them for the
selected generator only, and builds the two config snapshots a facade keeps:

- ``user_config``: the config AS PASSED, with a dict ``generator_params`` normalised to
  its Params instance (``TrackGenerator.config`` / ``GateGenerator.config``);
- ``run_config``: what the pipeline reads. For a built-in (flat-field Params) the
  resolved fields are written back onto it, so the generator functions keep reading
  their flat fields and outputs stay byte-identical; ``run_config.generator_params``
  is the resolved instance (a legacy spec with ``params=`` reads it there).

A :class:`~track_gen.plugin.Stage` generator is adapted to the legacy accepts-ctx
generator ABI by :class:`_StageGenerator` (``alloc`` through a StageBuilder at build,
``generate`` = the stage's ``enqueue``), so the rest of the track pipeline is unchanged.
A gate generator Stage (``output="anchors"``, roadmap PR11) is bound as is: the planner
builds it like any other stage of the gate pipeline. Binding a legacy
:class:`~track_gen.plugin.GateGeneratorSpec` emits a ``DeprecationWarning`` (once per
registry and name).

Host-only: binding allocates nothing and launches nothing.
"""
from __future__ import annotations

import dataclasses
import numbers
import sys
import types
import warnings
from typing import Any, Mapping

import warp as wp

from ..plugin.params import (NoParams, ParamTable, _check_values, _coerce,
                             _declares_size, _field_hints, _fields, _instance_bounds,
                             _new_param_table, _new_stage_params, _static_view)
from ..plugin.stage import (_FAMILY_RESOLVER, CaptureClass, Layout, _close_builder,
                            _declared_of, _is_stage, _new_builder)
from .runtime import _init, _snapshot_config

_STAGE_ID = "generator"


@dataclasses.dataclass
class GeneratorBinding:
    kind: str                       # "generator" | "gate_generator"
    name: str
    spec: object                    # GeneratorSpec | GateGeneratorSpec | type[Stage]
    params: "Any | None"            # resolved + validated; None: a spec without params=
    user_config: object             # snapshot as passed (dict -> instance)
    run_config: object              # snapshot + built-in overlay; .generator_params = params
    pipeline_spec: object           # what the pipeline calls (the spec, or _StageGenerator)
    capturable: bool                # capture != HOST (may be recorded into a caller's graph)
    accepts_ctx: bool
    layout: Layout
    # The generator's CaptureClass: a Stage's capture_class(static params); a legacy
    # GeneratorSpec / GateGeneratorSpec: STATIC if capturable else HOST.
    capture: CaptureClass = CaptureClass.STATIC
    # Roadmap PR13b: a centerline generator Stage's declared (writes, provides) for its
    # params (Stage.declare); ({}, frozenset()) for a legacy spec.
    writes: "Mapping[str, str]" = dataclasses.field(default_factory=dict)
    provides: frozenset = frozenset()


def _stride(kind: str, config) -> int:
    return int(config.max_gates) if kind == "gate_generator" else int(config.N_max)


def _is_flat(params) -> bool:
    from .builtin_params import _FlatParams  # noqa: PLC0415

    return isinstance(params, _FlatParams)


def _outside_track_gen_stacklevel() -> int:
    """The ``stacklevel`` (for a ``warnings.warn`` in the CALLER of this function) of the
    first frame outside the ``track_gen`` package: the user's call."""
    frame, level = sys._getframe(1), 1
    while frame is not None:
        name = frame.f_globals.get("__name__", "")
        if name != "track_gen" and not name.startswith("track_gen."):
            break
        frame, level = frame.f_back, level + 1
    return level


def _warn_legacy_gate(registry, name: str) -> None:
    """The ``GateGeneratorSpec`` deprecation, once per (registry, name), attributed to the
    user's call (the first frame outside ``track_gen``)."""
    warned = getattr(registry, "_legacy_gate_warned", None)
    if warned is None or name in warned:
        return
    meta = getattr(registry, "_meta", {}).get("gate_generator", {}).get(name)
    origin = str(meta) if meta is not None else "a legacy spec"
    warnings.warn(
        f"GateGeneratorSpec {name!r} ({origin}) is deprecated: write a Stage with "
        f"kind='gate_generator' and output='anchors' "
        f"(docs/contributing/writing-a-generator.rst); legacy gate generators run "
        f"through an adapter until 1.0",
        DeprecationWarning, stacklevel=_outside_track_gen_stacklevel())
    # Marked only once the warning returned: under an error filter every bind raises
    # (Python's own registries do not record a warning turned into an exception).
    warned.add(name)


def _with_flat_size(params, config, spec, kind: str):
    """Roadmap PR13d: the flat ``config.size`` (the INITIAL per-env size) on a plugin
    Stage's Params that declare ``size`` and that ``from_legacy`` left at the class
    default, as the built-ins' ``from_legacy`` maps it (the plugin ``size`` convention);
    the way ``legacy._legacy_params`` maps ``half_width``. Not on
    a gate generator re-normalized by ``normalize.extent``: that stage holds its size.
    Only a field that follows the convention (a dynamic ``float``, ``_declares_size``):
    a static or ``int`` ``size`` is the plugin's own, and ``_check_flat_size`` reports
    the mismatch."""
    fields = getattr(type(params), "__dataclass_fields__", {})
    size = getattr(config, "size", 1.0)
    if not _declares_size(type(params)) or isinstance(size, bool):
        return params
    try:
        if float(size) == 1.0:
            return params
    except (TypeError, ValueError, OverflowError):
        return params                    # left to the converters' checks
    if params.size != fields["size"].default or _size_erased(spec, kind):
        return params                    # from_legacy chose its own value
    return dataclasses.replace(params, size=size)


def _size_erased(spec, kind: str) -> bool:
    """A gate generator whose gates ``normalize.extent`` re-normalizes: the
    normalization erases a size of its own (the flat size goes to ``normalize.size``)."""
    if kind != "gate_generator":
        return False
    from ..stages.gates import _preferred  # noqa: PLC0415 - import cycle

    return _preferred(spec).get("normalize") == "extent"


def bind_generator(kind: str, config, *, registry) -> GeneratorBinding:
    """Resolve and validate ``config.generator`` / ``config.generator_params``.

    Raises (before the facade allocates anything): the registry's unknown-name /
    quarantine errors; ``TypeError`` for a wrong Params type; ``ValueError`` for
    unknown ``generator_params`` keys, a flat / ``generator_params`` conflict, a
    generator-specific rule (``Params.validate``) or a lo/hi/choices violation.
    """
    from ..plugin._registry import _params_class  # noqa: PLC0415 - import cycle

    spec = registry.get(kind, config.generator)
    stage = _is_stage(spec)
    user_config = config
    if kind == "gate_generator" and getattr(config, "gate_ordering", "") is None:
        # GateGenConfig.gate_ordering=None: the generator's default order. Resolved here,
        # before anything reads the config (a legacy spec's Params.from_legacy included),
        # so no generator code ever sees None; gen.config keeps the config as passed.
        from ..stages.gates import _default_order  # noqa: PLC0415 - import cycle

        config = _snapshot_config(config)
        config.gate_ordering = _default_order(spec, registry)
    # Stage.Params, or the params= class of a GeneratorSpec / GateGeneratorSpec; a
    # duck-typed legacy spec's own ``params`` attribute is not a Params class.
    params_cls = _params_class(spec)
    _init()
    layout = Layout(int(config.num_envs), _stride(kind, config),
                    str(wp.get_device(config.device)))
    given = getattr(config, "generator_params", None)
    params = None
    if params_cls is not None or given is not None:
        params_cls = params_cls or NoParams
        label = kind.replace("_", " ")
        params = _coerce(params_cls, given, what=f"{label} {config.generator!r}", cfg=config)
        flat = _is_flat(params)
        if given is None and stage and not flat:
            params = _with_flat_size(params, config, spec, kind)
        if flat and given is not None:
            params._check_no_flat_conflict(config)
        params.validate(layout)
        if flat:
            params._check_legacy_config(config)
        _check_values(params)

    user = _snapshot_config(user_config)
    if given is not None:
        user.generator_params = params
    run = _snapshot_config(config)
    if params is not None:
        if _is_flat(params):
            params._apply_legacy(run)
        from ..plugin.params import _declares_size  # noqa: PLC0415

        size_field = getattr(type(params), "__dataclass_fields__", {}).get("size")
        # Only a size that follows the per-env convention (a dynamic float) is ever
        # written: a static or int size is the plugin's own field.
        if given is None and size_field is not None and _declares_size(type(params)) and \
                params.size != size_field.default and _size_erased(spec, kind):
            # Roadmap PR13d: a from_legacy that mapped the flat size onto a size the
            # normalization erases; the run config keeps it (normalize.size's initial
            # value), the generator's own stays at its default (one key per array).
            params = dataclasses.replace(params, size=size_field.default)
        run.generator_params = params

    if stage and kind == "gate_generator":
        # An anchors Stage: the planner builds it as a stage of the gate pipeline (its
        # capacity is checked there); only the Params and the capture class here.
        cap = spec.capture_class(_static_view(params))
        if not isinstance(cap, CaptureClass):
            raise TypeError(f"{spec.__name__}.capture_class() must return a CaptureClass, "
                            f"got {cap!r}")
        return GeneratorBinding(kind, str(run.generator), spec, params, user, run, spec,
                                capturable=cap != CaptureClass.HOST, accepts_ctx=True,
                                layout=layout, capture=cap)
    if stage:
        instance = spec()
        # build / capacity / capture_class see the static fields only (_static_view).
        view = _static_view(params)
        cap = spec.capture_class(view)
        if not isinstance(cap, CaptureClass):
            raise TypeError(f"{spec.__name__}.capture_class() must return a CaptureClass, "
                            f"got {cap!r}")
        capacity = dict(instance.capacity(view, layout) or {})
        unknown = sorted(str(k) for k in capacity if k != "raw")
        if unknown:
            raise ValueError(f"{spec.__name__}.capacity() returned unknown keys {unknown}; "
                             f"a generator Stage accepts ['raw']")
        raw = capacity.get("raw", run.num_points)
        if isinstance(raw, bool) or not isinstance(raw, numbers.Integral) or raw < 3:
            raise ValueError(f"{spec.__name__}.capacity()['raw'] must be an int >= 3, "
                             f"got {raw!r}")
        writes, provides = _declared_of(spec, params)
        adapter = _StageGenerator(instance, spec, params, layout, scale=float(run.scale),
                                  raw_points=int(raw),
                                  capturable=cap != CaptureClass.HOST, capture=cap,
                                  writes=writes)
        return GeneratorBinding(kind, str(run.generator), spec, params, user, run, adapter,
                                capturable=adapter.capturable, accepts_ctx=True,
                                layout=layout, capture=cap, writes=writes,
                                provides=provides)
    if kind == "gate_generator":
        _warn_legacy_gate(registry, str(run.generator))
    capturable = bool(getattr(spec, "capturable", True))
    return GeneratorBinding(kind, str(run.generator), spec, params, user, run, spec,
                            capturable=capturable,
                            accepts_ctx=bool(getattr(spec, "accepts_ctx", False)),
                            layout=layout,
                            capture=CaptureClass.STATIC if capturable else CaptureClass.HOST)


def param_table(binding: GeneratorBinding, *, default_stream=None,
                drawn=None, hints=None) -> ParamTable:
    """``{"generator": StageParams}`` of a bound generator (live dynamic arrays for a
    Stage generator, none for a legacy spec); ``drawn``: ``field -> randomizer id`` of
    the fields a ``randomize.*`` stage draws. Roadmap PR13d: the arrays the stage reads
    (``in_effect``: the working copies, the drawn arrays), its Params' per-instance bounds
    and field hints (``hints`` adds to them)."""
    adapter = binding.pipeline_spec
    stage = isinstance(adapter, _StageGenerator)
    arrays = adapter.param_arrays if stage else {}
    work = adapter.work_arrays if stage else {}
    layout = binding.layout
    return _new_param_table({_STAGE_ID: _new_stage_params(
        stage_id=_STAGE_ID, params=binding.params, arrays=arrays, num_envs=layout.num_envs,
        device=layout.device, default_stream=default_stream, drawn=drawn,
        bounds=_instance_bounds(binding.params), effective=work,
        hints={**_field_hints(binding.params), **dict(hints or {})})})


# ---------------------------------------------------------------------------------------
# The Stage adapter
# ---------------------------------------------------------------------------------------

# The latch is a kernel in both forms (not a ctx.copy for the unmasked one), so the
# warm-up pass (unmasked) loads this module: a later masked generation recorded into a
# caller's capture then never loads a module while capturing.
@wp.kernel
def _latch_all_f32_k(live: wp.array(dtype=wp.float32), work: wp.array(dtype=wp.float32)):
    e = wp.tid()
    work[e] = live[e]


@wp.kernel
def _latch_all_i32_k(live: wp.array(dtype=wp.int32), work: wp.array(dtype=wp.int32)):
    e = wp.tid()
    work[e] = live[e]


@wp.kernel
def _latch_masked_f32_k(live: wp.array(dtype=wp.float32), mask: wp.array(dtype=wp.int32),
                        work: wp.array(dtype=wp.float32)):
    e = wp.tid()
    if mask[e] != 0:
        work[e] = live[e]


@wp.kernel
def _latch_masked_i32_k(live: wp.array(dtype=wp.int32), mask: wp.array(dtype=wp.int32),
                        work: wp.array(dtype=wp.int32)):
    e = wp.tid()
    if mask[e] != 0:
        work[e] = live[e]


_LATCH_ALL = {wp.float32: _latch_all_f32_k, wp.int32: _latch_all_i32_k}
_LATCH_MASKED = {wp.float32: _latch_masked_f32_k, wp.int32: _latch_masked_i32_k}


@dataclasses.dataclass
class _BuiltStage:
    """What a Stage generator's ``build`` produced (``TrackGenerator._scratch.gen``)."""

    stage: object
    builder: object
    enqueue: object
    scratch: Mapping[str, "wp.array"]


class _StageGenerator:
    """Adapts a Stage generator to the accepts-ctx ``GeneratorSpec`` call the pipeline
    makes. Built once (:meth:`build`, by the pipeline's generator slot, or by the PR9
    ``_inflate_warp_alloc``)."""

    accepts_ctx = True

    def __init__(self, stage, spec, params, layout: Layout, *, scale: float,
                 raw_points: int, capturable: bool, capture=None,
                 writes: "Mapping[str, str] | None" = None) -> None:
        self.stage = stage
        self.spec = spec
        self.name = str(spec.name)
        self.params = params
        self.layout = layout
        self.scale = float(scale)
        self.raw_points = int(raw_points)
        self.capturable = bool(capturable)
        self.capture = capture                  # its resolved CaptureClass
        self.inputs: Mapping[str, wp.array] = types.MappingProxyType({})
        # name -> pipeline.InputInfo (latched per="env" / unlatched per="raw" inputs)
        self.input_info: Mapping[str, object] = types.MappingProxyType({})
        self.param_arrays: Mapping[str, wp.array] = types.MappingProxyType({})
        # field -> the array the stage reads (its working copy, or the drawn array):
        # StageParams.in_effect (roadmap PR13d).
        self.work_arrays: Mapping[str, wp.array] = types.MappingProxyType({})
        # The generator's declared writes for its params (Stage.declare).
        self.writes = dict(spec.writes) if writes is None else dict(writes)
        self._pairs: list = []                  # (live, work): the latched arrays only
        self._built: "_BuiltStage | None" = None
        self._seed = self._out_xy = self._out_valid = None
        self._enqueue = None

    def __repr__(self) -> str:
        return f"<Stage generator {self.name!r} ({type(self.stage).__name__})>"

    def alloc_scratch(self, config):
        raise TypeError("Stage generators allocate through their StageBuilder; they run "
                        "through TrackGenerator")

    def build(self, *, seed, out_xy, out_valid, resolve=None,
              drawn=None) -> _BuiltStage:
        """Allocate the dynamic-param arrays, run ``Stage.build`` on a StageBuilder bound
        to the orchestrator-owned outputs, and check its post-conditions. ``resolve``
        (the pipeline's generator slot, roadmap PR13a): ``resolve(channel, dtype, interp,
        export) -> wp.array`` allocates a raw attribute (``raw.z``, ``raw.half_width``,
        ``raw.<pkg>.<name>``) as a pipeline channel; without it (the PR9 reference
        composition) the builder allocates it privately (nothing carries it there).
        ``drawn`` (roadmap PR13b): ``field -> [E] array`` of the dynamic fields a
        ``randomize.*`` stage draws: that single array replaces the field's latch pair
        (unlatched). A field the stage draws itself (roadmap PR13c: bezier's
        ``style_sampling`` ranges, ``Params._self_drawn()``) gets such an array here
        when the planner did not pass one (the PR9 reference composition)."""
        if self._built is not None:
            raise RuntimeError(f"Stage generator {self.name!r} is already built; its "
                               f"buffers are bound at build (construct a new generator)")
        from ..exec.context import LaunchContext  # noqa: PLC0415

        E, dev = self.layout.num_envs, self.layout.device
        drawn = dict(drawn or {})
        self_drawn = getattr(self.params, "_self_drawn", None)
        for name in (self_drawn() if callable(self_drawn) else {}):
            if name not in drawn:
                drawn[name] = wp.full(E, float(getattr(self.params, name)),
                                      dtype=wp.float32, device=dev)
        dynamic = {}
        if self.params is not None:
            for f in _fields(type(self.params)):
                if f.kind != "dynamic":
                    continue
                if f.name in drawn:
                    dynamic[f.name] = (drawn[f.name], drawn[f.name])
                    continue
                value = getattr(self.params, f.name)
                value = float(value) if f.dtype is wp.float32 else int(value)
                dynamic[f.name] = (wp.full(E, value, dtype=f.dtype, device=dev),
                                   wp.full(E, value, dtype=f.dtype, device=dev))
        view = _static_view(self.params)              # dynamic fields raise
        outputs = {"raw.xy": out_xy, "gen.valid": out_valid}
        if resolve is not None:
            outputs[_FAMILY_RESOLVER] = resolve
        b = _new_builder(stage=self.stage, params=view, layout=self.layout,
                         scale=self.scale, seed=seed, raw=self.raw_points,
                         outputs=outputs, dynamic=dynamic, stage_id=_STAGE_ID,
                         capture=self.capture)
        b._declared_writes = types.MappingProxyType(dict(self.writes))
        enqueue = self.stage.build(b, view)
        _close_builder(b)
        # The resolver is a build-time hook (it holds the pipeline's plan): drop it.
        b._outputs.pop(_FAMILY_RESOLVER, None)
        if not callable(enqueue):
            raise TypeError(f"{type(self.stage).__name__}.build() must return an "
                            f"enqueue(ctx) callable, got {enqueue!r}")
        written = b._written_channels
        if "raw.xy" not in written:
            raise ValueError(f"{type(self.stage).__name__}.build() must write 'raw.xy' "
                             f"(b.write(port, wp.vec2f, per='raw'))")
        if "gen.valid" not in written:
            # Nothing else writes it: set it once, at build.
            LaunchContext.current(dev).fill(out_valid, 1)
        self._seed, self._out_xy, self._out_valid = seed, out_xy, out_valid
        self._enqueue = enqueue
        from ..pipeline.program import InputInfo  # noqa: PLC0415 - import cycle

        self.param_arrays = types.MappingProxyType(
            {name: pair[0] for name, pair in dynamic.items()})
        self.work_arrays = types.MappingProxyType(
            {name: pair[1] for name, pair in dynamic.items()})
        self.inputs = types.MappingProxyType(
            {**{name: pair[0] for name, pair in b._inputs.items()}, **b._raw_inputs})
        self.input_info = types.MappingProxyType({
            **{name: InputInfo(name, pair[0].dtype, "env", E, True, _STAGE_ID)
               for name, pair in b._inputs.items()},
            **{name: InputInfo(name, arr.dtype, "raw", int(arr.shape[0]), False,
                               _STAGE_ID) for name, arr in b._raw_inputs.items()}})
        self._pairs = [pair for name, pair in dynamic.items() if name not in drawn] + \
            list(b._inputs.values())
        self._built = _BuiltStage(stage=self.stage, builder=b, enqueue=enqueue,
                                  scratch=types.MappingProxyType(dict(b._scratch)))
        return self._built

    def latch(self, ctx, mask) -> None:
        """Copy the live dynamic params / inputs into the stage's working copies, for
        every env (``mask is None``) or where ``mask[e] != 0``: one launch per pair,
        recorded right after the effective-seed prologue."""
        E = self.layout.num_envs
        for live, work in self._pairs:
            if mask is None:
                ctx.launch(_LATCH_ALL[work.dtype], E, [live, work])
            else:
                ctx.launch(_LATCH_MASKED[work.dtype], E, [live, mask, work])

    def generate(self, seeds_wp, config, out_centerline, out_valid_wp, scratch,
                 ctx=None) -> None:
        from ..exec.context import resolve  # noqa: PLC0415

        if self._built is None or out_centerline is not self._out_xy \
                or out_valid_wp is not self._out_valid:
            raise RuntimeError(f"Stage generator {self.name!r}: its output buffers are "
                               f"bound at build; run it through TrackGenerator")
        ctx = resolve(ctx, self.layout.device)
        if seeds_wp is not self._seed:
            ctx.copy(self._seed, seeds_wp)          # private direct calls only
        self._enqueue(ctx)
