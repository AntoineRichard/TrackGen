"""The :class:`Stage` protocol and its :class:`StageBuilder`.

A Stage is a class that allocates everything it needs ONCE, through the
:class:`StageBuilder` the orchestrator hands to :meth:`Stage.build`, and returns an
``enqueue(ctx)`` callable that emits its device work on a
:class:`~track_gen.exec.LaunchContext`. Register the CLASS with
``registry.add(MyStage)``. Third parties register stages of five kinds:

- ``"generator"`` (``output="centerline"``): the raw closed centerline, selected with
  ``TrackGenConfig(generator=MyStage.name)``;
- ``"relax"``, ``"width"``, ``"elevation"``, ``"validity"`` (roadmap PR10b): a stage of
  the track pipeline, selected with :func:`track_gen.pipeline.presets.track` (or the
  flat shortcuts ``TrackGenConfig.relax_solver`` / ``z_profile``). It reads and writes
  named channels through :meth:`StageBuilder.read` / :meth:`StageBuilder.write`.

The other track stages (resample, frames, offset, lift) are TrackGen's built-ins.

A generator stage writes the raw closed centerline (channel ``"raw.xy"``,
``[E * raw]`` ``vec2f``, ``raw`` from :meth:`Stage.capacity`, default
``TrackGenConfig.num_points``) and optionally a per-env validity flag (``"gen.valid"``,
``[E]`` ``int32``; when it is never written the orchestrator sets it to 1 once).

A pipeline stage declares the channels it reads and writes (``reads`` / ``writes``:
catalogue channels, see :doc:`/how-it-works/pipeline-preview`, or its own namespaced
channels ``cl.<package>.<name>``). Each kind may write one catalogue channel: ``relax``
``cl.xy``, ``width`` ``cl.half_width``, ``elevation`` ``cl.z``, ``validity``
``valid.fail`` (required). A channel the stage also reads (same port) is a rewrite: it
reads the previous version and writes a NEW buffer (never the read one). A validity
stage owns at least one fail bit (:meth:`StageBuilder.fail_bit`) and ORs it into
``valid.fail``.

Gate generator stages (``output="anchors"``) arrive with roadmap PR11; until then gate
generators register a :class:`~track_gen.plugin.GateGeneratorSpec`.
"""
from __future__ import annotations

import abc
import dataclasses
import enum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Mapping

import warp as wp

from .params import NoParams, _fields

if TYPE_CHECKING:  # pragma: no cover
    from ..exec.context import LaunchContext

__all__ = ["CaptureClass", "Layout", "Stage", "StageBuilder"]

# The channels a PR8 generator stage may write: name -> (dtype, per).
_GENERATOR_CHANNELS = {"raw.xy": (wp.vec2f, "raw"), "gen.valid": (wp.int32, "env")}
_STAGE_ID = "generator"
_BUILDER_TOKEN = object()

# The track-pipeline kinds third parties may register (roadmap PR10b), and the one
# catalogue channel each may write (besides its own plugin channels cl.<pkg>.<name>).
_OPEN_KINDS = ("relax", "width", "elevation", "validity")
_KIND_WRITES = {"relax": ("cl.xy",), "width": ("cl.half_width",),
                "elevation": ("cl.z",), "validity": ("valid.fail",)}
# The sizes a plugin channel may have (raw is the generator's).
_PLUGIN_PERS = ("env", "cl")


class CaptureClass(enum.IntEnum):
    """How a stage's ``enqueue`` may be captured.

    - ``STATIC``: a fixed launch sequence, capturable by anything.
    - ``BOUNDED``: a static-bound iteration through ``ctx.bounded_loop``. Reserved: both
      arrive with roadmap PR12, and until then a stage declaring ``BOUNDED`` (or whose
      :meth:`Stage.capture_class` returns it) is rejected; declare ``STATIC``.
    - ``HOST``: host readback / control flow: runs eagerly on every call, and a
      caller capture raises :class:`~track_gen.exec.NotCapturableError` before any
      launch. Exempt from the no-sync rule.
    """

    STATIC = 0
    BOUNDED = 1
    HOST = 2


@dataclasses.dataclass(frozen=True)
class Layout:
    """The run's shape, handed to :meth:`Params.validate`, :meth:`Stage.capacity` and
    the :class:`StageBuilder`.

    Attributes:
        num_envs: ``E``.
        stride: Output row capacity per env (tracks: ``N_max``; gates: ``max_gates``).
        device: Canonical device string (``str(wp.get_device(d))``, e.g. ``"cuda:0"``).
        closed: Whether the output loops are closed (always ``True`` in 0.x).
    """

    num_envs: int
    stride: int
    device: str
    closed: bool = True


class Stage(abc.ABC):
    """Base of a pipeline stage.

    Class attributes a stage sets:

    - ``kind``: the registry kind: ``"generator"``, or a track-pipeline kind
      ``"relax"``, ``"width"``, ``"elevation"`` or ``"validity"``.
    - ``name``: the registered name; third parties use a dotted ``"package.name"``.
    - ``Params``: the stage's :class:`~track_gen.plugin.Params` class (default
      :class:`~track_gen.plugin.NoParams`).
    - ``capture``: its :class:`CaptureClass` (default ``STATIC``; ``BOUNDED`` is
      rejected until roadmap PR12); override :meth:`capture_class` when it depends on
      the params.
    - ``reads``: ``local port -> channel`` the stage reads (:meth:`StageBuilder.read`).
      Must be empty for a generator.
    - ``writes``: ``local port -> channel`` the stage writes
      (:meth:`StageBuilder.write`); a generator writes ``"raw.xy"`` (required) and
      optionally ``"gen.valid"``. A channel in both ``reads`` and ``writes``, through
      ONE port, is a rewrite: ``write`` returns a NEW version of the channel (later
      stages see it), ``read`` the previous one, so the stage writes its whole output
      into the ``write`` buffer and never writes the ``read`` one.
    - ``provides``: must be empty (PR11/PR13).
    - ``topology``: ``"closed"`` or ``"any"``.
    - ``output``: ``"centerline"`` for a generator (``"anchors"`` arrives with PR11);
      ``None`` for the other kinds.
    - ``api_version``: the plugin API version the stage targets (``<= API_VERSION``).

    The orchestrator instantiates the class once per pipeline with no arguments.

    :meth:`capture_class`, :meth:`capacity` and :meth:`build` receive the params with
    their STATIC fields only: reading a dynamic field there raises ``AttributeError``
    (a host scalar taken from it would be baked into the launches and ignore every
    ``set``); read it on the device through :meth:`StageBuilder.param`.
    """

    kind: ClassVar[str]
    name: ClassVar[str]
    Params: ClassVar[type] = NoParams
    capture: ClassVar[CaptureClass] = CaptureClass.STATIC
    reads: ClassVar[Mapping[str, str]] = MappingProxyType({})
    writes: ClassVar[Mapping[str, str]] = MappingProxyType({})
    provides: ClassVar[frozenset] = frozenset()
    topology: ClassVar[str] = "closed"
    output: ClassVar["str | None"] = None
    api_version: ClassVar[int] = 1

    @classmethod
    def capture_class(cls, params) -> CaptureClass:
        """The stage's capture class for ``params`` (default: :attr:`capture`)."""
        return cls.capture

    def capacity(self, params, layout: Layout) -> Mapping[str, int]:
        """Buffer capacities this stage needs. A generator may return ``{"raw": n}``
        (points per env of its ``"raw.xy"`` output, an integer ``n >= 3``); the default
        ``{}`` uses ``TrackGenConfig.num_points``. Any other key raises ``ValueError``."""
        return {}

    @abc.abstractmethod
    def build(self, b: "StageBuilder", params) -> "Callable[[LaunchContext], None]":
        """Allocate everything through ``b``, ONCE, and return ``enqueue(ctx)``.

        ``enqueue`` emits only ``ctx.launch`` / ``ctx.copy`` / ``ctx.fill`` /
        ``track_gen.ops`` work: no allocation, no host sync, no branch on device data
        or on ``ctx.capturing`` (a ``HOST`` stage may sync).
        """


def _fnv1a_32(data: bytes) -> int:
    """32-bit FNV-1a of ``data``."""
    h = 0x811C9DC5
    for byte in data:
        h ^= byte
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def _salt_of(stage_id: str, stream: str) -> int:
    """``hash32_host(fnv1a_32(f"{stage_id}/{stream}"))``: frozen for API_VERSION 1."""
    from ..ops.hash import hash32_host  # noqa: PLC0415 - keep plugin import light

    return hash32_host(_fnv1a_32(f"{stage_id}/{stream}".encode()))


class StageBuilder:
    """Handed to :meth:`Stage.build`; created by the orchestrator, never by plugins.

    This release implements :meth:`read`, :meth:`write`, :meth:`temp_like`,
    :meth:`scratch`, :meth:`param`, :meth:`input`, :meth:`salt` and :meth:`fail_bit`.
    Every method allocates at build time only, and raises once ``build()`` returned.

    Attributes:
        E: The number of envs.
        layout: The run's :class:`Layout`.
        device: The ``wp.Device``.
        stage_id: The stage's id in the pipeline (``"generator"`` for the generator).
        scale: ``TrackGenConfig.scale`` (a core field, not a generator Param).
        seed: ``[E]`` int32: the effective per-env seed of the current generation
            (seed and epoch folded); read it in ``enqueue``.
        raw: Points per env of the ``"raw.xy"`` channel.

    The arrays :meth:`param` and :meth:`input` return are READ-ONLY for the stage:
    never write them (not even as scratch), since a generation's values come from the
    live arrays the caller writes.
    """

    def __init__(self, *, _token=None, stage, params, layout: Layout, scale: float,
                 seed: "wp.array", raw: int, outputs: Mapping[str, "wp.array"],
                 dynamic: Mapping[str, tuple], reserved_inputs=("regen_mask",),
                 stage_id: str = _STAGE_ID, plan=None) -> None:
        if _token is not _BUILDER_TOKEN:
            raise TypeError("StageBuilder is created by the orchestrator (TrackGenerator); "
                            "a Stage receives one in build()")
        self.E = int(layout.num_envs)
        self.layout = layout
        self.device = wp.get_device(layout.device)
        self.stage_id = str(stage_id)
        self.scale = float(scale)
        self.seed = seed
        self.raw = int(raw)
        self._stage = stage
        self._params = params
        self._outputs = dict(outputs)
        self._dynamic = dict(dynamic)                 # field -> (live, work)
        self._static = {f.name for f in _fields(type(params)) if f.kind != "dynamic"} \
            if params is not None else set()
        self._written: dict = {}                      # port -> channel
        self._scratch: dict = {}                      # "scratch:<id>/<name>" -> array
        self._inputs: dict = {}                       # name -> (live, work)
        self._reserved_inputs = frozenset(reserved_inputs)
        self._salts: dict = {}                        # stream -> salt
        self._temps: dict = {}                        # port -> temp_like() count
        # The planner's per-stage view (track_gen.pipeline): resolved ports, channel
        # versions and the channel catalogue. None for a generator Stage built by the
        # PR8 adapter: the generator subset below, unchanged.
        self._plan = plan
        self._closed = False

    def __repr__(self) -> str:
        return (f"StageBuilder(stage={type(self._stage).__name__}, E={self.E}, "
                f"device={self.layout.device!r})")

    def _check_open(self, what: str) -> None:
        if self._closed:
            raise RuntimeError(f"StageBuilder.{what}() after build() returned: allocate "
                               f"everything inside build()")

    @property
    def _written_channels(self) -> frozenset:
        return frozenset(self._written.values())

    def write(self, port: str, dtype=None, *, per: "str | None" = None, interp=None,
              export=None) -> "wp.array":
        """The output buffer of ``writes[port]``.

        A generator writes ``"raw.xy"`` (``wp.vec2f``, ``per="raw"``, ``[E * raw]``)
        and ``"gen.valid"`` (``wp.int32``, ``per="env"``, ``[E]``). ``per`` sizes the
        buffer: ``"env"`` is ``[E]``, ``"cl"`` is ``[E * N_max]``, ``"raw"`` is
        ``[E * raw]``. The buffer is owned by the orchestrator and read directly by
        later stages (no copy). For a channel the stage also reads (a rewrite) it is a
        NEW version: :meth:`read` keeps returning the previous one. A plugin channel
        (``cl.<package>.<name>``) takes the ``dtype`` / ``per`` (``"env"`` or ``"cl"``)
        of its first writer; a later writer must match.

        ``dtype`` and ``per`` may be omitted for a catalogue channel, or a plugin
        channel an earlier stage wrote: they are then the channel's (``b.write("z")``).
        Given, they must match it. A plugin channel's first writer passes ``dtype``
        (``per`` defaults to ``"env"``).

        Raises:
            ValueError: an undeclared port (lists the declared ones), a channel the
                stage may not write, or a second write of the same port.
            TypeError: a ``dtype`` / ``per`` that does not match the channel.
            NotImplementedError: ``interp=`` / ``export=`` (roadmap PR13).
        """
        self._check_open("write")
        if interp is not None or export is not None:
            raise NotImplementedError("StageBuilder.write(interp=, export=) arrives with "
                                      "roadmap PR13")
        if self._plan is not None:
            return self._plan_write(port, dtype, per)
        writes = type(self._stage).writes
        if port not in writes:
            raise ValueError(f"stage {type(self._stage).__name__} declares no write port "
                             f"{port!r}; declared ports: {sorted(writes)}")
        channel = writes[port]
        if channel not in _GENERATOR_CHANNELS:
            raise ValueError(f"channel {channel!r} arrives with roadmap PR10/PR13; a "
                             f"generator stage writes {sorted(_GENERATOR_CHANNELS)}")
        if port in self._written:
            raise ValueError(f"port {port!r} ({channel!r}) is already written; call "
                             f"write() once per port")
        want_dtype, want_per = _GENERATOR_CHANNELS[channel]
        dtype = want_dtype if dtype is None else dtype
        per = want_per if per is None else per
        if dtype is not want_dtype:
            raise TypeError(f"channel {channel!r} has dtype {want_dtype.__name__}, got "
                            f"{getattr(dtype, '__name__', dtype)}")
        if per != want_per:
            raise TypeError(f"channel {channel!r} is per={want_per!r}, got per={per!r}")
        self._written[port] = channel
        return self._outputs[channel]

    def _plan_write(self, port: str, dtype, per: str) -> "wp.array":
        plan = self._plan
        if port not in plan.writes:
            raise ValueError(f"stage {self._label()} declares no write port {port!r}; "
                             f"declared ports: {sorted(plan.writes)}")
        channel = plan.writes[port]
        if port in self._written:
            raise ValueError(f"port {port!r} ({channel!r}) is already written; call "
                             f"write() once per port")
        info = plan.channel(channel)
        if info is None:
            # A plugin channel's first writer fixes its dtype and size.
            if dtype is None:
                raise TypeError(f"plugin channel {channel!r}: its first writer passes "
                                f"dtype= (and per=, default 'env')")
            per = "env" if per is None else per
            if per not in _PLUGIN_PERS:
                raise ValueError(f"plugin channel {channel!r}: per must be one of "
                                 f"{list(_PLUGIN_PERS)}, got {per!r}")
            if not _is_warp_dtype(dtype):
                raise TypeError(f"plugin channel {channel!r}: dtype must be a Warp "
                                f"scalar or vector type, got {dtype!r}")
            plan.define(channel, dtype, per)
            info = (dtype, per)
        want_dtype, want_per = info
        dtype = want_dtype if dtype is None else dtype
        per = want_per if per is None else per
        if dtype is not want_dtype:
            raise TypeError(f"channel {channel!r} has dtype {want_dtype.__name__}, got "
                            f"{getattr(dtype, '__name__', dtype)}")
        if per != want_per:
            raise TypeError(f"channel {channel!r} is per={want_per!r}, got per={per!r}")
        arr = plan.write(port)
        self._written[port] = channel
        return arr

    def _label(self) -> str:
        return f"{self.stage_id!r} ({type(self._stage).__name__})"

    def _length(self, per: str) -> int:
        if per == "env":
            return self.E
        if per == "raw":
            return self.E * self.raw
        if per == "cl":
            return self.E * int(self.layout.stride)
        raise ValueError(f"unknown per={per!r}; expected 'env', 'raw' or 'cl'")

    def read(self, port: str) -> "wp.array":
        """The buffer of ``reads[port]``: the latest version of its channel written by
        an EARLIER stage (for a channel this stage also writes, the version before its
        own). READ-ONLY for the stage.

        Raises:
            ValueError: an undeclared read port (lists the declared ones). A generator
                declares none.
        """
        self._check_open("read")
        reads = self._plan.reads if self._plan is not None else {}
        if port not in reads:
            raise ValueError(f"stage {self._label()} declares no read port {port!r}; "
                             f"declared read ports: {sorted(reads)}")
        return self._plan.read(port)

    def temp_like(self, port: str) -> "wp.array":
        """A private zero-initialised buffer with the dtype and length of ``port``'s
        channel (e.g. the ping-pong partner of an in-place rewrite), allocated as
        :meth:`scratch` ``"<port>.tmp<k>"`` (``k = 0, 1, ...`` per port).

        Raises:
            ValueError: ``port`` is neither a read nor a write port of the stage.
        """
        self._check_open("temp_like")
        if self._plan is not None:
            ports = {**self._plan.reads, **self._plan.writes}
            channel = ports.get(port)
            info = None if channel is None else self._plan.channel(channel)
            if channel is not None and info is None:
                raise ValueError(f"stage {self._label()}: temp_like({port!r}): plugin "
                                 f"channel {channel!r} has no dtype yet; call "
                                 f"b.write({port!r}, dtype, per=...) first")
        else:
            ports = dict(type(self._stage).writes)
            channel = ports.get(port)
            info = _GENERATOR_CHANNELS.get(channel)
        if info is None:
            raise ValueError(f"stage {self._label()} declares no port {port!r}; declared "
                             f"ports: {sorted(ports)}")
        dtype, per = info
        k = self._temps.get(port, 0)
        self._temps[port] = k + 1
        return self.scratch(f"{port}.tmp{k}", dtype, self._length(per))

    def scratch(self, name: str, dtype, shape) -> "wp.array":
        """A private zero-initialised buffer, namespaced ``"scratch:<stage_id>/<name>"``.
        Never re-zeroed: write before you read. A duplicate ``name`` raises
        ``ValueError``."""
        self._check_open("scratch")
        key = f"scratch:{self.stage_id}/{name}"
        if key in self._scratch:
            raise ValueError(f"scratch buffer {name!r} is already allocated ({key})")
        arr = wp.zeros(shape, dtype=dtype, device=self.device)
        self._scratch[key] = arr
        return arr

    def param(self, field: str) -> "wp.array":
        """The stage-side ``[E]`` array of a DYNAMIC Params field (float32 or int32).

        This is the working copy, latched from the live array the user writes
        (``gen.params[stage_id][field]``) at the start of every generation, for the
        envs that generation regenerates. READ-ONLY for the stage: never write it (a
        later release reads the live array directly).

        Raises:
            KeyError: ``field`` is static ("read it from params") or unknown (lists the
                dynamic fields).
        """
        pair = self._dynamic.get(field)
        if pair is not None:
            return pair[1]
        if field in self._static:
            raise KeyError(f"{field!r} is a static parameter: read it from params "
                           f"(params.{field}); StageBuilder.param() serves dynamic fields")
        raise KeyError(f"no dynamic parameter {field!r}; dynamic fields: "
                       f"{sorted(self._dynamic)}")

    def input(self, name: str, dtype, default=0) -> "wp.array":
        """A per-env device input: a stable LIVE ``[E]`` array exposed as
        ``gen.inputs[name]`` (filled with ``default``), latched into the returned
        working copy at every generation like a dynamic param. The returned array is
        READ-ONLY for the stage, like :meth:`param`'s.

        Args:
            name: The input's name (``"regen_mask"`` and existing names clash).
            dtype: ``wp.float32`` or ``wp.int32``.
            default: The initial value of every env.

        Raises:
            TypeError: another dtype.
            ValueError: a name clash.
        """
        self._check_open("input")
        if dtype is not wp.float32 and dtype is not wp.int32:
            raise TypeError(f"input {name!r}: dtype must be wp.float32 or wp.int32, got "
                            f"{getattr(dtype, '__name__', dtype)}")
        if not isinstance(name, str) or not name:
            raise ValueError(f"input name must be a non-empty string, got {name!r}")
        if name in self._reserved_inputs or name in self._inputs:
            raise ValueError(f"input {name!r} already exists (reserved: "
                             f"{sorted(self._reserved_inputs)})")
        value = float(default) if dtype is wp.float32 else int(default)
        live = wp.full(self.E, value, dtype=dtype, device=self.device)
        work = wp.full(self.E, value, dtype=dtype, device=self.device)
        self._inputs[name] = (live, work)
        return work

    def salt(self, stream: str) -> int:
        """A ``uint32`` salt for an independent random stream:
        ``hash32_host(fnv1a_32(f"{stage_id}/{stream}".encode()))``.

        Frozen for plugin API version 1, so outputs stay reproducible across releases.
        Key the per-env RNG as ``wp.rand_init(int32(hash32(uint32(seed[e]) ^ salt) &
        0x7FFFFFFF))``. Idempotent per stream; two streams whose salts collide raise
        ``ValueError``.
        """
        stream = str(stream)
        if stream in self._salts:
            return self._salts[stream]
        value = _salt_of(self.stage_id, stream)
        for other, v in self._salts.items():
            if v == value:
                raise ValueError(f"salt streams {other!r} and {stream!r} collide "
                                 f"(0x{value:08x}); rename one")
        self._salts[stream] = value
        return value


    def fail_bit(self, name: "str | None" = None) -> int:
        """The index (16..30) of a new fail bit owned by this validity stage.

        In the kernel: ``fail_out[e] = fail_in[e] | (1 << bit)`` where the check fails
        (``fail_in`` / ``fail_out``: the read / write buffers of the stage's
        ``valid.fail`` port). Bits are handed out in pipeline order; call it once per
        independent check.

        Args:
            name: The bit's key in ``Program.fail_bits`` / ``Track.fail_bits``
                (default: the stage id). Unique in the pipeline.

        Raises:
            ValueError: the stage is not a validity stage.
            track_gen.pipeline.PlanError: the name is taken, or bits 16-30 are all
                taken (a pipeline holds at most 15 plugin fail bits).
        """
        self._check_open("fail_bit")
        kind = getattr(self._stage, "kind", None)
        if self._plan is None or kind != "validity":
            raise ValueError(f"stage {self._label()}: fail_bit() is for validity stages; "
                             f"this stage is kind {kind!r}")
        return self._plan.fail_bit(self.stage_id if name is None else str(name))

    def _reserve_fail_bits(self, table: Mapping[str, int]) -> None:
        """Claim TrackGen's built-in bits (0-15) by name; built-in stages only."""
        self._check_open("_reserve_fail_bits")
        if self._plan is None:
            raise ValueError(f"stage {self._label()}: fail bits belong to a pipeline")
        self._plan.reserve_fail_bits(type(self._stage), dict(table))


def _is_warp_dtype(dtype) -> bool:
    """A Warp scalar or vector/matrix type usable as an array dtype."""
    if dtype in (wp.float32, wp.float64, wp.float16, wp.int32, wp.int64, wp.uint32,
                 wp.int8, wp.uint8, wp.int16, wp.uint16, wp.uint64):
        return True
    return isinstance(dtype, type) and hasattr(dtype, "_wp_scalar_type_")


def _new_builder(**kwargs) -> StageBuilder:
    """The orchestrator's StageBuilder constructor."""
    return StageBuilder(_token=_BUILDER_TOKEN, **kwargs)


def _close_builder(b: StageBuilder) -> None:
    b._closed = True


def _check_stage(cls, *, kind: "str | None" = None, api_version: int = 1,
                 builtin: bool = False) -> str:
    """The registration rules of a Stage class; returns its kind. ``builtin`` (a stage
    shipped with track_gen) is exempt from the per-kind write rule only."""
    from .params import _check_params_class  # noqa: PLC0415

    if isinstance(cls, Stage):
        raise TypeError(f"register the Stage class, not an instance: "
                        f"registry.add({type(cls).__name__})")
    if not (isinstance(cls, type) and issubclass(cls, Stage)):
        raise TypeError(f"not a Stage subclass: {cls!r}")
    label = cls.__name__
    if getattr(cls, "__abstractmethods__", None):
        raise TypeError(f"Stage {label} is abstract (implement "
                        f"{sorted(cls.__abstractmethods__)})")
    declared = getattr(cls, "kind", None)
    if not isinstance(declared, str) or not declared:
        raise TypeError(f"Stage {label}.kind must be a non-empty string, got {declared!r}")
    if kind is not None and kind != declared:
        raise ValueError(f"Stage {label} declares kind {declared!r}, not {kind!r}")
    if declared == "gate_generator":
        raise ValueError(f"Stage {label}: gate generators become Stages in roadmap PR11; "
                         f"register a GateGeneratorSpec")
    if declared != "generator" and declared not in _OPEN_KINDS:
        raise ValueError(f"Stage {label}: Stage kind {declared!r} is not open to plugins; "
                         f"open kinds: {sorted(('generator',) + _OPEN_KINDS)} (the other "
                         f"track stages are built-ins, see track_gen.pipeline)")
    version = getattr(cls, "api_version", 1)
    if isinstance(version, bool) or not isinstance(version, int) or version > api_version:
        raise ValueError(f"Stage {label}.api_version={version!r}: this track_gen provides "
                         f"plugin API {api_version}")
    if not isinstance(cls.capture, CaptureClass):
        raise TypeError(f"Stage {label}.capture must be a CaptureClass, got {cls.capture!r}")
    if cls.capture == CaptureClass.BOUNDED:
        raise ValueError(f"Stage {label}.capture=BOUNDED arrives with roadmap PR12; "
                         f"declare STATIC")
    _check_params_class(cls.Params, owner=f"Stage {label}")
    if declared != "generator":
        _check_pipeline_stage(cls, declared, builtin=builtin)
        return declared
    output = cls.output
    if output == "anchors":
        raise ValueError(f"Stage {label}: output='anchors' arrives with roadmap PR11; "
                         f"register a GateGeneratorSpec")
    if output != "centerline":
        raise ValueError(f"Stage {label}: a generator Stage must declare "
                         f"output='centerline', got {output!r}")
    writes = dict(cls.writes)
    channels = list(writes.values())
    bad = sorted(str(c) for c in channels if c not in _GENERATOR_CHANNELS)
    if bad:
        raise ValueError(f"Stage {label}.writes: channels {bad} arrive with roadmap "
                         f"PR10/PR13; a generator writes {sorted(_GENERATOR_CHANNELS)}")
    if "raw.xy" not in channels:
        raise ValueError(f"Stage {label}.writes must map a port to 'raw.xy'")
    if len(set(channels)) != len(channels):
        raise ValueError(f"Stage {label}.writes maps two ports to one channel: {writes}")
    if dict(cls.reads):
        raise ValueError(f"Stage {label}.reads must be empty (a generator reads no "
                         f"channel)")
    if frozenset(cls.provides):
        raise ValueError(f"Stage {label}.provides must be empty (roadmap PR11/PR13)")
    if cls.topology not in ("closed", "any"):
        raise ValueError(f"Stage {label}.topology must be 'closed' or 'any', got "
                         f"{cls.topology!r}")
    return declared


def _port_map(cls, attr: str) -> dict:
    value = getattr(cls, attr)
    if not isinstance(value, Mapping):
        raise TypeError(f"Stage {cls.__name__}.{attr} must be a mapping of ports to "
                        f"channels, got {type(value).__name__}")
    out = {}
    for port, ch in value.items():
        if not isinstance(port, str) or not port:
            raise TypeError(f"Stage {cls.__name__}.{attr}: ports must be non-empty "
                            f"strings, got {port!r}")
        if not isinstance(ch, str) or not ch:
            raise TypeError(f"Stage {cls.__name__}.{attr}[{port!r}] must be a channel "
                            f"name, got {ch!r}")
        out[port] = ch
    return out


def _rewrite_problems(label: str, reads: Mapping, writes: Mapping) -> "list[str]":
    """The port rules of a stage's declared (or rewired) ``reads`` / ``writes``: one
    write port per channel, a port in both maps names one channel, and a rewritten
    channel goes through one port (R1)."""
    problems = []
    seen: dict = {}
    for port, ch in writes.items():
        if ch in seen:
            problems.append(f"{label} writes {ch!r} through two ports ({seen[ch]!r}, "
                            f"{port!r})")
        seen[ch] = port
    for port in sorted(set(reads) & set(writes)):
        if reads[port] != writes[port]:
            problems.append(f"{label}: port {port!r} reads {reads[port]!r} and writes "
                            f"{writes[port]!r}; a port in both reads and writes names "
                            f"one channel")
    for rport, ch in reads.items():
        wport = seen.get(ch)
        if wport is not None and wport != rport:
            problems.append(f"{label}: channel {ch!r} is read through port {rport!r} and "
                            f"written through port {wport!r}; an in-place rewrite reads "
                            f"and writes it through one port")
    return problems


def _write_rule_problem(label: str, kind: str, writes: Mapping) -> "str | None":
    """The per-kind write rule: a ``kind`` stage writes its one catalogue channel and
    plugin channels only."""
    from ..stages.channels import is_plugin_channel  # noqa: PLC0415

    allowed = _KIND_WRITES.get(kind, ())
    for port, ch in writes.items():
        if ch not in allowed and not is_plugin_channel(ch):
            return (f"{label} (kind {kind!r}) writes {ch!r} (port {port!r}): a {kind} "
                    f"stage may write {list(allowed)} and plugin channels "
                    f"'cl.<package>.<name>'")
    return None


def _check_pipeline_stage(cls, kind: str, *, builtin: bool) -> None:
    """The rules of a relax / width / elevation / validity Stage (see the module
    docstring)."""
    from ..stages.channels import CHANNELS, FAIL, is_plugin_channel  # noqa: PLC0415

    label = f"Stage {cls.__name__}"
    if cls.output is not None:
        raise ValueError(f"{label}: output is for generators (kind {kind!r} declares "
                         f"none)")
    reads, writes = _port_map(cls, "reads"), _port_map(cls, "writes")
    unknown = sorted({ch for ch in (*reads.values(), *writes.values())
                      if ch not in CHANNELS and not is_plugin_channel(ch)})
    if unknown:
        raise ValueError(f"{label}: unknown channels {unknown}; use catalogue channels or "
                         f"namespaced plugin channels 'cl.<package>.<name>'")
    problems = _rewrite_problems(label, reads, writes)
    if problems:
        raise ValueError(problems[0])
    if not builtin:
        problem = _write_rule_problem(label, kind, writes)
        if problem is not None:
            raise ValueError(problem)
    if kind == "validity" and FAIL not in writes.values():
        raise ValueError(f"{label} (kind 'validity') must write {FAIL!r}")
    if frozenset(cls.provides):
        raise ValueError(f"{label}.provides must be empty (roadmap PR11/PR13)")
    if cls.topology not in ("closed", "any"):
        raise ValueError(f"{label}.topology must be 'closed' or 'any', got "
                         f"{cls.topology!r}")


def _is_stage(spec: Any) -> bool:
    return isinstance(spec, type) and issubclass(spec, Stage)
