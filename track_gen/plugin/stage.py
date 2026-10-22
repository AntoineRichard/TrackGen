"""The :class:`Stage` protocol and its :class:`StageBuilder`.

A Stage is a class that allocates everything it needs ONCE, through the
:class:`StageBuilder` the orchestrator hands to :meth:`Stage.build`, and returns an
``enqueue(ctx)`` callable that emits its device work on a
:class:`~track_gen.exec.LaunchContext`. Third parties register ``kind="generator"``
stages with ``output="centerline"``: register the CLASS with ``registry.add(MyStage)``
and select it with ``TrackGenConfig(generator=MyStage.name)``. The other stages of the
track pipeline (resample, relax, width, frames, offset, elevation, validity, lift) are
TrackGen's own built-ins in this release (``track_gen.pipeline``, preview); they read and
write named channels through :meth:`StageBuilder.read` / :meth:`StageBuilder.write`.

A generator stage writes the raw closed centerline (channel ``"raw.xy"``,
``[E * raw]`` ``vec2f``, ``raw`` from :meth:`Stage.capacity`, default
``TrackGenConfig.num_points``) and optionally a per-env validity flag (``"gen.valid"``,
``[E]`` ``int32``; when it is never written the orchestrator sets it to 1 once). The
relax / inflate / validity tail is unchanged.

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
    """Base of a pipeline stage (this release consumes generator stages only).

    Class attributes a stage sets:

    - ``kind``: the registry kind, ``"generator"`` (third-party stages of the other
      kinds arrive with roadmap PR10b and are rejected at registration until then).
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
      optionally ``"gen.valid"``. A channel in both ``reads`` and ``writes`` is an
      in-place rewrite: ``write`` returns a NEW version of the channel (later stages
      see it), ``read`` the previous one, so the stage writes its whole output into
      the ``write`` buffer (an update made to the ``read`` buffer is never seen).
    - ``provides``: must be empty (PR11/PR13).
    - ``topology``: ``"closed"`` or ``"any"``.
    - ``output``: ``"centerline"`` for a generator (``"anchors"`` arrives with PR11).
    - ``api_version``: the plugin API version the stage targets (``<= API_VERSION``).

    The orchestrator instantiates the class once per generator with no arguments.

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
    :meth:`scratch`, :meth:`param`, :meth:`input` and :meth:`salt`. Every method
    allocates at build time only, and raises once ``build()`` returned.

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

    def write(self, port: str, dtype, *, per: str = "env", interp=None,
              export=None) -> "wp.array":
        """The output buffer of ``writes[port]``.

        A generator writes ``"raw.xy"`` (``wp.vec2f``, ``per="raw"``, ``[E * raw]``)
        and ``"gen.valid"`` (``wp.int32``, ``per="env"``, ``[E]``). ``per`` sizes the
        buffer: ``"env"`` is ``[E]``, ``"cl"`` is ``[E * N_max]``, ``"raw"`` is
        ``[E * raw]``. The buffer is owned by the orchestrator and read directly by
        later stages (no copy). For a channel the stage also reads (an in-place
        rewrite) it is a NEW version: :meth:`read` keeps returning the previous one.

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
        want_dtype, want_per = plan.channel(channel)
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


def _new_builder(**kwargs) -> StageBuilder:
    """The orchestrator's StageBuilder constructor."""
    return StageBuilder(_token=_BUILDER_TOKEN, **kwargs)


def _close_builder(b: StageBuilder) -> None:
    b._closed = True


def _check_stage(cls, *, kind: "str | None" = None, api_version: int = 1) -> str:
    """The registration rules of a Stage class; returns its kind."""
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
    if declared != "generator":
        raise ValueError(f"Stage {label}: Stage kind {declared!r} arrives with roadmap "
                         f"PR10b; this release registers kind='generator' only (the "
                         f"other track stages are built-ins, see track_gen.pipeline)")
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


def _is_stage(spec: Any) -> bool:
    return isinstance(spec, type) and issubclass(spec, Stage)
