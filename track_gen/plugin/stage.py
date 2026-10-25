"""The :class:`Stage` protocol and its :class:`StageBuilder`.

A Stage is a class that allocates everything it needs ONCE, through the
:class:`StageBuilder` the orchestrator hands to :meth:`Stage.build`, and returns an
``enqueue(ctx)`` callable that emits its device work on a
:class:`~track_gen.exec.LaunchContext`. Register the CLASS with
``registry.add(MyStage)``. Third parties register stages of these kinds:

- ``"generator"`` (``output="centerline"``): the raw closed centerline, selected with
  ``TrackGenConfig(generator=MyStage.name)``;
- ``"gate_generator"`` (``output="anchors"``, roadmap PR11): gate anchors, selected with
  ``GateGenConfig(generator=MyStage.name)`` or
  :func:`track_gen.pipeline.presets.gates`;
- ``"relax"``, ``"width"``, ``"elevation"``, ``"validity"`` (roadmap PR10b): a stage of
  the track pipeline, selected with :func:`track_gen.pipeline.presets.track` (or the
  flat shortcuts ``TrackGenConfig.relax_solver`` / ``z_profile``). It reads and writes
  named channels through :meth:`StageBuilder.read` / :meth:`StageBuilder.write`.
  ``elevation`` and ``validity`` stages also run in the gate pipeline (an elevation's
  ``cl.*`` ports are rewired to their ``gates.*`` counterparts by the planner);
- ``"order"`` and ``"stations"`` (roadmap PR11): the gate pipeline's ordering (anchors
  -> gate slots) and the anchors a centerline generator's loop yields.

The other track and gate stages (resample, frames, offset, lift, normalize, carry,
gate_relax, gate_arclen, tangents, gate_frames) are TrackGen's built-ins.

A generator stage writes the raw closed centerline (channel ``"raw.xy"``,
``[E * raw]`` ``vec2f``, ``raw`` from :meth:`Stage.capacity`, default
``TrackGenConfig.num_points``) and optionally a per-env validity flag (``"gen.valid"``,
``[E]`` ``int32``; when it is never written the orchestrator sets it to 1 once). It may
also write per-point attributes (roadmap PR13a): the altitude ``"raw.z"`` and the
half-width ``"raw.half_width"`` it ``provides`` (``{"z"}`` / ``{"half_width"}``), and
its own ``"raw.<package>.<name>"`` channels, each with an ``interp=`` policy; the
resamples carry every one to the centerline (``cl.<attr>``), and ``export=True`` makes
one a ``Track.attributes`` entry.

A pipeline stage declares the channels it reads and writes (``reads`` / ``writes``:
catalogue channels, see :doc:`/how-it-works/pipeline-stages`, or its own namespaced
channels ``cl.<package>.<name>`` / ``gates.<package>.<name>``; a ``reads`` value
``"cl.*"`` is a required read port the user wires with ``StageRef.ports``). Each kind
may write one catalogue channel: ``relax`` ``cl.xy``, ``width`` ``cl.half_width``,
``elevation`` ``cl.z``, ``validity`` ``valid.fail`` (required). A channel the stage
also reads (same port) is a rewrite: it reads the previous version and writes a NEW
buffer (never the read one). A validity stage owns at least one fail bit
(:meth:`StageBuilder.fail_bit`) and ORs it into ``valid.fail``.

A gate generator stage writes the anchors (``"anchors.xy"``, ``[E * A]`` ``vec2f``, ``A``
from :meth:`Stage.capacity` ``["gates"]``, and ``"anchors.count"``, ``[E]`` ``int32``),
plus the per-anchor attributes it ``provides`` (``"anchors.z"``, ``"anchors.tangent"``,
``"anchors.half_size"``, ``"anchors.roll"``) and its own ``"anchors.<package>.<name>"``
attributes (carried to the gate slots by index: ``orderings={"raw"}``); it declares the
orders it supports
(:attr:`Stage.orderings` or :attr:`Stage.excluded_orderings`) and may prefer one
(:attr:`Stage.preferred`).
"""
from __future__ import annotations

import abc
import enum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Mapping

import warp as wp

from .._src.types import Layout
from .params import NoParams, _fields

if TYPE_CHECKING:  # pragma: no cover
    from ..exec.context import LaunchContext

__all__ = ["CaptureClass", "Layout", "Stage", "StageBuilder"]

# The channels a PR8 generator stage may write: name -> (dtype, per).
_GENERATOR_CHANNELS = {"raw.xy": (wp.vec2f, "raw"), "gen.valid": (wp.int32, "env")}
# The catalogue attributes a centerline generator may provide (roadmap PR13a): channel ->
# the ``provides`` name. Carried to cl.<attr> by the constant-spacing resample.
_RAW_PROVIDABLE = {"raw.z": "z", "raw.half_width": "half_width"}
_STAGE_ID = "generator"
_BUILDER_TOKEN = object()
# The key of the generator slot's attribute resolver in a PR8 adapter builder's outputs:
# ``resolve(channel, dtype, interp, export) -> wp.array`` (roadmap PR13a).
_FAMILY_RESOLVER = "<raw attributes>"


def _is_raw_attribute(channel) -> bool:
    """A centerline generator's raw attribute: ``raw.z``, ``raw.half_width`` or a
    ``raw.<package>.<name>`` plugin channel."""
    from ..stages.channels import channel_domain, is_plugin_channel  # noqa: PLC0415

    return channel in _RAW_PROVIDABLE or (is_plugin_channel(channel)
                                          and channel_domain(channel) == "raw")

# The pipeline kinds third parties may register (roadmap PR10b; order / stations: PR11),
# and the catalogue channels each may write (besides its own plugin channels
# cl.<pkg>.<name> / gates.<pkg>.<name>; a stations stage: anchors.<pkg>.<name>). An
# elevation writes cl.z, or gates.z once the planner rewired it.
_OPEN_KINDS = ("relax", "width", "elevation", "validity", "order", "stations")
# The stage kinds that write another stage's dynamic params (roadmap PR13b; built-in
# only in 0.x): re-exported as track_gen.stages.PARAM_WRITER_KINDS. Equal to
# track_gen.pipeline.spec.PARAM_WRITER_KINDS (kept there import-free; a test pins it).
_PARAM_WRITER_KINDS = ("randomize",)
_KIND_WRITES = {"relax": ("cl.xy",), "width": ("cl.half_width",),
                "elevation": ("cl.z", "gates.z"), "validity": ("valid.fail",),
                "order": ("gates.xy", "gates.count"),
                "stations": ("anchors.xy", "anchors.count", "anchors.z")}
# The catalogue channels a kind MUST write.
_KIND_REQUIRED = {"order": ("gates.xy", "gates.count"),
                  "stations": ("anchors.xy", "anchors.count")}
# Kept for importers: the kinds with required writes.
_KIND_WRITES_ALL = tuple(_KIND_REQUIRED)
# The plugin-channel domains each kind may write (default: the centerline and the gates).
_KIND_PLUGIN_DOMAINS = {"stations": ("anchors",)}
_DEFAULT_PLUGIN_DOMAINS = ("cl", "gates")
# A gate generator's channels: the anchors (required: xy, count) and the per-anchor
# attributes it may provide.
_ANCHOR_CHANNELS = ("anchors.xy", "anchors.count", "anchors.z", "anchors.tangent",
                    "anchors.half_size", "anchors.roll")
_PROVIDABLE = ("z", "tangent", "half_size", "roll")
_PREFERRED_KEYS = ("order", "normalize")
# The sizes a stage-written plugin channel may have (raw / anchors are fixed by the
# prefix of raw.<pkg>.<name> / anchors.<pkg>.<name>).
_PLUGIN_PERS = ("env", "cl", "gates")
# The dtypes of a carried (or exported) attribute, and the policies each admits.
_FAMILY_DTYPES = (wp.float32, wp.int32, wp.vec2f, wp.vec3f)
_INTERPS = ("linear", "nearest", "previous", "angle")
# The result field a catalogue family reaches (export= is for plugin channels).
_CATALOGUE_FIELD = {"raw.z": "Track.center.z", "cl.z": "Track.center.z",
                    "raw.half_width": "Track.half_width",
                    "cl.half_width": "Track.half_width", "anchors.z": "GateSequence.position.z",
                    "gates.z": "GateSequence.position.z"}


def _dtype_name(dtype) -> str:
    return getattr(dtype, "__name__", str(dtype))


def _check_family_write(channel: str, dtype, interp, export, *, first: bool,
                        per: "str | None" = None) -> "tuple[object, str | None, object]":
    """The ``interp=`` / ``export=`` rules of a write of ``channel`` (roadmap PR13a);
    returns ``(dtype, interp, export_key)`` (``export_key``: the result key, or
    ``None``). ``first``: the channel's first writer (only it may pass them)."""
    from ..stages.channels import CHANNELS, channel_domain, family_key, is_plugin_channel

    plugin = is_plugin_channel(channel)
    domain = channel_domain(channel)
    if not first:
        if interp is not None or (export is not False and export is not None):
            what = "interp=" if interp is not None else "export="
            raise ValueError(f"{channel!r}: {what} belongs to the channel's first writer "
                             f"(a rewrite inherits its interp and export)")
        return dtype, None, None
    # -- interp ----------------------------------------------------------------------
    raw_family = domain == "raw" and channel != "raw.xy"
    if raw_family and plugin:
        if interp is None:
            raise TypeError(f"raw channel {channel!r}: its writer passes interp= (one of "
                            f"{list(_INTERPS)}): a raw channel exists to be carried")
    if interp is not None:
        if not raw_family:
            if channel == "raw.xy":
                raise ValueError("interp= is for a generator's raw attributes (raw.z, "
                                 "raw.half_width, raw.<package>.<name>); raw.xy is "
                                 "resampled by the pipeline itself")
            raise ValueError(f"interp= is declared by the generator's raw write "
                             f"(raw.<pkg>.<name>); a stage-written channel ({channel!r}) "
                             f"is not resampled")
        if interp not in _INTERPS:
            raise ValueError(f"raw channel {channel!r}: interp must be one of "
                             f"{list(_INTERPS)}, got {interp!r}")
        if not plugin and interp != CHANNELS[channel].interp:
            raise TypeError(f"raw channel {channel!r} is carried with interp="
                            f"{CHANNELS[channel].interp!r} (a catalogue channel); got "
                            f"interp={interp!r}")
    if raw_family and not plugin:
        interp = CHANNELS[channel].interp
    carried = raw_family or (plugin and domain == "anchors")
    if carried and plugin and dtype is not None and dtype not in _FAMILY_DTYPES:
        raise TypeError(f"{domain} channel {channel!r}: a carried attribute is float32, "
                        f"int32, vec2f or vec3f, got {_dtype_name(dtype)}")
    if interp is not None and dtype is not None:
        if interp in ("linear", "angle") and dtype is wp.int32:
            raise TypeError(f"raw channel {channel!r}: an int32 attribute is carried with "
                            f"interp='nearest' or 'previous', got interp={interp!r}")
        if interp == "angle" and dtype is not wp.float32:
            raise TypeError(f"raw channel {channel!r}: interp='angle' carries float32 "
                            f"radians, got {_dtype_name(dtype)}")
    # -- export ----------------------------------------------------------------------
    if export is None or export is False:
        return dtype, interp, None
    if not isinstance(export, (bool, str)):
        raise TypeError(f"{channel!r}: export= must be True, False or the channel's key, "
                        f"got {type(export).__name__}")
    if not plugin:
        field = _CATALOGUE_FIELD.get(channel)
        what = f"reaches {field}" if field else "is a catalogue channel"
        raise ValueError(f"{channel} {what}; export= is for plugin channels")
    key = family_key(channel)
    if isinstance(export, str) and export != key:
        raise ValueError(f"export={export!r}: the key of {channel} is {key!r} (renaming is "
                         f"not supported)")
    if domain in ("cl", "gates") and per not in (None, "cl", "gates"):
        raise ValueError(f"{channel!r}: export= needs a per-point channel: per='{domain}' "
                         f"(got per={per!r})")
    if domain in ("cl", "gates") and per is None:
        raise ValueError(f"{channel!r}: export= needs a per-point channel: per='{domain}' "
                         f"(got the default per='env')")
    if dtype is not None and dtype not in _FAMILY_DTYPES:
        raise TypeError(f"{channel!r}: an exported attribute is float32, int32, vec2f or "
                        f"vec3f, got {_dtype_name(dtype)}")
    return dtype, interp, key


class CaptureClass(enum.IntEnum):
    """How a stage's ``enqueue`` may be captured (ordered: ``STATIC < BOUNDED < HOST``;
    a pipeline's class is the largest of its stages').

    - ``STATIC``: a fixed launch sequence, capturable by anything.
    - ``BOUNDED``: data-dependent convergence with a static bound, through
      :meth:`LaunchContext.bounded_loop <track_gen.exec.LaunchContext.bounded_loop>`
      with ``active=``: capturable by anything; its loop may stop early (an eager host
      poll on CPU, conditional graph nodes inside Warp-registered captures), so every
      iteration after ``active[0]`` reads 0 must be an exact no-op (an absorbing
      freeze; the conformance kit checks it).
    - ``HOST``: host readback / control flow (through :meth:`LaunchContext.host
      <track_gen.exec.LaunchContext.host>`): runs eagerly on every call, between the
      graph launches of the rest of its pipeline (segmented execution); a caller
      capture raises :class:`~track_gen.exec.NotCapturableError` before any launch.
      It reaches the host only through ``ctx.host()`` (HostOps, over
      ``StageBuilder.host_buffer`` buffers); a direct ``.numpy()``,
      ``wp.synchronize*()`` or per-call allocation still fails the conformance kit's
      ``hot_path.host_sync`` / ``hot_path.allocation``.
    """

    STATIC = 0
    BOUNDED = 1
    HOST = 2


class Stage(abc.ABC):
    """Base of a pipeline stage.

    Class attributes a stage sets:

    - ``kind``: the registry kind: ``"generator"``, ``"gate_generator"``, or a pipeline
      kind ``"relax"``, ``"width"``, ``"elevation"``, ``"validity"``, ``"order"`` or
      ``"stations"``.
    - ``name``: the registered name; third parties use a dotted ``"package.name"``.
    - ``Params``: the stage's :class:`~track_gen.plugin.Params` class (default
      :class:`~track_gen.plugin.NoParams`).
    - ``capture``: its :class:`CaptureClass` (default ``STATIC``; ``BOUNDED`` for a
      stage whose ``ctx.bounded_loop`` passes ``active=``, ``HOST`` for one that uses
      ``ctx.host()``); override :meth:`capture_class` when it depends on the params.
    - ``reads``: ``local port -> channel`` the stage reads (:meth:`StageBuilder.read`).
      Must be empty for a generator.
    - ``writes``: ``local port -> channel`` the stage writes
      (:meth:`StageBuilder.write`); a generator writes ``"raw.xy"`` (required) and
      optionally ``"gen.valid"`` and its raw attributes (``"raw.z"``,
      ``"raw.half_width"``, ``"raw.<package>.<name>"``); a gate generator
      ``"anchors.xy"`` and ``"anchors.count"`` (required), the attributes it provides
      and its ``"anchors.<package>.<name>"`` attributes. A channel in both
      ``reads`` and ``writes``, through ONE port, is a rewrite: ``write`` returns a NEW
      version of the channel (later stages see it), ``read`` the previous one, so the
      stage writes its whole output into the ``write`` buffer and never writes the
      ``read`` one.
    - ``provides``: a gate generator's per-anchor attributes, a subset of ``{"z",
      "tangent", "half_size", "roll"}`` equal to the ``anchors.<attr>`` channels it
      writes (the gate preset then drops the stages that would compute them); a
      centerline generator's, a subset of ``{"z", "half_width"}`` equal to the
      ``raw.z`` / ``raw.half_width`` it writes (the presets drop the elevation / the
      width stage); empty for every other kind.
    - ``orderings`` / ``excluded_orderings`` (gate generators): the ``order`` stages it
      supports: a whitelist (``orderings``, a frozenset of order names), or, with
      ``orderings=None``, every registered order but ``excluded_orderings``. A
      generator that provides attributes declares ``orderings=frozenset({"raw"})``
      (they are carried to the gate slots by index).
    - ``preferred``: a gate generator's ``{"order": name, "normalize": "extent"}`` (the
      order the gate preset picks by default, and whether it normalizes the gates'
      extent after ordering); a centerline generator's ``{"relax": name}`` (the relax
      stage ``presets.track(relax="auto")`` picks, ``"none"`` for an ingestion
      generator whose geometry must not move).
    - ``seeded``: ``False`` for a generator whose output does not depend
      on the seed (an ingestion generator): the conformance kit reports ``diversity``
      as skipped; default ``True``.
    - ``topology``: ``"closed"`` or ``"any"`` (the topologies the stage supports;
      ``BatchSpec.closed=False`` is refused in 0.x, naming the closed-only stages).
    - ``output``: ``"centerline"`` for a generator, ``"anchors"`` for a gate generator;
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
    orderings: ClassVar["frozenset[str] | None"] = None
    excluded_orderings: ClassVar[frozenset] = frozenset()
    preferred: ClassVar[Mapping[str, str]] = MappingProxyType({})
    topology: ClassVar[str] = "closed"
    output: ClassVar["str | None"] = None
    api_version: ClassVar[int] = 1
    seeded: ClassVar[bool] = True

    @classmethod
    def capture_class(cls, params) -> CaptureClass:
        """The stage's capture class for ``params`` (default: :attr:`capture`)."""
        return cls.capture

    @classmethod
    def declare(cls, params) -> "tuple[Mapping[str, str], frozenset]":
        """``(writes, provides)`` for these params (default: the class's
        :attr:`writes` and :attr:`provides`).

        Read for centerline generators only in 0.x: a generator whose
        raw attributes depend on its params (an ingestion generator writes ``raw.z``
        only when it is given altitudes) overrides it; the declaration is checked at
        bind time with the class rules (``raw.xy`` mapped, ``provides`` naming exactly
        the catalogue attributes written) and routes the presets (dropped stages,
        carried attributes) and the planner. ``params`` holds the static fields only.
        Overriding it on another kind is refused at registration.
        """
        return dict(cls.writes), frozenset(cls.provides)

    def capacity(self, params, layout: Layout) -> Mapping[str, int]:
        """Buffer capacities this stage needs. A generator may return ``{"raw": n}``
        (points per env of its ``"raw.xy"`` output, an integer ``n >= 3``); the default
        ``{}`` uses ``TrackGenConfig.num_points``. A gate generator (and a ``stations``
        stage) returns ``{"gates": A, "min_gates": k}``: ``A`` anchor rows per env (the
        most real anchors it emits, ``>= 1``; it must fit ``max_gates``) and ``k`` the
        fewest (default 0, at most ``A``). Any other key raises ``ValueError``."""
        return {}

    @abc.abstractmethod
    def build(self, b: "StageBuilder", params) -> "Callable[[LaunchContext], None]":
        """Allocate everything through ``b``, ONCE, and return ``enqueue(ctx)``.

        ``enqueue`` emits only ``ctx.launch`` / ``ctx.copy`` / ``ctx.fill`` /
        ``ctx.bounded_loop`` / ``track_gen.ops`` work: no allocation, no host sync, no
        branch on device data or on ``ctx.capturing`` (a ``HOST`` stage reaches the
        host through ``ctx.host()``).
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
    :meth:`scratch`, :meth:`host_buffer`, :meth:`param`, :meth:`input`, :meth:`salt`,
    :meth:`fail_bit`, :meth:`stride_of`, :meth:`per_of` and :meth:`dtype_of`.
    Every method allocates at build time only, and raises once ``build()`` returned.

    Attributes:
        E: The number of envs.
        layout: The run's :class:`~track_gen.results.Layout`.
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
                 stage_id: str = _STAGE_ID, plan=None, capture=None) -> None:
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
        self._raw_inputs: dict = {}                   # name -> array (per="raw", unlatched)
        self._reserved_inputs = frozenset(reserved_inputs)
        # Roadmap PR13b: a randomizer's target (array, field lo, field hi), set by the
        # planner; the generator's declared writes (Stage.declare) on the adapter path.
        self._drawn = None
        self._declared_writes: "Mapping[str, str] | None" = None
        self._salts: dict = {}                        # stream -> salt
        self._temps: dict = {}                        # port -> temp_like() count
        self._families: dict = {}                     # adapter: port -> (channel, dtype)
        # The stage's CaptureClass as the orchestrator resolved it (capture_class() of
        # the static params); None: resolved here on demand (host_buffer).
        self._capture = capture
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

    def write(self, port: str, dtype=None, *, per: "str | None" = None,
              interp: "str | None" = None, export: "bool | str" = False) -> "wp.array":
        """The output buffer of ``writes[port]``.

        A generator writes ``"raw.xy"`` (``wp.vec2f``, ``per="raw"``, ``[E * raw]``)
        and ``"gen.valid"`` (``wp.int32``, ``per="env"``, ``[E]``), plus its raw
        attributes: ``"raw.z"`` / ``"raw.half_width"`` (``float32``, the
        catalogue attributes it ``provides``) and its own ``"raw.<package>.<name>"``
        channels, each carried to the centerline (``cl.<package>.<name>``) by the
        resamples with the ``interp`` policy its write declares. ``per`` sizes the
        buffer: ``"env"`` is ``[E]``, ``"cl"`` is ``[E * N_max]``, ``"raw"`` is
        ``[E * raw]``, ``"anchors"`` is ``[E * A]`` and ``"gates"`` is
        ``[E * max_gates]`` (gate pipelines). ``dtype`` / ``per`` are those of the
        channel the port DECLARES; a port the planner rewired (an elevation's ``cl.z``
        on gates) gets a buffer sized for the channel it is wired to (:meth:`per_of`,
        :meth:`stride_of`). The buffer is owned by the orchestrator and read directly by
        later stages (no copy). For a channel the stage also reads (a rewrite) it is a
        NEW version: :meth:`read` keeps returning the previous one. A plugin channel
        (``<domain>.<package>.<name>``) takes the ``dtype`` / ``per`` of its first
        writer; a later writer must match. The prefix fixes ``per`` for ``raw.*``
        (``"raw"``) and ``anchors.*`` (``"anchors"``); ``cl.*`` / ``gates.*`` default
        to ``"env"``.

        ``dtype`` and ``per`` may be omitted for a catalogue channel, or a plugin
        channel an earlier stage wrote: they are then the channel's (``b.write("z")``).
        Given, they must match it. A plugin channel's first writer passes ``dtype``.

        Args:
            port: A write port of the stage.
            dtype: The Warp dtype (see above).
            per: The size (see above).
            interp: A raw attribute's carry policy (required on every
                ``raw.<package>.<name>``; ``"linear"``, ``"nearest"``, ``"previous"`` or
                ``"angle"``): how the resamples carry it to their output points. Only
                on a generator's raw writes.
            export: ``True`` (or the channel's key, e.g. ``"acme.surface_id"``) to
                export a per-point plugin attribute as ``Track.attributes[key]`` /
                ``GateSequence.attributes[key]`` (key: the channel name without its
                domain prefix). On per-point plugin channels only (``raw.*``,
                ``anchors.*``, or ``cl.*`` / ``gates.*`` written with ``per="cl"`` /
                ``"gates"``), by the channel's first writer.

        Raises:
            ValueError: an undeclared port (lists the declared ones), a channel the
                stage may not write, a second write of the same port, ``interp=`` on a
                channel that is not a generator's raw attribute, ``export=`` on a
                catalogue or per-env channel or naming another key, ``interp=`` /
                ``export=`` on a rewrite.
            TypeError: a ``dtype`` / ``per`` that does not match the channel; a raw
                attribute without ``interp=``, of a dtype other than float32, int32,
                vec2f or vec3f, or with a policy its dtype does not admit (int32:
                ``"nearest"`` / ``"previous"``; ``"angle"``: float32).
        """
        self._check_open("write")
        if self._plan is not None:
            return self._plan_write(port, dtype, per, interp, export)
        writes = self._stage_writes()
        if port not in writes:
            raise ValueError(f"stage {type(self._stage).__name__} declares no write port "
                             f"{port!r}; declared ports: {sorted(writes)}")
        channel = writes[port]
        if port in self._written:
            raise ValueError(f"port {port!r} ({channel!r}) is already written; call "
                             f"write() once per port")
        if channel not in _GENERATOR_CHANNELS:
            if not _is_raw_attribute(channel):
                raise ValueError(f"channel {channel!r} is not a generator channel; a "
                                 f"generator stage writes {sorted(_GENERATOR_CHANNELS)} "
                                 f"and its raw attributes (raw.z, raw.half_width, "
                                 f"raw.<package>.<name>)")
            return self._adapter_attribute(port, channel, dtype, per, interp, export)
        if interp is not None or (export is not False and export is not None):
            _check_family_write(channel, dtype, interp, export, first=True, per=per)
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

    def _adapter_attribute(self, port: str, channel: str, dtype, per, interp,
                           export) -> "wp.array":
        """A generator's raw attribute on the PR8 adapter path: checked like the
        planner's write, its buffer from the generator slot's resolver (the planner
        allocates the channel and records the family), or allocated privately when no
        pipeline carries it (the PR9 reference composition)."""
        from ..stages.channels import CHANNELS  # noqa: PLC0415

        if channel in CHANNELS:
            want = CHANNELS[channel].dtype
            if dtype is not None and dtype is not want:
                raise TypeError(f"channel {channel!r} has dtype {want.__name__}, got "
                                f"{getattr(dtype, '__name__', dtype)}")
            dtype = want
        elif dtype is None:
            raise TypeError(f"plugin channel {channel!r}: its first writer passes dtype=")
        if per is not None and per != "raw":
            raise TypeError(f"channel {channel!r} is per='raw' (fixed by its raw. prefix), "
                            f"got per={per!r}")
        dtype, interp, key = _check_family_write(channel, dtype, interp, export,
                                                 first=True, per="raw")
        resolve = self._outputs.get(_FAMILY_RESOLVER)
        if resolve is not None:
            arr = resolve(channel, dtype, interp, export)
        else:
            arr = wp.zeros(self.E * self.raw, dtype=dtype, device=self.device)
        self._families[port] = (channel, dtype)
        self._written[port] = channel
        return arr

    def _plan_write(self, port: str, dtype, per: str, interp=None,
                    export=False, fill=None) -> "wp.array":
        from ..stages.channels import channel_domain, is_plugin_channel  # noqa: PLC0415

        plan = self._plan
        if port not in plan.writes:
            raise ValueError(f"stage {self._label()} declares no write port {port!r}; "
                             f"declared ports: {sorted(plan.writes)}")
        channel = plan.writes[port]
        if port in self._written:
            raise ValueError(f"port {port!r} ({channel!r}) is already written; call "
                             f"write() once per port")
        info = plan.channel(channel)
        declared = plan.declared_channel(port)
        if info is not None and declared is not None and declared != channel:
            # A port rewired to another channel of its point class (cl.z -> gates.z):
            # dtype / per are checked against the DECLARED channel, the buffer is sized
            # by the wired one.
            want = plan.channel(declared)
            if want is not None:
                d_dtype, d_per = want
                if dtype is not None and dtype is not d_dtype:
                    raise TypeError(f"channel {declared!r} has dtype {d_dtype.__name__}, "
                                    f"got {getattr(dtype, '__name__', dtype)}")
                if per is not None and per != d_per:
                    raise TypeError(f"channel {declared!r} is per={d_per!r}, got "
                                    f"per={per!r}")
                dtype, per = None, None
        domain = channel_domain(channel)
        plugin = is_plugin_channel(channel)
        if (plugin and declared is not None and declared != channel
                and is_plugin_channel(declared)
                and {channel_domain(declared), domain} == {"cl", "gates"}
                and per == channel_domain(declared)):
            # A per-point plugin channel rewired across point classes (an elevation in a
            # gate Program: cl.<pkg>.<name> -> gates.<pkg>.<name>): per= names the
            # DECLARED channel's point class, the buffer is sized by the wired one.
            per = domain
        first = plan.is_first_write(channel)
        if plugin and domain in ("raw", "anchors"):
            if per is not None and per != domain:
                raise TypeError(f"channel {channel!r} is per={domain!r} (fixed by its "
                                f"{domain}. prefix), got per={per!r}")
            per = domain if info is None else per
        _, interp, key = _check_family_write(
            channel, dtype if dtype is not None else (info[0] if info else None), interp,
            export, first=first, per=per if per is not None or info is None else info[1])
        if info is None:
            # A plugin channel's first writer fixes its dtype and size.
            if dtype is None:
                raise TypeError(f"plugin channel {channel!r}: its first writer passes "
                                f"dtype= (and per=, default 'env')")
            per = "env" if per is None else per
            allowed = (domain,) if domain in ("raw", "anchors") else ("env", domain)
            if per not in allowed:
                raise ValueError(f"plugin channel {channel!r}: per must be one of "
                                 f"{list(allowed)}, got {per!r}")
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
        arr = plan.write(port, interp=interp, export=key, fill=fill)
        self._written[port] = channel
        return arr

    def dtype_of(self, port: str):
        """The Warp dtype of the channel ``port`` is WIRED to (a required read port
        ``"cl.*"``: the channel ``StageRef.ports`` names).

        Raises:
            ValueError: an undeclared port, or a plugin channel nobody wrote yet.
        """
        channel = self._port_channel(port, "dtype_of")
        if self._plan is None:
            fam = self._families.get(port)
            if fam is not None:
                return fam[1]
            return _GENERATOR_CHANNELS[channel][0]
        info = self._plan.channel(channel)
        if info is None:
            raise ValueError(f"stage {self._label()}: dtype_of({port!r}): plugin channel "
                             f"{channel!r} has no dtype yet")
        return info[0]

    # -- private (TrackGen's built-in stages) -----------------------------------------
    def _declare_constant(self, port: str, value: float) -> "wp.array":
        """Write ``port`` (a float32 ``per="cl"`` channel) as a CONSTANT version: allocated
        filled with ``value`` at build (no launch), and recorded so a consumer reading it
        (:meth:`_constant_of`) may take the value as a host scalar. ``width.constant``."""
        self._check_open("write")
        if self._plan is None:
            raise ValueError(f"stage {self._label()}: a constant version belongs to a "
                             f"pipeline")
        arr = self._plan_write(port, wp.float32, "cl", fill=float(value))
        self._plan.declare_constant(arr, float(value))
        return arr

    def _constant_of(self, port: str) -> "float | None":
        """The value of the version ``port`` reads when it is a constant one, else
        ``None``."""
        if self._plan is None:
            return None
        return self._plan.constant_of(self.read(port))

    def _carried(self) -> list:
        """The families this carrier stage carries (its implicit ``~`` ports), in
        order: ``_Carry(src_port, dst_port, src, dst, family, constant)``."""
        if self._plan is None:
            return []
        return self._plan.carried()

    def _label(self) -> str:
        return f"{self.stage_id!r} ({type(self._stage).__name__})"

    def _stage_writes(self) -> Mapping[str, str]:
        """The generator's write ports on the PR8 adapter path: its declaration for its
        params (``Stage.declare``), else the class's ``writes``."""
        if self._declared_writes is not None:
            return self._declared_writes
        return type(self._stage).writes

    def _drawn_target(self) -> "tuple[wp.array, float, float]":
        """A randomizer's target: ``(the drawn [E] float32 array, the field's declared
        lo, hi)`` (private: ``randomize.*`` is built-in only in 0.x)."""
        if self._drawn is None:
            raise ValueError(f"stage {self._label()} draws no field (not a randomizer)")
        return self._drawn

    def _length(self, per: str) -> int:
        if self._plan is not None:
            return self._plan.length(per)
        if per == "env":
            return self.E
        if per == "raw":
            return self.E * self.raw
        if per == "cl":
            return self.E * int(self.layout.stride)
        raise ValueError(f"unknown per={per!r}; expected 'env', 'raw' or 'cl'")

    def _port_channel(self, port: str, what: str) -> str:
        ports = {**self._plan.reads, **self._plan.writes} if self._plan is not None \
            else dict(self._stage_writes())
        channel = ports.get(port)
        if channel is None:
            raise ValueError(f"stage {self._label()}: {what}({port!r}): no such port; "
                             f"declared ports: {sorted(ports)}")
        return channel

    def per_of(self, port: str) -> str:
        """The ``per`` (``"env"``, ``"raw"``, ``"cl"``, ``"anchors"`` or ``"gates"``) of
        the channel ``port`` is WIRED to (after any rewiring: an elevation's ``z`` port
        is ``"gates"`` in a gate pipeline).

        Raises:
            ValueError: an undeclared port, or a plugin channel nobody wrote yet.
        """
        channel = self._port_channel(port, "per_of")
        if self._plan is None:
            if port in self._families:
                return "raw"
            return _GENERATOR_CHANNELS[channel][1]
        info = self._plan.channel(channel)
        if info is None:
            raise ValueError(f"stage {self._label()}: per_of({port!r}): plugin channel "
                             f"{channel!r} has no size yet")
        return info[1]

    def stride_of(self, port: str) -> int:
        """The row stride (points per env) of the channel ``port`` is WIRED to: ``raw``
        for ``"raw"``, ``N_max`` for ``"cl"``, ``A`` (the anchors producer's
        ``capacity()["gates"]``) for ``"anchors"``, ``max_gates`` for ``"gates"``, 1 for
        ``"env"``. Index env ``e``'s row as ``e * stride_of(port) + i``.

        Raises:
            ValueError: an undeclared port, or a plugin channel nobody wrote yet.
        """
        return self._length(self.per_of(port)) // self.E

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
            ports = dict(self._stage_writes())
            channel = ports.get(port)
            info = _GENERATOR_CHANNELS.get(channel)
            if port in self._families:
                info = (self._families[port][1], "raw")
            elif channel is not None and info is None and _is_raw_attribute(channel):
                raise ValueError(f"stage {self._label()}: temp_like({port!r}): raw "
                                 f"attribute {channel!r} has no dtype yet; call "
                                 f"b.write({port!r}, dtype, interp=...) first")
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

    def _scratch_full(self, name: str, value, dtype, shape) -> "wp.array":
        """:meth:`scratch` allocated filled with ``value`` (``wp.full``, at build;
        built-in stages only: the carriers' resample-table sentinels)."""
        self._check_open("scratch")
        key = f"scratch:{self.stage_id}/{name}"
        if key in self._scratch:
            raise ValueError(f"scratch buffer {name!r} is already allocated ({key})")
        arr = wp.full(shape, value, dtype=dtype, device=self.device)
        self._scratch[key] = arr
        return arr

    def host_buffer(self, dtype, shape) -> "wp.array":
        """A zero-initialised HOST array (pinned on a CUDA device) for a HOST stage's
        :class:`~track_gen.exec.HostOps` (``via=`` of ``readback`` / ``upload``).

        Allocated once, here; the stage keeps it in its ``enqueue`` closure.

        Raises:
            ValueError: the stage's capture class is not ``CaptureClass.HOST``.
            RuntimeError: called after ``build()`` returned.
        """
        self._check_open("host_buffer")
        cls = type(self._stage)
        cap = self._capture
        if cap is None:
            capture_class = getattr(cls, "capture_class", None)
            cap = capture_class(self._params) if callable(capture_class) else None
        if cap is not CaptureClass.HOST:
            what = cap.name if isinstance(cap, CaptureClass) else repr(cap)
            raise ValueError(f"StageBuilder.host_buffer() is for HOST stages: stage "
                             f"{self.stage_id!r} ({cls.__name__}) has capture class "
                             f"{what}")
        return wp.zeros(shape, dtype=dtype, device="cpu", pinned=self.device.is_cuda)

    def param(self, field: str) -> "wp.array":
        """The stage-side ``[E]`` array of a DYNAMIC Params field (float32 or int32).

        This is the working copy, latched from the live array the user writes
        (``gen.params[stage_id][field]``) at the start of every generation, for the
        envs that generation regenerates. A field a ``randomize.*`` stage draws is the
        drawn array itself (unlatched: written by the randomizer earlier in the same
        generation). READ-ONLY for the stage: never write it.

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

    def input(self, name: str, dtype, default=0, *, per: str = "env") -> "wp.array":
        """A persistent device input exposed as ``gen.inputs[name]`` (and
        ``Program.input_info[name]``), filled with ``default``.

        - ``per="env"`` (default): a stable LIVE ``[E]`` ``float32`` / ``int32`` array,
          LATCHED into the returned working copy at every generation like a dynamic
          param (a write takes effect for the envs the next generation regenerates).
        - ``per="raw"`` (centerline generators only): a stable ``[E *
          raw]`` ``float32`` / ``int32`` / ``vec2f`` / ``vec3f`` array, UNLATCHED: the
          returned array IS the input, read during the generation (the caller writes
          it on the generation's stream, or orders the streams with events). The
          ingestion inputs (``centerline.*``) are of this kind.

        The returned array is READ-ONLY for the stage, like :meth:`param`'s. Name the
        input with a dotted path in the stage's own namespace (``centerline.*`` is the
        ingestion namespace :func:`track_gen.pipeline.load_centerlines` reads).

        Args:
            name: The input's name (``"regen_mask"`` and existing names clash).
            dtype: ``wp.float32`` or ``wp.int32`` (``per="raw"``: also ``wp.vec2f`` /
                ``wp.vec3f``).
            default: The initial value of every element (a scalar; a vector input
                broadcasts it to every component).
            per: ``"env"`` or ``"raw"``.

        Raises:
            TypeError: another dtype.
            ValueError: a name clash; an unknown ``per``; ``per="raw"`` on a stage that
                is not a centerline generator.
        """
        self._check_open("input")
        if per not in ("env", "raw"):
            raise ValueError(f"input {name!r}: per must be 'env' or 'raw', got {per!r}")
        allowed = (wp.float32, wp.int32) if per == "env" else _FAMILY_DTYPES
        if dtype not in allowed:
            what = "wp.float32 or wp.int32" if per == "env" else \
                "wp.float32, wp.int32, wp.vec2f or wp.vec3f"
            raise TypeError(f"input {name!r}: dtype must be {what} (per={per!r}), got "
                            f"{getattr(dtype, '__name__', dtype)}")
        if not isinstance(name, str) or not name:
            raise ValueError(f"input name must be a non-empty string, got {name!r}")
        if name in self._reserved_inputs or name in self._inputs or \
                name in self._raw_inputs:
            raise ValueError(f"input {name!r} already exists (reserved: "
                             f"{sorted(self._reserved_inputs)})")
        if per == "raw":
            kind = getattr(self._stage, "kind", None)
            if kind != "generator":
                raise ValueError(f"stage {self._label()}: input {name!r}: per='raw' is for "
                                 f"centerline generators (kind 'generator'); this stage is "
                                 f"kind {kind!r}")
            value = int(default) if dtype is wp.int32 else float(default)
            arr = wp.full(self.E * self.raw, value, dtype=dtype, device=self.device)
            self._raw_inputs[name] = arr
            return arr
        value = float(default) if dtype is wp.float32 else int(default)
        live = wp.full(self.E, value, dtype=dtype, device=self.device)
        work = wp.full(self.E, value, dtype=dtype, device=self.device)
        self._inputs[name] = (live, work)
        return work

    def salt(self, stream: str) -> int:
        """A ``uint32`` salt for an independent random stream:
        ``hash32_host(fnv1a_32(f"{stage_id}/{stream}".encode()))``.

        Frozen for plugin API version 1, so outputs stay reproducible across releases.
        Key the per-env RNG with :func:`track_gen.ops.rng.init`
        (``init(seed[e], salt)``), or :func:`~track_gen.ops.rng.init_at` for an indexed
        stream. Idempotent per stream; two streams whose salts collide raise
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
    if declared not in ("generator", "gate_generator") and declared not in _OPEN_KINDS:
        raise ValueError(f"Stage {label}: Stage kind {declared!r} is not open to plugins; "
                         f"open kinds: {sorted(('generator', 'gate_generator') + _OPEN_KINDS)}"
                         f" (the other track and gate stages are built-ins, see "
                         f"track_gen.pipeline)")
    version = getattr(cls, "api_version", 1)
    if isinstance(version, bool) or not isinstance(version, int) or version > api_version:
        raise ValueError(f"Stage {label}.api_version={version!r}: this track_gen provides "
                         f"plugin API {api_version}")
    if not isinstance(cls.capture, CaptureClass):
        raise TypeError(f"Stage {label}.capture must be a CaptureClass, got {cls.capture!r}")
    _check_params_class(cls.Params, owner=f"Stage {label}")
    if not isinstance(getattr(cls, "seeded", True), bool):
        raise TypeError(f"Stage {label}.seeded must be a bool, got {cls.seeded!r}")
    if declared != "generator" and _overrides_declare(cls):
        raise ValueError(f"Stage {label}.declare() is read for centerline generators only "
                         f"in 0.x")
    if declared not in ("gate_generator", "generator") and \
            dict(getattr(cls, "preferred", {}) or {}):
        raise ValueError(f"Stage {label}.preferred is read for gate generators only "
                         f"(roadmap PR11) and centerline generators ('relax', roadmap "
                         f"PR13b)")
    if declared == "relax" and getattr(cls, "name", None) == "auto":
        raise ValueError(f"Stage {label}: the relax name 'auto' is reserved "
                         f"(presets.track(relax='auto') picks the generator's preferred "
                         f"relax)")
    if declared == "gate_generator":
        _check_gate_generator(cls, label)
        return declared
    if declared != "generator":
        _check_pipeline_stage(cls, declared, builtin=builtin)
        return declared
    output = cls.output
    if output == "anchors":
        raise ValueError(f"Stage {label}: output='anchors' is a gate generator: declare "
                         f"kind='gate_generator'")
    if output != "centerline":
        raise ValueError(f"Stage {label}: a generator Stage must declare "
                         f"output='centerline', got {output!r}")
    _check_centerline_preferred(cls, label)
    writes = _port_map(cls, "writes")
    if dict(cls.reads):
        raise ValueError(f"Stage {label}.reads must be empty (a generator reads no "
                         f"channel)")
    _check_generator_ports(f"Stage {label}", writes, cls.provides)
    if cls.topology not in ("closed", "any"):
        raise ValueError(f"Stage {label}.topology must be 'closed' or 'any', got "
                         f"{cls.topology!r}")
    return declared


def _overrides_declare(cls) -> bool:
    """Whether ``cls`` overrides :meth:`Stage.declare`."""
    fn = getattr(cls, "declare", None)
    return getattr(fn, "__func__", fn) is not Stage.declare.__func__


def _check_centerline_preferred(cls, label: str) -> None:
    """A centerline generator's ``preferred``: only ``{"relax": name}`` (a relax stage
    name or ``"none"``; ``"auto"`` is reserved)."""
    preferred = getattr(cls, "preferred", {}) or {}
    if not isinstance(preferred, Mapping):
        raise TypeError(f"Stage {label}.preferred must be a mapping, got "
                        f"{type(preferred).__name__}")
    unknown = sorted(str(k) for k in preferred if k != "relax")
    if unknown:
        raise ValueError(f"Stage {label}.preferred: unknown keys {unknown}; a centerline "
                         f"generator's key is 'relax'")
    relax = preferred.get("relax")
    if relax is not None and (not isinstance(relax, str) or not relax or relax == "auto"):
        raise ValueError(f"Stage {label}.preferred['relax'] must be a relax stage name or "
                         f"'none', got {relax!r}")


def _check_generator_ports(label: str, writes: Mapping, provides) -> None:
    """The port rules of a centerline generator's ``writes`` / ``provides`` (its class
    attributes at registration, its :meth:`Stage.declare` result at bind time)."""
    channels = list(writes.values())
    bad = sorted(str(c) for c in channels
                 if c not in _GENERATOR_CHANNELS and not _is_raw_attribute(c))
    if bad:
        raise ValueError(f"{label}.writes: channels {bad} are not generator "
                         f"channels; a generator writes {sorted(_GENERATOR_CHANNELS)}, the "
                         f"catalogue attributes it provides {sorted(_RAW_PROVIDABLE)} and "
                         f"its own raw attributes 'raw.<package>.<name>'")
    if "raw.xy" not in channels:
        raise ValueError(f"{label}.writes must map a port to 'raw.xy'")
    if len(set(channels)) != len(channels):
        raise ValueError(f"{label}.writes maps two ports to one channel: {dict(writes)}")
    if not isinstance(provides, (frozenset, set)):
        raise TypeError(f"{label}.provides must be a frozenset, got {provides!r}")
    written = sorted(_RAW_PROVIDABLE[c] for c in channels if c in _RAW_PROVIDABLE)
    if sorted(map(str, provides)) != written:
        raise ValueError(f"{label}: provides {sorted(map(str, provides))} must name "
                         f"exactly the catalogue attributes it writes ({written}: raw.z "
                         f"-> 'z', raw.half_width -> 'half_width')")


def _declared_of(spec, params) -> "tuple[Mapping[str, str], frozenset]":
    """``(writes, provides)`` of a centerline generator for ``params`` (roadmap PR13b):
    a Stage's :meth:`Stage.declare` (``params``: its static view), a legacy spec's
    nothing (``({}, frozenset())``). Every read of a generator's declared ports goes
    through here."""
    if not _is_stage(spec):
        return {}, frozenset()
    if spec.kind != "generator" or not _overrides_declare(spec):
        return dict(spec.writes), frozenset(spec.provides)
    from .params import _static_view  # noqa: PLC0415

    view = _static_view(params if params is not None else spec.Params())
    got = spec.declare(view)
    try:
        writes, provides = got
    except (TypeError, ValueError):
        raise TypeError(f"{spec.__name__}.declare() must return (writes, provides), got "
                        f"{got!r}") from None
    if not isinstance(writes, Mapping):
        raise TypeError(f"{spec.__name__}.declare(): writes must be a mapping of ports to "
                        f"channels, got {type(writes).__name__}")
    provides = frozenset(provides) if isinstance(provides, (set, frozenset)) else provides
    for port, ch in writes.items():
        if not isinstance(port, str) or not port or not isinstance(ch, str) or not ch:
            raise TypeError(f"{spec.__name__}.declare(): writes must map non-empty port "
                            f"names to channel names, got {port!r}: {ch!r}")
    _check_generator_ports(f"Stage {spec.__name__}.declare()", writes, provides)
    return dict(writes), provides


def _order_names(value) -> bool:
    return isinstance(value, frozenset) and all(isinstance(o, str) and o for o in value)


def _check_orderings(cls, label: str) -> None:
    """A gate generator's ``orderings`` / ``excluded_orderings`` / ``preferred``."""
    orderings, excluded = cls.orderings, cls.excluded_orderings
    if orderings is not None and (not _order_names(orderings) or not orderings):
        raise TypeError(f"Stage {label}.orderings must be None or a non-empty frozenset of "
                        f"order names, got {orderings!r}")
    if not _order_names(excluded):
        raise TypeError(f"Stage {label}.excluded_orderings must be a frozenset of order "
                        f"names, got {excluded!r}")
    if orderings is not None and excluded:
        raise ValueError(f"Stage {label}: declare orderings (a whitelist) or "
                         f"excluded_orderings (with orderings=None), not both")
    preferred = cls.preferred
    if not isinstance(preferred, Mapping):
        raise TypeError(f"Stage {label}.preferred must be a mapping, got "
                        f"{type(preferred).__name__}")
    unknown = sorted(str(k) for k in preferred if k not in _PREFERRED_KEYS)
    if unknown:
        raise ValueError(f"Stage {label}.preferred: unknown keys {unknown}; keys: "
                         f"{sorted(_PREFERRED_KEYS)}")
    order = preferred.get("order")
    if order is not None:
        if not isinstance(order, str) or not order:
            raise ValueError(f"Stage {label}.preferred['order'] must be an order name, got "
                             f"{order!r}")
        if orderings is not None and order not in orderings:
            raise ValueError(f"Stage {label}.preferred['order']={order!r} is not one of "
                             f"its orderings {sorted(orderings)}")
        if order in excluded:
            raise ValueError(f"Stage {label}.preferred['order']={order!r} is one of its "
                             f"excluded_orderings")
    normalize = preferred.get("normalize")
    if normalize is not None and normalize != "extent":
        raise ValueError(f"Stage {label}.preferred['normalize'] must be 'extent' (the "
                         f"built-in normalize stage), got {normalize!r}")


def _check_gate_generator(cls, label: str) -> None:
    """The rules of a ``kind="gate_generator"`` Stage (``output="anchors"``)."""
    output = cls.output
    if output != "anchors":
        raise ValueError(f"Stage {label}: a gate generator Stage declares output='anchors',"
                         f" got {output!r}")
    from ..stages.channels import channel_domain, is_plugin_channel  # noqa: PLC0415

    writes = _port_map(cls, "writes")
    channels = list(writes.values())
    attrs = sorted(c for c in channels
                   if is_plugin_channel(c) and channel_domain(c) == "anchors")
    bad = sorted(str(c) for c in channels if c not in _ANCHOR_CHANNELS and c not in attrs)
    if bad:
        raise ValueError(f"Stage {label}.writes: channels {bad} are not anchors channels; "
                         f"a gate generator writes {sorted(_ANCHOR_CHANNELS)} and its own "
                         f"per-anchor attributes 'anchors.<package>.<name>'")
    if "anchors.xy" not in channels or "anchors.count" not in channels:
        raise ValueError(f"Stage {label}.writes must map ports to 'anchors.xy' and "
                         f"'anchors.count'")
    if len(set(channels)) != len(channels):
        raise ValueError(f"Stage {label}.writes maps two ports to one channel: {writes}")
    if _port_map(cls, "reads"):
        raise ValueError(f"Stage {label}.reads must be empty (a generator reads no "
                         f"channel)")
    provides = cls.provides
    if not isinstance(provides, (frozenset, set)):
        raise TypeError(f"Stage {label}.provides must be a frozenset, got {provides!r}")
    written = {c.split(".", 1)[1] for c in channels} & set(_PROVIDABLE)
    if set(provides) != written:
        raise ValueError(f"Stage {label}: provides {sorted(map(str, provides))} must name "
                         f"exactly the anchors attributes it writes ({sorted(written)})")
    _check_orderings(cls, label)
    if provides and cls.orderings != frozenset({"raw"}):
        raise ValueError(f"Stage {label} provides per-anchor attributes {sorted(provides)}: "
                         f"they are carried to the gate slots by index, so orderings "
                         f"must be frozenset({{'raw'}})")
    if attrs and cls.orderings != frozenset({"raw"}):
        raise ValueError(f"Stage {label} writes per-anchor attributes {attrs}: they are "
                         f"carried to the gate slots by index (carry.attributes), so "
                         f"orderings must be frozenset({{'raw'}})")
    if ("half_size" in provides) != ("roll" in provides):
        raise ValueError(f"Stage {label} provides 'half_size' without 'roll' (or 'roll' "
                         f"without 'half_size'): a posed gate frame needs both (write "
                         f"roll = 0 for upright gates)")
    if cls.topology not in ("closed", "any"):
        raise ValueError(f"Stage {label}.topology must be 'closed' or 'any', got "
                         f"{cls.topology!r}")


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
    """The per-kind write rule: a ``kind`` stage writes its catalogue channels and plugin
    channels of its domains only (``cl.*`` / ``gates.*``; a stations stage
    ``anchors.*``)."""
    from ..stages.channels import channel_domain, is_plugin_channel  # noqa: PLC0415

    allowed = _KIND_WRITES.get(kind, ())
    domains = _KIND_PLUGIN_DOMAINS.get(kind, _DEFAULT_PLUGIN_DOMAINS)
    for port, ch in writes.items():
        if ch in allowed:
            continue
        if is_plugin_channel(ch) and channel_domain(ch) in domains:
            continue
        plugins = " or ".join(f"'{d}.<package>.<name>'" for d in domains)
        return (f"{label} (kind {kind!r}) writes {ch!r} (port {port!r}): a {kind} "
                f"stage may write {list(allowed)} and plugin channels {plugins}")
    required = _KIND_REQUIRED.get(kind, ())
    if required and not set(required) <= set(writes.values()):
        return (f"{label} (kind {kind!r}) must write "
                f"{' and '.join(repr(c) for c in required)}")
    return None


def _check_pipeline_stage(cls, kind: str, *, builtin: bool) -> None:
    """The rules of a relax / width / elevation / validity Stage (see the module
    docstring)."""
    from ..stages.channels import (CHANNELS, FAIL, is_plugin_channel,  # noqa: PLC0415
                                   required_port_domain)

    label = f"Stage {cls.__name__}"
    if cls.output is not None:
        raise ValueError(f"{label}: output is for generators (kind {kind!r} declares "
                         f"none)")
    reads, writes = _port_map(cls, "reads"), _port_map(cls, "writes")
    wildcard = sorted(p for p, ch in writes.items() if required_port_domain(ch))
    if wildcard:
        raise ValueError(f"{label}: write port {wildcard[0]!r} names "
                         f"{writes[wildcard[0]]!r}; a '<domain>.*' port is a required READ "
                         f"port (the user wires it with StageRef.ports)")
    unknown = sorted({ch for ch in (*reads.values(), *writes.values())
                      if ch not in CHANNELS and not is_plugin_channel(ch)
                      and not required_port_domain(ch)})
    if unknown:
        raise ValueError(f"{label}: unknown channels {unknown}; use catalogue channels or "
                         f"namespaced plugin channels 'cl.<package>.<name>' (prefix raw., "
                         f"cl., anchors. or gates.), or a required read port '<domain>.*'")
    raw = sorted(ch for ch in writes.values()
                 if is_plugin_channel(ch) and ch.startswith("raw."))
    if raw:
        raise ValueError(f"{label} writes {raw[0]!r}: raw.<package>.<name> attributes are "
                         f"written by a centerline generator only")
    problems = _rewrite_problems(label, reads, writes)
    if problems:
        raise ValueError(problems[0])
    if not builtin:
        problem = _write_rule_problem(label, kind, writes)
        if problem is not None:
            raise ValueError(problem)
    if kind == "validity" and FAIL not in writes.values():
        raise ValueError(f"{label} (kind 'validity') must write {FAIL!r}")
    required = _KIND_REQUIRED.get(kind, ())
    if required and not set(required) <= set(writes.values()):
        raise ValueError(f"{label} (kind {kind!r}) must write "
                         f"{' and '.join(repr(c) for c in required)}")
    if frozenset(cls.provides):
        raise ValueError(f"{label}.provides must be empty (generators provide attributes: "
                         f"a centerline generator 'z' / 'half_width', a gate generator its "
                         f"per-anchor attributes)")
    if cls.topology not in ("closed", "any"):
        raise ValueError(f"{label}.topology must be 'closed' or 'any', got "
                         f"{cls.topology!r}")


def _is_stage(spec: Any) -> bool:
    return isinstance(spec, type) and issubclass(spec, Stage)
