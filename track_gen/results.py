"""The result objects (L1): :class:`Track`, :class:`GateSequence`, :class:`Centerline`,
their :class:`Layout`, the capability protocols and :func:`check_source` (roadmap PR13b).

Every result is flat, fixed-stride device storage: per-point arrays ``[E * stride]``
(NaN past ``count[e]``; int32 ``-1``) and per-env arrays ``[E]``, described by a
:class:`Layout` (``num_envs``, ``stride``, ``device``, ``closed``). Each result class
offers:

- ``validate(need=())``: check every field against the layout (host only); a
  hand-built result with a mismatched field is rejected with a message naming the field
  and the expected shape;
- ``view(e0, e1)``: a zero-copy env slice (kernel-safe: every kernel indexes from the
  view's element 0), e.g. ``TrackGenerator(..., out=big.view(0, 4))`` for mixed batches;
- ``copy_from(src, *, stream=None, ctx=None)``: stream-ordered copies, no allocation,
  capture-safe (the hot-path snapshot);
- ``clone(*, stream=None)``: an owned deep copy (allocates: cold);
- ``alloc(layout, *, attributes=None)``: a fully allocated result in the "no track yet"
  state (cold).

Each protocol is a capability a family of consumers shares, and each consumer checks
the subset of it it reads (``check_source(..., members=...)``), so a utility accepts any
object with those members (a ``Track``, a ``Centerline``, a user's own object):
``HasCenterline`` (``TrackLocalizer``, ``localize.curvature`` / ``speed_profile``),
``HasBand`` (``CollisionChecker``, ``PropSampler``, ``CheckpointSampler``,
``HeightFieldBaker``), ``HasGates`` (a gates-mode ``Course``'s course line),
``HasGateFrames`` (its frame checker, ``CheckpointSet.from_gates``) and
``HasGeneration`` (generation-aware consumers; checked together with another protocol:
``check_source(obj, (HasCenterline, HasGeneration), ...)``), ``HasValidity`` (the per-env
``valid`` flags, roadmap PR14) and ``HasCheckpoints`` (an ordered checkpoint set: what
``ProgressTracker`` reads, roadmap PR14). A shipped protocol never changes: a new
capability is a new protocol, composed with the existing ones. The library never relies
on ``isinstance`` against them (their runtime semantics differ across Python versions):
consumers check their source with :func:`check_source`.

Optional reads (roadmap PR14): a consumer that reads a member only when the source has
it (``valid``: the invalid-env sentinels; ``generation``: the generation-aware
self-reset) names it in ``check_source(..., optional=...)``: a present member is checked
like a required one (same messages), a missing (or ``None``) one is skipped.
"""
from __future__ import annotations

import types
from typing import Protocol, runtime_checkable

import warp as wp

from ._src.types import (_MEMBER_SCHEMA, Centerline, GateSequence, Layout, Track,
                         _canonical_device, _check_members, _infer_layout,
                         _stored_layout)

__all__ = [
    "Centerline",
    "GateSequence",
    "HasBand",
    "HasCenterline",
    "HasCheckpoints",
    "HasGateFrames",
    "HasGates",
    "HasGeneration",
    "HasValidity",
    "Layout",
    "Track",
    "check_source",
]


@runtime_checkable
class HasCenterline(Protocol):
    """A batched closed centerline: what ``TrackLocalizer`` and
    ``localize.curvature`` / ``speed_profile`` read. ``Track`` and :class:`Centerline`
    satisfy it."""

    center: wp.array      #: ``[E * stride]`` vec3f
    arclen: wp.array      #: ``[E * stride]`` float32
    length: wp.array      #: ``[E]`` float32
    count: wp.array       #: ``[E]`` int32


@runtime_checkable
class HasBand(Protocol):
    """A centerline with its road band: what ``CollisionChecker``, ``PropSampler``,
    ``CheckpointSampler`` and ``HeightFieldBaker`` read. ``Track`` satisfies it."""

    center: wp.array      #: ``[E * stride]`` vec3f
    outer: wp.array       #: ``[E * stride]`` vec3f
    inner: wp.array       #: ``[E * stride]`` vec3f
    count: wp.array       #: ``[E]`` int32
    valid: wp.array       #: ``[E]`` int32


@runtime_checkable
class HasGates(Protocol):
    """Gate positions: what a gates-mode ``Course``'s course line reads.
    ``GateSequence`` satisfies it."""

    position: wp.array    #: ``[E * stride]`` vec3f
    count: wp.array       #: ``[E]`` int32
    valid: wp.array       #: ``[E]`` int32


@runtime_checkable
class HasGateFrames(HasGates, Protocol):
    """Posed gates: what a gates-mode ``Course``'s frame checker and
    ``CheckpointSet.from_gates`` read. ``GateSequence`` satisfies it."""

    forward: wp.array     #: ``[E * stride]`` vec3f
    left: wp.array        #: ``[E * stride]`` vec3f
    right: wp.array       #: ``[E * stride]`` vec3f
    orientation: wp.array  #: ``[E * stride]`` quatf
    half_size: wp.array   #: ``[E * stride]`` float32


@runtime_checkable
class HasGeneration(Protocol):
    """The per-env generation epoch (``[E]`` int32): a ``Track`` / ``GateSequence`` /
    ``Centerline`` whose ``generation`` is set. Per env only, so check it together with
    the protocol that gives the stride: ``check_source(obj, (HasCenterline,
    HasGeneration), consumer=...)``."""

    generation: wp.array  #: ``[E]`` int32


@runtime_checkable
class HasValidity(Protocol):
    """The per-env validity flags (``[E]`` int32, 1 = valid): a ``Track`` /
    ``GateSequence`` / ``Centerline`` / ``CheckpointSet`` whose ``valid`` is set (roadmap
    PR14). Per env only, so check it together with the protocol that gives the stride:
    ``check_source(obj, (HasCenterline, HasValidity), optional=("valid",), ...)``."""

    valid: wp.array       #: ``[E]`` int32


@runtime_checkable
class HasCheckpoints(Protocol):
    """An ordered per-env checkpoint set: what ``ProgressTracker`` reads (roadmap PR14).
    ``CheckpointSet`` satisfies it (``CheckpointSampler``'s set, or
    ``CheckpointSet.from_gates`` of a ``GateSequence``)."""

    position: wp.array    #: ``[E * stride]`` vec3f checkpoint centers
    left: wp.array        #: ``[E * stride]`` vec3f crossing-segment endpoints
    right: wp.array       #: ``[E * stride]`` vec3f
    tangent: wp.array     #: ``[E * stride]`` vec3f unit forward directions
    up_half: wp.array     #: ``[E * stride]`` float32 vertical half-openings
    count: wp.array       #: ``[E]`` int32


_MEMBERS: dict = {}


def _protocols(protocol) -> tuple:
    """``protocol`` (one capability protocol, or a tuple of them) as a non-empty tuple of
    protocols; ``TypeError`` for anything else."""
    group = tuple(protocol) if isinstance(protocol, (tuple, list)) else (protocol,)
    if not group:
        raise TypeError("check_source: protocol must be a capability protocol (or a "
                        "tuple of them), got an empty tuple")
    for p in group:
        if not (isinstance(p, type) and getattr(p, "_is_protocol", False)
                and _members_of(p)):
            raise TypeError(f"check_source: protocol must be a capability protocol of "
                            f"track_gen.results (HasCenterline, HasBand, ...) or a tuple "
                            f"of them, got {getattr(p, '__name__', p)!r}")
    return group


def _members_of(protocol) -> "tuple[str, ...]":
    """The annotated members of ``protocol`` (its MRO, base protocols first)."""
    cached = _MEMBERS.get(protocol)
    if cached is not None:
        return cached
    names: list = []
    for klass in reversed(protocol.__mro__):
        if klass in (Protocol, object) or not getattr(klass, "_is_protocol", False):
            continue
        for name in getattr(klass, "__annotations__", {}):
            if name not in names and not name.startswith("_"):
                names.append(name)
    out = tuple(names)
    _MEMBERS[protocol] = out
    return out


def _source_layout(obj, owner: str) -> "Layout | None":
    """``obj.layout`` when it is a :class:`Layout`. TrackGen's own results must hold a
    ``Layout`` or ``None`` (``TypeError`` otherwise); any other object's unrelated
    ``layout`` attribute (a string, an enum of another library) is ignored."""
    if isinstance(obj, (Track, GateSequence, Centerline)):
        return _stored_layout(obj, owner)
    lay = getattr(obj, "layout", None)
    return lay if isinstance(lay, Layout) else None


def _check_and_bind(obj, protocol, *, consumer: str, topology: str = "closed",
                    members: "tuple[str, ...] | None" = None, optional=()):
    """:func:`check_source`, also returning the checked members as read (a
    ``SimpleNamespace``): what a consumer binds at construction, so it never reads (or
    validates) a member twice. An ``optional`` member the source lacks is ``None`` in
    the namespace."""
    if topology not in ("closed", "any"):
        raise ValueError(f"topology must be 'closed' or 'any', got {topology!r}")
    group = _protocols(protocol)
    every: list = []
    for p in group:
        every.extend(n for n in _members_of(p) if n not in every)
    label = " + ".join(p.__name__ for p in group)
    opt = (optional,) if isinstance(optional, str) else tuple(optional)
    bad_opt = [n for n in opt if n not in every]
    if bad_opt:
        raise ValueError(f"check_source: optional members {bad_opt} are not members of "
                         f"{label} ({', '.join(every)})")
    if members is None:
        names = tuple(every)
    else:
        if isinstance(members, str):
            members = (members,)
        names = tuple(members)
        extra = [n for n in names if n not in every]
        if extra or not (names or opt):
            bad = list(extra or names)
            raise ValueError(f"check_source: members {bad} are not members of {label} "
                             f"({', '.join(every)})")
    required = tuple(n for n in names if n not in opt)
    unknown = [n for n in required + opt if n not in _MEMBER_SCHEMA]
    if unknown:
        raise ValueError(f"check_source: {label} has members {unknown} that no TrackGen "
                         f"result schema defines (a protocol member is a field of Track, "
                         f"GateSequence or Centerline)")
    owner = type(obj).__name__
    values = {n: getattr(obj, n, None) for n in required + opt}  # each member read ONCE
    missing = [n for n in required if values[n] is None]
    if missing:
        # Name the protocols and members that are required; the optional ones apart.
        need = [p.__name__ for p in group if any(n not in opt for n in _members_of(p))]
        listed = ", ".join(n for n in every if n not in opt)
        if opt:
            listed += f"; optional: {', '.join(opt)}"
        raise TypeError(f"{consumer} needs a {' + '.join(need) or label} source "
                        f"({listed}): {owner} has no "
                        f"{', '.join(repr(m) for m in missing)}")
    bound = types.SimpleNamespace(**values)
    checked = required + tuple(n for n in opt if values[n] is not None)
    schema = tuple((n, *_MEMBER_SCHEMA[n]) for n in checked)
    try:
        lay = _source_layout(obj, owner)
        if lay is None:
            if not schema:
                raise ValueError(f"{owner} has none of the optional members "
                                 f"{list(opt)} and no layout: nothing gives its shape")
            prefer = next((n for n, per, _ in schema if per == "point"), "center")
            lay = _infer_layout(bound, owner, schema, prefer)
        _check_members(bound, owner, lay, schema, consumer=consumer)
    except ValueError as exc:
        raise ValueError(f"{consumer}: {exc}") from None
    except TypeError as exc:
        raise TypeError(f"{consumer}: {exc}") from None
    if topology == "closed" and not lay.closed:
        raise ValueError(f"{consumer} supports closed loops only (topology='closed'); its "
                         f"source has layout.closed=False")
    return Layout(lay.num_envs, lay.stride, _canonical_device(lay.device), lay.closed), \
        bound


def check_source(obj, protocol, *, consumer: str, topology: str = "closed",
                 members: "tuple[str, ...] | None" = None, optional=()) -> Layout:
    """Check that ``obj`` holds what a consumer reads; return its :class:`Layout`.

    The members of ``protocol`` (or the narrower ``members``) are read with ``getattr``
    (a ``__getattr__`` proxy works; ``obj.validate`` is never called): a missing (or
    ``None``) member raises ``TypeError``. ``protocol`` may be a tuple of protocols
    (``(HasCenterline, HasGeneration)``): their members are checked together, against
    one layout. The layout is ``obj.layout`` when it is a :class:`Layout` (on a
    ``Track`` / ``GateSequence`` / ``Centerline`` it must be one, or ``None``; another
    object's unrelated ``layout`` attribute is ignored), else inferred from the checked
    members only: ``num_envs`` from ``count`` (else the first per-env member),
    ``stride`` from the first per-point member (``1``, i.e. undefined, when the checked
    members are all per env, e.g. ``HasGeneration`` alone), the device of the member
    that gave ``num_envs``, ``closed=True``. Only those members are checked (their
    dtype and per-point / per-env length come from the result schemas: ``center`` is
    ``[E * stride]`` vec3f, ``count`` ``[E]`` int32, ...), as flat 1-D arrays on the
    layout's device, contiguous; fields outside them are never looked at. Host only.

    Args:
        obj: The source (a ``Track``, ``GateSequence``, ``Centerline``, or any object
            with the members).
        protocol: The capability protocol (``HasCenterline``, ``HasBand``, ...), or a
            tuple of them.
        consumer: The consumer's name, prefixed to every message
            (``"CollisionChecker"``).
        topology: The consumer's topology: ``"closed"`` refuses a source whose layout
            has ``closed=False``; ``"any"`` accepts both.
        members: The members the consumer reads when it is a subset of the protocol's.
        optional: Members of ``protocol`` read only when present (roadmap PR14): each
            one the source has (not ``None``) is checked like a required member (same
            messages), a missing one is skipped (``valid`` / ``generation``: a consumer
            then falls back to "every env valid" / "no self-reset").

    Returns:
        The source's :class:`Layout` (canonical device).

    Raises:
        TypeError: ``protocol`` is not a capability protocol (or a tuple of them); a
            missing member (naming the protocol, its members and the missing ones); a
            TrackGen result whose ``layout`` is neither a ``Layout`` nor ``None``.
        ValueError: ``members`` (or ``optional``) that are not members of ``protocol``;
            a member that
            does not match (``CollisionChecker: Track.inner: expected a contiguous
            [3072] vec3f ([E*stride] = [8*384]) array on cpu, got shape (3064,) vec3f on
            cpu``); a ``closed=False`` source for a closed-only consumer.
    """
    return _check_and_bind(obj, protocol, consumer=consumer, topology=topology,
                           members=members, optional=optional)[0]
