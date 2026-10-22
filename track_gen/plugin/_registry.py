"""The process-global :class:`Registry` behind :mod:`track_gen.plugin`.

One registry holds every pluggable spec, keyed by ``(kind, name)``. Kinds are an open
set of strings: ``"generator"`` (:class:`~track_gen.plugin.GeneratorSpec`) and
``"gate_generator"`` (:class:`~track_gen.plugin.GateGeneratorSpec`) today; later
releases add more (``"relax"``, ``"elevation"``, ...).

Loading (built-ins, then the ``"track_gen.plugins"`` entry points), the
``TRACK_GEN_PLUGINS`` policy and quarantine are described in the :mod:`track_gen.plugin`
docstring, which is the rendered reference.
"""
from __future__ import annotations

import collections
import contextlib
import dataclasses
import functools
import importlib.metadata as _metadata
import os
import re
import sys
import threading
import types
import warnings
from typing import Any, Iterator, Mapping

from .._src.gate_generator_registry import GateGeneratorSpec
from .._src.generator_registry import GeneratorSpec
from .params import NoParams, Params, _check_params_class, _schema, _static_view
from .stage import CaptureClass, Stage, _check_stage, _is_stage

# The legacy spec types that may declare a static Params class through ``params=``.
_PARAM_SPECS = (GeneratorSpec, GateGeneratorSpec)

__all__ = [
    "API_VERSION",
    "ENTRY_POINT_GROUP",
    "POLICY_ENV",
    "DuplicateSpecError",
    "PluginLoadError",
    "PluginOverrideWarning",
    "Registry",
    "SpecInfo",
    "registry",
]

#: Version of the plugin API this track_gen provides. A plugin declares the version it
#: targets as ``track_gen_api = <int>`` in the module its entry point names (or on its
#: register callable); a newer one is quarantined instead of being called.
API_VERSION = 1

#: ``importlib.metadata`` entry-point group scanned for third-party plugins.
ENTRY_POINT_GROUP = "track_gen.plugins"

#: Environment variable selecting the third-party plugin policy: warn | strict | off.
POLICY_ENV = "TRACK_GEN_PLUGINS"
_POLICIES = ("warn", "strict", "off")

# Kinds that always exist: their per-kind dicts back the GENERATORS / GATE_GENERATORS
# views of the deprecated _src shims, so they are never dropped.
_DEFAULT_KINDS = ("generator", "gate_generator")


def entry_points(**params):
    """``importlib.metadata.entry_points``, looked up at call time: monkeypatching either
    this name or ``importlib.metadata.entry_points`` takes effect."""
    return _metadata.entry_points(**params)


class DuplicateSpecError(ValueError):
    """A spec was added under a ``(kind, name)`` that is already registered (and
    ``replace=True`` was not passed). The message names both origins."""


class PluginOverrideWarning(UserWarning):
    """``add(..., replace=True)`` replaced an existing spec. The message names both
    origins."""


class PluginLoadError(RuntimeError):
    """A third-party plugin failed to load: raised at load under
    ``TRACK_GEN_PLUGINS=strict``, and by :meth:`Registry.get` when a name provided by
    a quarantined plugin is requested (chained from the original error)."""


@dataclasses.dataclass(frozen=True)
class SpecInfo:
    """What :meth:`Registry.describe` reports about one registered spec.

    Attributes:
        kind: The registry kind (``"generator"``, ``"gate_generator"``, ...).
        name: The registered name.
        origin: The module that defines the spec's callables.
        distribution: ``(name, version)`` of the distribution whose entry point
            registered the spec; ``None`` for built-ins and for specs added at run time.
        capturable: The spec's ``capturable`` flag (``True`` when it has none). For a
            Stage: ``capture != HOST``, computed from the DEFAULT Params; a stage whose
            :meth:`~track_gen.plugin.Stage.capture_class` depends on its params may
            differ for other values (the generator reports its own).
        accepts_ctx: The spec's ``accepts_ctx`` flag (``False`` when it has none).
        builtin: ``True`` for the specs shipped with track_gen.
        entry_point: The name of the entry point that registered the spec, or ``None``.
        params_schema: The JSON schema (draft 2020-12) of the spec's Params class:
            per field ``type``, ``default``, ``minimum`` / ``maximum`` (from lo / hi),
            ``enum`` (choices), ``description`` (doc), ``x-static`` or ``x-dynamic``
            (with ``x-dtype``) and ``x-ui``. ``{}`` for a spec without parameters (a
            legacy spec without a ``params=`` class, or a Stage with ``NoParams``).
            Always ``json.dumps``-able.
        capture: A Stage's :class:`~track_gen.plugin.CaptureClass` for its default
            Params; ``None`` for a spec that is not a Stage.
        output: A Stage's ``output`` (``"centerline"``); ``None`` for a spec that is not
            a Stage.
        writes: A Stage's ``writes`` (``local port -> channel``); ``{}`` otherwise.
        provides: A Stage's ``provides``; empty otherwise.
    """

    kind: str
    name: str
    origin: str
    distribution: "tuple[str, str] | None"
    capturable: bool
    accepts_ctx: bool
    builtin: bool = False
    entry_point: "str | None" = None
    params_schema: Mapping[str, Any] = dataclasses.field(default_factory=dict, hash=False)
    capture: "CaptureClass | None" = None
    output: "str | None" = None
    writes: Mapping[str, str] = dataclasses.field(default_factory=dict, hash=False)
    provides: frozenset = frozenset()


@dataclasses.dataclass(frozen=True)
class _Origin:
    module: str
    builtin: bool = False
    distribution: "tuple[str, str] | None" = None
    entry_point: "str | None" = None

    def __str__(self) -> str:
        if self.builtin:
            return f"built-in module {self.module!r}"
        text = f"module {self.module!r}"
        if self.entry_point is not None:
            text += f" (plugin {self.entry_point!r}{_dist_text(self.distribution)})"
        return text


@dataclasses.dataclass
class _PluginLoad:
    """The third-party plugin being loaded (one at a time, under the lock)."""

    entry_point: str
    distribution: "tuple[str, str] | None"
    module: str = ""                                                # entry-point module
    value: str = ""                                                 # "module:attr"
    attempted: set = dataclasses.field(default_factory=set)        # {(kind, name)}
    added: set = dataclasses.field(default_factory=set)            # {(kind, name)}
    refused: list = dataclasses.field(default_factory=list)        # [Exception]


@dataclasses.dataclass(frozen=True)
class _Quarantine:
    entry_point: str
    distribution: "tuple[str, str] | None"
    error: BaseException
    message: str
    module: str              # the module the entry point names ("" when unknown)
    value: str               # the entry point's "module:attr"
    names: frozenset         # (kind, name) it tried to register before failing

    def key(self, qualified: bool) -> str:
        """Its key in :attr:`Registry.quarantined`: the entry-point name, qualified by
        the distribution when another quarantined plugin shares that name."""
        if not qualified:
            return self.entry_point
        if self.distribution is not None:
            return f"{self.distribution[0]}:{self.entry_point}"
        return f"{self.value or '?'}:{self.entry_point}"

    def claims(self, name: str) -> bool:
        """Whether ``name`` looks like one this plugin would provide: it is the
        entry-point name or the entry point's module, or is namespaced under one of
        them, or its first dotted component is the (normalized) distribution name."""
        for prefix in (self.entry_point, self.module):
            if prefix and (name == prefix or name.startswith(prefix + ".")):
                return True
        head = _normalize(name.split(".")[0])
        if head == _normalize(self.entry_point):
            return True
        return self.distribution is not None and head == _normalize(self.distribution[0])


@dataclasses.dataclass(frozen=True)
class _Snapshot:
    specs: dict
    meta: dict
    quarantine: list
    warned: frozenset
    loaded: bool


def _dist_text(dist: "tuple[str, str] | None") -> str:
    return "" if dist is None else f" from distribution {dist[0]} {dist[1]}"


def _same_items(a: dict, b: dict) -> bool:
    """Same keys mapped to the identical objects (never calls a spec's ``__eq__``)."""
    return len(a) == len(b) and all(k in a and a[k] is v for k, v in b.items())


def _kind_label(kind: str) -> str:
    return kind.replace("_", " ")


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "_", name).lower()


def _distribution(ep) -> "tuple[str, str] | None":
    """``(name, version)`` of the entry point's distribution, or ``None``. Total: a
    distribution with unreadable or missing metadata yields ``None`` (or ``"?"`` for
    the version) instead of raising, so it cannot break discovery."""
    try:
        dist = getattr(ep, "dist", None)
        if dist is None:
            return None
        name = getattr(dist, "name", None)
        if not name:
            name = dist.metadata["Name"]
        if not name:
            return None
        try:
            version = dist.version
        except Exception:  # noqa: BLE001 - metadata is best effort
            version = None
        return str(name), str(version) if version is not None else "?"
    except Exception:  # noqa: BLE001 - metadata is best effort
        return None


def _entry_point_module(ep) -> str:
    try:
        module = getattr(ep, "module", None)
        if not module:
            module = str(getattr(ep, "value", "") or "").partition(":")[0].strip()
        return str(module)
    except Exception:  # noqa: BLE001 - metadata is best effort
        return ""


def _defining_function(value):
    """The function behind a callable: through ``functools.partial`` (``.func``) and
    decorators (``__wrapped__``)."""
    for _ in range(32):                      # bounded: a proxy may fake attributes
        if isinstance(value, functools.partial):
            value = value.func
        elif isinstance(value, (types.FunctionType, types.MethodType)) and \
                getattr(value, "__wrapped__", None) is not None:
            value = value.__wrapped__
        else:
            break
    return value


def _origin_module(spec, fallback: "str | None" = None) -> str:
    """The module defining the spec's callables (a ``functools.partial`` or a
    decorated function counts as the function it wraps); else ``fallback`` (the module
    that called ``add``); else the spec type's module."""
    for attr in ("generate", "alloc_scratch", "max_gates", "build"):
        value = getattr(spec, attr, None)
        if value is None:
            continue
        module = getattr(_defining_function(value), "__module__", None)
        if isinstance(module, str) and module:
            return module
    if isinstance(spec, type):
        return spec.__module__
    return fallback or type(spec).__module__


def _caller_module(depth: int) -> "str | None":
    try:
        return sys._getframe(depth + 1).f_globals.get("__name__")
    except ValueError:  # pragma: no cover - shallower stack than expected
        return None


def _infer_kind(spec, kind: "str | None") -> str:
    if _is_stage(spec):
        declared = spec.kind
    else:
        declared = getattr(spec if isinstance(spec, type) else type(spec), "registry_kind",
                           None)
    if kind is None:
        if not declared:
            raise TypeError(
                f"cannot infer the registry kind of a {type(spec).__name__}; pass "
                f"kind=... (e.g. registry.add(spec, kind='relax'))")
        return str(declared)
    if not isinstance(kind, str) or not kind:
        raise TypeError(f"kind must be a non-empty string, got {kind!r}")
    if declared and declared != kind:
        raise ValueError(
            f"a {type(spec).__name__} registers as kind {declared!r}, not {kind!r}")
    return kind


def _check_spec(spec, kind: "str | None") -> None:
    """Declaration checks run before a spec is inserted (they raise, which quarantines
    a plugin under ``warn``): a Stage's rules, and a ``params=`` class's rules."""
    if isinstance(spec, Stage) or _is_stage(spec):
        _check_stage(spec, kind=kind, api_version=API_VERSION)
        return
    if isinstance(spec, _PARAM_SPECS) and spec.params is not None:
        _check_params_class(spec.params, allow_dynamic=False,
                            owner=f"{type(spec).__name__} {spec.name!r}")


def _params_class(spec):
    """The Params class a spec declares (``Stage.Params``, or a legacy spec's
    ``params=``), or ``None``."""
    if _is_stage(spec):
        return spec.Params
    params_cls = getattr(spec, "params", None) if isinstance(spec, _PARAM_SPECS) else None
    return params_cls if isinstance(params_cls, type) and issubclass(params_cls, Params) \
        else None


def _spec_name(spec) -> str:
    name = getattr(spec, "name", None)
    if not isinstance(name, str) or not name or ":" in name or name != name.strip():
        raise ValueError(
            f"spec name must be a non-empty string without ':' or surrounding "
            f"whitespace, got {name!r}")
    return name


def _policy() -> str:
    raw = os.environ.get(POLICY_ENV, "")
    policy = raw.strip().lower() or "warn"
    if policy not in _POLICIES:
        # Never break the built-ins over a typo: fall back to the default.
        warnings.warn(
            f"{POLICY_ENV}={raw!r} is not one of {', '.join(_POLICIES)}; using 'warn'",
            RuntimeWarning, stacklevel=4)
        policy = "warn"
    return policy


def _declared_api(fn, ep) -> "tuple[Any, str] | None":
    """The plugin API version a plugin declares, if any.

    ``track_gen_api`` is authoritative, looked up on the register callable, then on the
    module the entry point names, then on the callable's own module (``register`` may
    be re-exported from a submodule). ``API_VERSION`` in the same places is honoured
    only when it is an ``int``: a module's unrelated ``API_VERSION = "v3"`` is ignored.
    """
    holders = [(fn, "register callable")]
    seen = set()
    for module_name in (_entry_point_module(ep), getattr(fn, "__module__", None)):
        if not isinstance(module_name, str) or not module_name or module_name in seen:
            continue
        seen.add(module_name)
        module = sys.modules.get(module_name)
        if module is not None:
            holders.append((module, f"module {module_name!r}"))
    for holder, where in holders:
        value = getattr(holder, "track_gen_api", None)
        if value is not None:
            return value, f"{where}.track_gen_api"
    for holder, where in holders:
        value = getattr(holder, "API_VERSION", None)
        if isinstance(value, int) and not isinstance(value, bool):
            return value, f"{where}.API_VERSION"
    return None


class Registry:
    """Specs keyed by ``(kind, name)``. Thread-safe.

    Loading, the ``TRACK_GEN_PLUGINS`` policy and quarantine are described in
    :mod:`track_gen.plugin`. The process-global instance is :data:`registry`.
    Construct a separate ``Registry()`` only for isolated experiments; the library's
    facades resolve names through the global one.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._loaded = False
        self._loading = False
        # kind -> {name: spec}. The per-kind dicts of the default kinds are never
        # replaced (scoped() restores them in place), so read-only views stay live.
        self._specs: dict[str, dict[str, Any]] = {k: {} for k in _DEFAULT_KINDS}
        self._meta: dict[str, dict[str, _Origin]] = {k: {} for k in _DEFAULT_KINDS}
        self._quarantine: list[_Quarantine] = []      # in load order
        self._quarantined: dict[str, str] = {}        # public view: key -> error text
        self._warned: set = set()
        self._builtin = False
        self._plugin: "_PluginLoad | None" = None

    def __repr__(self) -> str:
        counts = {k: len(v) for k, v in self._specs.items() if v}
        return f"<track_gen.plugin.Registry loaded={self._loaded} specs={counts}>"

    # -- public API -------------------------------------------------------------------
    def add(self, spec, *, kind: "str | None" = None, replace: bool = False) -> None:
        """Register ``spec`` under ``(kind, spec.name)``.

        Adding the identical spec object again under the same kind is a no-op.

        Args:
            spec: The spec (``GeneratorSpec``, ``GateGeneratorSpec``, or any object
                with a ``name``).
            kind: Inferred for ``GeneratorSpec`` (``"generator"``) and
                ``GateGeneratorSpec`` (``"gate_generator"``); required otherwise.
            replace: Replace an existing spec of that name, emitting a
                :class:`PluginOverrideWarning` that names both origins.

        Raises:
            DuplicateSpecError: The name is already registered in that kind (by a
                different spec object) and ``replace`` is false. Raised from a
                plugin's ``register(reg)``, it quarantines that plugin (or raises under
                ``strict``).
            TypeError: The kind cannot be inferred, or is not a non-empty string.
            ValueError: The kind does not match the spec type, or the name is not a
                non-empty string.

        Third-party names should be dotted (``"package.name"``); a new non-dotted name
        added outside the built-ins emits a one-time ``UserWarning``.

        A :class:`~track_gen.plugin.Stage` is registered as its CLASS (``kind`` comes
        from ``Stage.kind``); its declaration is checked first (see
        :mod:`track_gen.plugin.stage`). A spec with a ``params=`` class has the class
        checked (frozen dataclass, every field with a default, static fields only).
        """
        self._add(spec, kind=kind, replace=replace, legacy=False, stacklevel=3)

    def get(self, kind: str, name: str):
        """The spec registered under ``(kind, name)``.

        Raises:
            PluginLoadError: ``name`` belongs to a quarantined plugin (chained from the
                plugin's original load error).
            ValueError: Unknown name; the message lists the available names and the
                quarantined plugins with their errors.
        """
        self._ensure_loaded()
        if isinstance(name, str):
            spec = self._specs.get(kind, {}).get(name)
            if spec is not None:
                return spec
        with self._lock:
            if isinstance(name, str):
                spec = self._specs.get(kind, {}).get(name)
                if spec is not None:
                    return spec
                quarantine, exact = self._provider(kind, name)
                if quarantine is not None:
                    raise PluginLoadError(self._quarantined_message(
                        kind, name, quarantine, exact)) from quarantine.error
            raise ValueError(self._unknown_message(kind, name))

    def available(self, kind: "str | None" = None) -> list[str]:
        """Sorted registered names of ``kind``; with ``kind=None``, every spec as
        ``"kind:name"``."""
        self._ensure_loaded()
        with self._lock:
            if kind is None:
                return sorted(f"{k}:{n}" for k, table in self._specs.items() for n in table)
            return sorted(self._specs.get(kind, {}))

    def unregister(self, kind: str, name: str) -> None:
        """Remove ``(kind, name)``. Raises ``ValueError`` when it is not registered.

        Inside a plugin's ``register(reg)``, a plugin may only unregister names it
        registered itself (override others with ``add(spec, replace=True)``)."""
        self._ensure_loaded()
        with self._lock:
            table = self._specs.get(kind, {})
            if name not in table:
                raise ValueError(self._unknown_message(kind, name))
            plugin = self._plugin
            if plugin is not None and (kind, name) not in plugin.added:
                err = PluginLoadError(
                    f"track_gen plugin {plugin.entry_point!r}{_dist_text(plugin.distribution)} "
                    f"may only unregister names it registered itself; "
                    f"{_kind_label(kind)} {name!r} is registered by "
                    f"{self._meta[kind].get(name) or _Origin(_origin_module(table[name]))}. "
                    f"Use add(spec, replace=True) to override it.")
                plugin.refused.append(err)
                raise err
            del table[name]
            del self._meta[kind][name]

    def describe(self, kind: str, name: str) -> SpecInfo:
        """Provenance and execution flags of ``(kind, name)`` (see :class:`SpecInfo`).
        Raises like :meth:`get`, and ``TypeError`` for a Stage whose ``capture_class``
        does not return a ``CaptureClass``."""
        with self._lock:
            spec = self.get(kind, name)
            origin = self._meta[kind][name]
        params_cls = _params_class(spec)
        stage_info = {}
        if _is_stage(spec):
            capture = spec.capture_class(_static_view(params_cls()))
            if not isinstance(capture, CaptureClass):
                raise TypeError(f"{spec.__name__}.capture_class() must return a "
                                f"CaptureClass, got {capture!r}")
            capturable = capture != CaptureClass.HOST
            accepts_ctx = True
            stage_info = dict(capture=capture, output=spec.output,
                              writes=types.MappingProxyType(dict(spec.writes)),
                              provides=frozenset(spec.provides))
        else:
            capturable = bool(getattr(spec, "capturable", True))
            accepts_ctx = bool(getattr(spec, "accepts_ctx", False))
        has_params = params_cls is not None and params_cls is not NoParams
        return SpecInfo(
            kind=kind,
            name=name,
            origin=origin.module,
            distribution=origin.distribution,
            capturable=capturable,
            accepts_ctx=accepts_ctx,
            builtin=origin.builtin,
            entry_point=origin.entry_point,
            params_schema=_schema(params_cls) if has_params else {},
            **stage_info,
        )

    @property
    def quarantined(self) -> Mapping[str, str]:
        """A read-only mapping of each quarantined plugin to the error text that
        quarantined it.

        Keys are entry-point names; when two quarantined plugins share an entry-point
        name, each of their keys is qualified as ``"<distribution>:<entry point>"``.
        """
        self._ensure_loaded()
        return types.MappingProxyType(self._quarantined)

    @contextlib.contextmanager
    def scoped(self, *, fresh: bool = False) -> Iterator["Registry"]:
        """Snapshot every kind (and the load/quarantine state) on enter, restore it on
        exit; for tests. Lookups from other threads while a scope is entered or exited
        may observe the change mid-way.

        Args:
            fresh: Start the scope from an unloaded, empty registry, so the next lookup
                re-runs built-in and third-party discovery (e.g. under a monkeypatched
                entry-point list). Without it the registry is loaded on enter.
        """
        with self._lock:
            if self._loading:
                raise RuntimeError("Registry.scoped() cannot be entered while loading")
            if not fresh:
                self._ensure_loaded()
            snapshot = self._snapshot()
            if fresh:
                self._clear()
        try:
            yield self
        finally:
            with self._lock:
                self._restore(snapshot)

    # -- private helpers used by the deprecated _src shims ----------------------------
    def _table(self, kind: str) -> dict:
        """The live ``{name: spec}`` dict of ``kind`` (stable identity)."""
        with self._lock:
            self._meta.setdefault(kind, {})
            return self._specs.setdefault(kind, {})

    def _add_legacy(self, spec, kind: str, *, same_module_only: bool = False) -> None:
        """Old ``register(spec)`` semantics: overwrite an existing name, with a
        ``DeprecationWarning``. With ``same_module_only`` (the old gate registry), only
        a spec from the same module may overwrite; another module's raises
        :class:`DuplicateSpecError`."""
        self._add(spec, kind=kind, replace=True, legacy=True, stacklevel=4,
                  same_module_only=same_module_only)

    # -- implementation ---------------------------------------------------------------
    def _add(self, spec, *, kind, replace, legacy, stacklevel,
             same_module_only=False) -> None:
        _check_spec(spec, kind)
        kind = _infer_kind(spec, kind)
        name = _spec_name(spec)
        self._ensure_loaded()
        caller = _caller_module(stacklevel - 1)
        with self._lock:
            origin = self._origin(spec, caller)
            table = self._specs.setdefault(kind, {})
            meta = self._meta.setdefault(kind, {})
            plugin = self._plugin
            if plugin is not None:
                plugin.attempted.add((kind, name))
            if table.get(name) is spec:
                return                    # the identical spec again: nothing to do
            previous = None
            if name in table:
                previous = meta.get(name) or _Origin(_origin_module(table[name]))
                if same_module_only and \
                        _origin_module(table[name]) != _origin_module(spec):
                    replace = False
            label = _kind_label(kind)
            if previous is not None:
                if not replace:
                    hint = ("Use track_gen.plugin.registry.add(spec, replace=True) to "
                            "override it" if legacy else "Pass replace=True to override it")
                    err = DuplicateSpecError(
                        f"{label} {name!r} is already registered by {previous}; refusing "
                        f"the registration from {origin}. {hint}, or choose another "
                        f"(dotted 'package.name') name.")
                    if plugin is not None:
                        plugin.refused.append(err)
                    raise err
                if legacy:
                    warnings.warn(
                        f"{label} {name!r} registered by {previous} was replaced through "
                        f"the deprecated track_gen._src registry register(); use "
                        f"track_gen.plugin.registry.add(spec, replace=True)",
                        DeprecationWarning, stacklevel=stacklevel)
                else:
                    warnings.warn(PluginOverrideWarning(
                        f"{label} {name!r} registered by {previous} is overridden by "
                        f"{origin} (replace=True)"), stacklevel=stacklevel)
            elif not legacy and not origin.builtin and "." not in name \
                    and (kind, name) not in self._warned:
                self._warned.add((kind, name))
                warnings.warn(
                    f"{label} {name!r} from {origin} is not a dotted name; third-party "
                    f"names should be namespaced as 'package.name' so they cannot "
                    f"collide with built-ins or other plugins", UserWarning,
                    stacklevel=stacklevel)
            table[name] = spec
            meta[name] = origin
            if plugin is not None:
                plugin.added.add((kind, name))

    def _origin(self, spec, caller: "str | None" = None) -> _Origin:
        module = _origin_module(spec, caller)
        if self._builtin:
            return _Origin(module, builtin=True)
        if self._plugin is not None:
            return _Origin(module, distribution=self._plugin.distribution,
                           entry_point=self._plugin.entry_point)
        return _Origin(module)

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._lock:
            # _loading: a plugin's register(reg) re-entering (same thread, RLock).
            if self._loaded or self._loading:
                return
            policy = _policy()
            self._loading = True
            before = self._snapshot()
            try:
                self._load_builtins()
                if policy != "off":
                    self._load_plugins(policy)
            except BaseException:
                self._restore(before)
                raise
            else:
                self._loaded = True
            finally:
                self._loading = False

    def _load_builtins(self) -> None:
        from .._src import builtin_plugins

        self._builtin = True
        try:
            builtin_plugins.register(self)
        finally:
            self._builtin = False

    def _load_plugins(self, policy: str) -> None:
        try:
            found = list(entry_points(group=ENTRY_POINT_GROUP))
        except Exception as exc:  # noqa: BLE001 - corrupt metadata must not break built-ins
            if policy == "strict":
                raise PluginLoadError(
                    f"discovering track_gen plugins failed: {type(exc).__name__}: {exc}"
                ) from exc
            warnings.warn(
                f"discovering track_gen plugins failed, third-party plugins are skipped: "
                f"{type(exc).__name__}: {exc}", RuntimeWarning, stacklevel=2)
            return

        # _distribution is total, so one distribution's unreadable metadata can neither
        # break the sort nor the other plugins (it loads with distribution None).
        plugins = [(ep, _distribution(ep)) for ep in found]

        def order(item):
            ep, dist = item
            return (_normalize(dist[0]) if dist else "", str(ep.name),
                    str(getattr(ep, "value", "") or ""))

        for ep, dist in sorted(plugins, key=order):
            self._load_entry_point(ep, dist, policy)

    def _load_entry_point(self, ep, dist, policy: str) -> None:
        load = _PluginLoad(str(ep.name), dist, module=_entry_point_module(ep),
                           value=str(getattr(ep, "value", "") or ""))
        before = self._snapshot()
        self._plugin = load
        try:
            fn = ep.load()
            declared = _declared_api(fn, ep)
            if declared is not None:
                value, where = declared
                if isinstance(value, bool) or not isinstance(value, int):
                    raise PluginLoadError(
                        f"{where} must be an int plugin API version, got {value!r}")
                if value > API_VERSION:
                    raise PluginLoadError(
                        f"requires track_gen plugin API {value} ({where}), but this "
                        f"track_gen provides API {API_VERSION}; upgrade track_gen or use "
                        f"an older release of the plugin")
            if not callable(fn):
                raise TypeError(
                    f"entry point {ep.name!r} resolves to {fn!r}, which is not "
                    f"callable; expected a register(registry) function")
            fn(self)
            if load.refused:
                # A swallowed refusal still quarantines the plugin: built-ins (and
                # earlier plugins) win unless the plugin passes replace=True.
                raise load.refused[0]
        except Exception as exc:  # noqa: BLE001 - a broken plugin must not break track_gen
            self._restore(before, tables_only=True)
            self._quarantine_plugin(load, exc, policy)
        finally:
            self._plugin = None

    def _quarantine_plugin(self, load: _PluginLoad, exc: Exception, policy: str) -> None:
        dist = load.distribution
        message = f"{type(exc).__name__}: {exc}"
        if policy == "strict":
            raise PluginLoadError(
                f"track_gen plugin {load.entry_point!r}{_dist_text(dist)} failed to load "
                f"({POLICY_ENV}=strict): {message}") from exc
        self._quarantine.append(_Quarantine(
            entry_point=load.entry_point, distribution=dist, error=exc, message=message,
            module=load.module, value=load.value, names=frozenset(load.attempted)))
        self._sync_quarantined()
        warnings.warn(
            f"track_gen plugin {load.entry_point!r}{_dist_text(dist)} failed to load and "
            f"is quarantined: {message}. Set {POLICY_ENV}=strict to raise instead, or "
            f"{POLICY_ENV}=off to skip third-party plugins.", RuntimeWarning, stacklevel=2)

    def _sync_quarantined(self) -> None:
        """Rebuild the public ``quarantined`` dict (in place: views stay live)."""
        counts = collections.Counter(q.entry_point for q in self._quarantine)
        self._quarantined.clear()
        for q in self._quarantine:
            self._quarantined[q.key(counts[q.entry_point] > 1)] = q.message

    def _provider(self, kind: str, name: str) -> "tuple[_Quarantine | None, bool]":
        """The quarantined plugin that provides ``(kind, name)``, and whether that is
        certain (it tried to register exactly that name) rather than inferred from the
        name's namespace (see :meth:`_Quarantine.claims`)."""
        for quarantine in self._quarantine:
            if (kind, name) in quarantine.names:
                return quarantine, True
        for quarantine in self._quarantine:
            if quarantine.claims(name):
                return quarantine, False
        return None, False

    def _quarantined_message(self, kind: str, name: str, quarantine: _Quarantine,
                             exact: bool) -> str:
        verb = "is provided by" if exact else "may be provided by"
        message = (f"{_kind_label(kind)} {name!r} {verb} track_gen plugin "
                   f"{quarantine.entry_point!r}{_dist_text(quarantine.distribution)}, "
                   f"which failed to load and is quarantined: {quarantine.message}")
        if not exact:
            message += f"; available: {sorted(self._specs.get(kind, {}))}"
        return message

    def _unknown_message(self, kind: str, name) -> str:
        with self._lock:
            message = (f"unknown {_kind_label(kind)} {name!r}; available: "
                       f"{sorted(self._specs.get(kind, {}))}")
            if kind not in self._specs:
                message += f" (unknown kind {kind!r}; kinds: {sorted(self._specs)})"
            if self._quarantine:
                parts = [f"{q.entry_point!r}{_dist_text(q.distribution)}: {q.message}"
                         for q in self._quarantine]
                message += "; quarantined plugins: " + "; ".join(parts)
            return message

    def _snapshot(self) -> _Snapshot:
        return _Snapshot(
            specs={k: dict(v) for k, v in self._specs.items()},
            meta={k: dict(v) for k, v in self._meta.items()},
            quarantine=list(self._quarantine),
            warned=frozenset(self._warned),
            loaded=self._loaded,
        )

    def _restore(self, snapshot: _Snapshot, *, tables_only: bool = False) -> None:
        # In place (live views stay valid), and only the kinds that changed. A kind
        # created since the snapshot is dropped (the default kinds always stay).
        for kind in set(self._specs) | set(self._meta) | set(snapshot.specs):
            if kind not in snapshot.specs and kind not in _DEFAULT_KINDS:
                self._specs.pop(kind, None)
                self._meta.pop(kind, None)
                continue
            for current, wanted in ((self._specs.setdefault(kind, {}),
                                     snapshot.specs.get(kind, {})),
                                    (self._meta.setdefault(kind, {}),
                                     snapshot.meta.get(kind, {}))):
                if not _same_items(current, wanted):
                    current.clear()
                    current.update(wanted)
        if tables_only:
            return
        self._quarantine[:] = snapshot.quarantine
        self._sync_quarantined()
        self._warned = set(snapshot.warned)
        self._loaded = snapshot.loaded

    def _clear(self) -> None:
        for kind in list(self._specs):
            if kind not in _DEFAULT_KINDS:
                del self._specs[kind]
                self._meta.pop(kind, None)
                continue
            self._specs[kind].clear()
            self._meta.setdefault(kind, {}).clear()
        for kind in list(self._meta):
            if kind not in self._specs:
                del self._meta[kind]
        self._quarantine.clear()
        self._quarantined.clear()
        self._warned = set()
        self._loaded = False


#: The process-global registry the facades (``TrackGenerator``, ``GateGenerator``,
#: ``Course``) resolve ``config.generator`` through.
registry = Registry()
