"""``track_gen.plugin``: one public Registry, built-ins + entry-point plugins (roadmap PR6).

A fake distribution (``tests._plugin_toys``: a monkeypatched ``entry_points`` returning
EntryPoint-like objects) registers toy generators that run through the public facades;
the duplicate, quarantine, API-version, policy, scoping, threading and legacy-shim
semantics are checked against it. Everything runs on CPU.
"""
from __future__ import annotations

import dataclasses
import sys
import threading
import time
import types
import warnings

import numpy as np
import pytest

from tests._plugin_toys import (
    TOY_CIRCLE,
    TOY_GATES,
    TOY_RING,
    FakeDist,
    FakeEntryPoint,
    install_entry_points,
    register_toys,
)
from track_gen import GateGenConfig, GateGenerator, PerEnvSeededRNG, TrackGenConfig, TrackGenerator
from track_gen._src import builtin_plugins, gate_generator_registry, generator_registry
from track_gen.plugin import (
    API_VERSION,
    ENTRY_POINT_GROUP,
    POLICY_ENV,
    DuplicateSpecError,
    GateGeneratorSpec,
    GeneratorSpec,
    PluginLoadError,
    PluginOverrideWarning,
    Registry,
    SpecInfo,
    registry,
)

E = 4
BUILTIN_TRACK = {"bezier", "checkpoint", "hull", "polar", "repulsive", "voronoi"}
BUILTIN_GATES = {"bezier", "checkpoint", "hull", "polar", "voronoi"}
TOY_DIST = FakeDist("toy-plugins", "1.2.3")


@pytest.fixture
def fresh(monkeypatch):
    """The global registry, emptied and unloaded for the test (restored afterwards), with
    a fake ``entry_points``: ``fresh(*eps, policy=...)`` installs the entry points and
    returns the list of groups ``entry_points`` was queried with."""
    monkeypatch.delenv(POLICY_ENV, raising=False)

    def install(*eps, policy=None):
        if policy is not None:
            monkeypatch.setenv(POLICY_ENV, policy)
        return install_entry_points(monkeypatch, *eps)

    with registry.scoped(fresh=True):
        yield install


def _toy_ep(**kw):
    return FakeEntryPoint("toy", register_toys, TOY_DIST, **kw)


def _spec(name, module="tests.fake_plugin"):
    def generate(*args, **kwargs):
        return None

    generate.__module__ = module
    return GeneratorSpec(name=name, alloc_scratch=lambda config: None, generate=generate)


def _rng(seed=0):
    return PerEnvSeededRNG(seeds=seed, num_envs=E, device="cpu")


# ---------------------------------------------------------------------------------------
# A fake distribution's generators run through the public facades
# ---------------------------------------------------------------------------------------

def test_plugin_track_generator_runs_through_track_generator(fresh):
    fresh(_toy_ep())
    track = TrackGenerator(TrackGenConfig(generator="toy.circle", device="cpu",
                                          num_envs=E), _rng(5)).generate()
    valid, count = track.valid.numpy(), track.count.numpy()
    assert valid.tolist() == [1] * E
    assert (count > 8).all()
    center = track.center.numpy()
    n_max = center.shape[0] // E
    for e in range(E):
        pts = center[e * n_max: e * n_max + count[e]]
        assert np.isfinite(pts).all()
        radius = np.linalg.norm(pts[:, :2], axis=1)
        assert radius.min() > 0.4 and radius.max() < 0.8     # still a ~0.6 circle
    lengths = track.length.numpy()
    assert len(set(np.round(lengths, 5))) > 1               # per-env seed jitter

    assert "toy.circle" in registry.available("generator")
    assert "generator:toy.circle" in registry.available()
    info = registry.describe("generator", "toy.circle")
    assert info == SpecInfo(kind="generator", name="toy.circle",
                            origin="tests._plugin_toys",
                            distribution=("toy-plugins", "1.2.3"), capturable=True,
                            accepts_ctx=True, builtin=False, entry_point="toy")


def test_plugin_gate_generator_runs_through_gate_generator(fresh):
    fresh(_toy_ep())
    cfg = GateGenConfig(generator="toy.ring", gate_ordering="raw", device="cpu",
                        num_envs=E, gate_width=0.05)
    gates = GateGenerator(cfg, _rng(7)).generate()
    assert gates.valid.numpy().tolist() == [1] * E
    assert gates.count.numpy().tolist() == [TOY_GATES] * E
    pos = gates.position.numpy().reshape(E, int(cfg.max_gates), 3)
    assert np.isfinite(pos[:, :TOY_GATES]).all()
    assert np.isnan(pos[:, TOY_GATES:]).all()
    assert "toy.ring" in registry.available("gate_generator")
    info = registry.describe("gate_generator", "toy.ring")
    assert info.distribution == ("toy-plugins", "1.2.3")
    assert info.entry_point == "toy" and not info.builtin


def test_plugin_generator_runs_through_course(fresh):
    from track_gen.course import Course, CourseConfig

    fresh(_toy_ep())
    course = Course(CourseConfig(
        mode="track", gen=TrackGenConfig(generator="toy.circle", device="cpu", num_envs=2),
        seeds=3, checkpoint_spacing=0.3))
    assert course.result.valid.numpy().tolist() == [1, 1]


def test_builtins_are_registered_explicitly_and_described(fresh):
    calls = fresh()
    assert BUILTIN_TRACK <= set(registry.available("generator"))
    assert BUILTIN_GATES <= set(registry.available("gate_generator"))
    assert calls == [ENTRY_POINT_GROUP]
    info = registry.describe("generator", "bezier")
    assert info.builtin and info.distribution is None and info.entry_point is None
    assert info.origin == "track_gen._src.warp_generate"
    assert registry.describe("gate_generator", "polar").origin == \
        "track_gen._src.warp_generate_polar_gates"


def test_entry_points_load_in_distribution_then_name_order(fresh):
    order = []

    def recorder(tag):
        def register(reg):
            order.append(tag)
        return register

    fresh(FakeEntryPoint("b", recorder("zeta/b"), FakeDist("zeta")),
          FakeEntryPoint("z", recorder("alpha/z"), FakeDist("Alpha")),
          FakeEntryPoint("a", recorder("alpha/a"), FakeDist("alpha")),
          FakeEntryPoint("m", recorder("nodist/m"), None))
    registry.available("generator")
    assert order == ["nodist/m", "alpha/a", "alpha/z", "zeta/b"]


# ---------------------------------------------------------------------------------------
# Duplicates and overrides
# ---------------------------------------------------------------------------------------

def test_duplicate_name_raises_naming_both_origins(plugin_registry):
    plugin_registry.add(_spec("acme.dup", module="acme_one"))
    with pytest.raises(DuplicateSpecError, match="already registered") as e:
        plugin_registry.add(_spec("acme.dup", module="acme_two"))
    assert isinstance(e.value, ValueError)
    assert "acme_one" in str(e.value) and "acme_two" in str(e.value)
    assert plugin_registry.get("generator", "acme.dup").generate.__module__ == "acme_one"


def test_replace_true_warns_naming_both_origins(plugin_registry):
    plugin_registry.add(_spec("acme.dup", module="acme_one"))
    new = _spec("acme.dup", module="acme_two")
    with pytest.warns(PluginOverrideWarning) as rec:
        plugin_registry.add(new, replace=True)
    assert "acme_one" in str(rec[0].message) and "acme_two" in str(rec[0].message)
    assert plugin_registry.get("generator", "acme.dup") is new


def test_overriding_a_builtin_needs_replace(plugin_registry):
    with pytest.raises(DuplicateSpecError, match="built-in module 'track_gen._src"):
        plugin_registry.add(_spec("bezier"))
    with pytest.warns(PluginOverrideWarning, match="built-in"):
        plugin_registry.add(_spec("bezier"), replace=True)
    assert not plugin_registry.describe("generator", "bezier").builtin


def test_same_name_in_another_kind_is_not_a_duplicate(plugin_registry):
    plugin_registry.add(TOY_CIRCLE)
    plugin_registry.add(GateGeneratorSpec(
        name="toy.circle", alloc_scratch=TOY_RING.alloc_scratch,
        generate=TOY_RING.generate, max_gates=TOY_RING.max_gates,
        supported_orderings=TOY_RING.supported_orderings, accepts_ctx=True))
    assert "toy.circle" in plugin_registry.available("generator")
    assert "toy.circle" in plugin_registry.available("gate_generator")


def _colliding_plugin(replace=False, swallow=False):
    def register(reg):
        reg.add(_spec("acme.ok", module="acme"))
        try:
            reg.add(_spec("bezier", module="acme"), replace=replace)
        except DuplicateSpecError:
            if not swallow:
                raise
    return register


@pytest.mark.parametrize("swallow", [False, True])
def test_plugin_colliding_with_builtin_is_quarantined_under_warn(fresh, swallow):
    fresh(FakeEntryPoint("acme", _colliding_plugin(swallow=swallow), FakeDist("acme")))
    with pytest.warns(RuntimeWarning, match="'acme'.*quarantined.*DuplicateSpecError"):
        names = registry.available("generator")
    assert "acme.ok" not in names                  # its registrations are rolled back
    assert registry.describe("generator", "bezier").builtin   # the built-in wins
    assert "DuplicateSpecError" in registry.quarantined["acme"]
    with pytest.raises(PluginLoadError) as e:
        registry.get("generator", "acme.ok")
    assert isinstance(e.value.__cause__, DuplicateSpecError)


def test_plugin_colliding_with_builtin_raises_under_strict(fresh):
    fresh(FakeEntryPoint("acme", _colliding_plugin(), FakeDist("acme")), policy="strict")
    for _ in range(2):    # nothing is half-loaded: the next lookup retries and raises again
        with pytest.raises(PluginLoadError, match="acme.*strict") as e:
            registry.get("generator", "bezier")
        assert isinstance(e.value.__cause__, DuplicateSpecError)


def test_plugin_may_override_a_builtin_with_replace(fresh):
    fresh(FakeEntryPoint("acme", _colliding_plugin(replace=True), FakeDist("acme", "2.0")))
    with pytest.warns(PluginOverrideWarning, match="built-in.*acme"):
        info = registry.describe("generator", "bezier")
    assert not info.builtin and info.distribution == ("acme", "2.0")
    assert registry.quarantined == {}


# ---------------------------------------------------------------------------------------
# Broken plugins, policy, API version
# ---------------------------------------------------------------------------------------

def test_broken_plugin_is_quarantined_and_named_in_lookup_errors(fresh):
    boom = ImportError("No module named 'acme_deps'")
    fresh(_toy_ep(), FakeEntryPoint("broken", boom, FakeDist("broken-plugin", "0.9"),
                                    value="broken_plugin:register"))
    with pytest.warns(RuntimeWarning, match="'broken'.*quarantined") as rec:
        names = registry.available("generator")
    assert len([w for w in rec if issubclass(w.category, RuntimeWarning)]) == 1
    assert "toy.circle" in names                     # the healthy plugin still loads
    assert set(registry.quarantined) == {"broken"}
    assert "acme_deps" in registry.quarantined["broken"]

    with pytest.raises(ValueError) as e:
        registry.get("generator", "does-not-exist")
    msg = str(e.value)
    assert "bezier" in msg and "toy.circle" in msg   # available names ...
    assert "broken" in msg and "acme_deps" in msg    # ... and the quarantined plugin
    assert not isinstance(e.value, PluginLoadError)

    # Requesting a name the quarantined plugin provides re-raises its load error.
    for name in ("broken", "broken.thing", "broken_plugin.thing"):
        with pytest.raises(PluginLoadError, match="quarantined") as e:
            registry.get("generator", name)
        assert e.value.__cause__ is boom
    with pytest.raises(PluginLoadError):
        TrackGenerator(TrackGenConfig(generator="broken.thing", device="cpu", num_envs=E),
                       _rng())


def test_quarantined_view_is_read_only(fresh):
    fresh(FakeEntryPoint("broken", RuntimeError("x"), FakeDist("b")))
    with pytest.warns(RuntimeWarning):
        view = registry.quarantined
    with pytest.raises(TypeError):
        view["other"] = "y"


def test_failing_register_rolls_back_its_partial_registrations(fresh):
    def register(reg):
        reg.add(_spec("half.done", module="half"))
        raise RuntimeError("crashed midway")

    fresh(FakeEntryPoint("half", register, FakeDist("half")))
    with pytest.warns(RuntimeWarning, match="crashed midway"):
        assert "half.done" not in registry.available("generator")
    with pytest.raises(PluginLoadError, match="crashed midway"):
        registry.get("generator", "half.done")


def test_strict_policy_raises_at_load(fresh):
    boom = ImportError("missing dependency")
    fresh(_toy_ep(), FakeEntryPoint("broken", boom, FakeDist("b")), policy="strict")
    with pytest.raises(PluginLoadError, match="broken.*missing dependency") as e:
        registry.available()
    assert e.value.__cause__ is boom
    # Rolled back: the built-ins are not left half-registered, and a retry raises again.
    assert registry._specs["generator"] == {}
    with pytest.raises(PluginLoadError):
        registry.get("generator", "bezier")


def test_off_policy_skips_discovery(fresh):
    calls = fresh(_toy_ep(), policy="off")
    names = registry.available("generator")
    assert BUILTIN_TRACK <= set(names) and "toy.circle" not in names
    assert calls == []                               # entry_points() was never queried
    with pytest.raises(ValueError, match="unknown generator 'toy.circle'"):
        registry.get("generator", "toy.circle")


@pytest.mark.parametrize("value", ["sometimes", "0", "false"])
def test_invalid_policy_warns_and_falls_back_to_warn(fresh, value):
    # A typo (or a natural "disable" spelling) must not break the built-ins.
    fresh(_toy_ep(), policy=value)
    with pytest.warns(RuntimeWarning, match=f"TRACK_GEN_PLUGINS='{value}'.*using 'warn'"):
        names = registry.available("generator")
    assert BUILTIN_TRACK <= set(names) and "toy.circle" in names
    assert TrackGenerator(TrackGenConfig(generator="bezier", device="cpu", num_envs=E),
                          _rng()).generate().valid.numpy().any()


def test_policy_is_case_insensitive(fresh):
    calls = fresh(_toy_ep(), policy=" OFF ")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert "toy.circle" not in registry.available("generator")
    assert calls == []


@pytest.mark.parametrize("where", ["callable", "module"])
def test_newer_api_version_is_quarantined(fresh, monkeypatch, where):
    called = []

    def register(reg):
        called.append(1)
        reg.add(TOY_CIRCLE)

    if where == "callable":
        register.track_gen_api = API_VERSION + 1
    else:
        module = types.ModuleType("future_plugin")
        module.API_VERSION = API_VERSION + 1
        monkeypatch.setitem(sys.modules, "future_plugin", module)
        register.__module__ = "future_plugin"
    fresh(FakeEntryPoint("future", register, FakeDist("future-plugin", "9.0")))
    with pytest.warns(RuntimeWarning, match="future.*plugin API"):
        assert "toy.circle" not in registry.available("generator")
    assert called == []                              # never called
    assert f"API {API_VERSION + 1}" in registry.quarantined["future"]


def test_api_version_declared_in_the_entry_point_module_is_checked(fresh, monkeypatch):
    # The usual layout: register lives in a submodule, is re-exported by the package the
    # entry point names, and the package declares track_gen_api.
    called = []

    def register(reg):
        called.append(1)
        reg.add(TOY_CIRCLE)

    register.__module__ = "future_pkg._impl"
    package = types.ModuleType("future_pkg")
    package.track_gen_api = API_VERSION + 98
    package.register = register
    monkeypatch.setitem(sys.modules, "future_pkg", package)
    monkeypatch.setitem(sys.modules, "future_pkg._impl", types.ModuleType("future_pkg._impl"))
    fresh(FakeEntryPoint("future", register, FakeDist("future-pkg", "9.0"),
                         value="future_pkg:register"))
    with pytest.warns(RuntimeWarning, match="future.*plugin API"):
        assert "toy.circle" not in registry.available("generator")
    assert called == []
    assert "module 'future_pkg'.track_gen_api" in registry.quarantined["future"]


@pytest.mark.parametrize("unrelated", ["v3", "2024-01", 1.5])
def test_unrelated_module_api_version_constant_is_ignored(fresh, monkeypatch, unrelated):
    # A plugin module's own API_VERSION (a REST API version, ...) is not a track_gen
    # plugin API declaration unless it is an int.
    module = types.ModuleType("rest_plugin")
    module.API_VERSION = unrelated
    monkeypatch.setitem(sys.modules, "rest_plugin", module)

    def register(reg):
        register_toys(reg)

    register.__module__ = "rest_plugin"
    fresh(FakeEntryPoint("toy", register, TOY_DIST))
    assert "toy.circle" in registry.available("generator")
    assert registry.quarantined == {}


def test_non_int_track_gen_api_is_quarantined(fresh):
    def register(reg):
        register_toys(reg)

    register.track_gen_api = "1"
    fresh(FakeEntryPoint("toy", register, TOY_DIST))
    with pytest.warns(RuntimeWarning, match="track_gen_api must be an int"):
        assert "toy.circle" not in registry.available("generator")


def test_current_api_version_is_accepted(fresh):
    def register(reg):
        register_toys(reg)

    register.track_gen_api = API_VERSION
    fresh(FakeEntryPoint("toy", register, TOY_DIST))
    assert "toy.circle" in registry.available("generator")
    assert registry.quarantined == {}


def test_non_dotted_third_party_name_warns_once(fresh):
    def register(reg):
        reg.add(_spec("circle", module="acme"))

    fresh(FakeEntryPoint("acme", register, FakeDist("acme")))
    with pytest.warns(UserWarning, match="'circle'.*not a dotted name"):
        assert "circle" in registry.available("generator")
    registry.unregister("generator", "circle")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        registry.add(_spec("circle", module="acme"))       # one-time: already warned
        registry.add(_spec("acme.dotted", module="acme"))  # dotted: never warns
    with pytest.warns(UserWarning, match="'square'.*not a dotted name"):
        registry.add(_spec("square", module="acme"))


def test_builtin_names_never_warn(fresh):
    fresh()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        registry.available()


# ---------------------------------------------------------------------------------------
# Kinds, unregister, scoped
# ---------------------------------------------------------------------------------------

def test_kinds_are_an_open_set(plugin_registry):
    relax = types.SimpleNamespace(name="acme.relax")
    with pytest.raises(TypeError, match="pass kind="):
        plugin_registry.add(relax)
    plugin_registry.add(relax, kind="relax")
    assert plugin_registry.available("relax") == ["acme.relax"]
    assert "relax:acme.relax" in plugin_registry.available()
    assert plugin_registry.get("relax", "acme.relax") is relax
    info = plugin_registry.describe("relax", "acme.relax")
    assert info.kind == "relax" and info.origin == __name__
    with pytest.raises(ValueError, match="registers as kind 'generator'"):
        plugin_registry.add(_spec("acme.x"), kind="gate_generator")
    with pytest.raises(ValueError, match="unknown elevation 'x'.*unknown kind"):
        plugin_registry.get("elevation", "x")
    with pytest.raises(ValueError, match="non-empty string"):
        plugin_registry.add(_spec(""))


def test_unregister(plugin_registry):
    plugin_registry.add(TOY_CIRCLE)
    plugin_registry.unregister("generator", "toy.circle")
    assert "toy.circle" not in plugin_registry.available("generator")
    with pytest.raises(ValueError, match="unknown generator 'toy.circle'"):
        plugin_registry.unregister("generator", "toy.circle")


def test_scoped_restores_every_kind():
    before = registry.available()
    kinds = set(registry._specs)
    bezier = registry.get("generator", "bezier")
    view = generator_registry.GENERATORS
    with registry.scoped() as reg:
        assert reg is registry
        reg.add(TOY_CIRCLE)
        reg.add(TOY_RING)
        reg.add(types.SimpleNamespace(name="acme.relax"), kind="relax")
        reg.unregister("generator", "bezier")
        assert "toy.circle" in view and "bezier" not in view     # the view is live
    assert registry.available() == before
    assert set(registry._specs) == kinds and set(registry._meta) == kinds
    assert registry.get("generator", "bezier") is bezier
    assert "toy.circle" not in view and view["bezier"] is bezier
    assert registry.available("relax") == []
    with pytest.raises(ValueError, match="unknown kind 'relax'"):
        registry.get("relax", "acme.relax")


def test_scoped_fresh_restores_the_loaded_state(fresh):
    fresh(_toy_ep())
    assert "toy.circle" in registry.available("generator")
    with registry.scoped(fresh=True):
        assert registry._specs["generator"] == {} and not registry._loaded
    assert "toy.circle" in registry.available("generator")


# ---------------------------------------------------------------------------------------
# Loading: lazy, idempotent, thread-safe
# ---------------------------------------------------------------------------------------

def test_loading_is_lazy_and_idempotent(monkeypatch):
    monkeypatch.delenv(POLICY_ENV, raising=False)
    calls = install_entry_points(monkeypatch, _toy_ep())
    builtins = []
    real = builtin_plugins.register
    monkeypatch.setattr(builtin_plugins, "register",
                        lambda reg: builtins.append(1) or real(reg))
    reg = Registry()
    assert builtins == [] and calls == []            # construction loads nothing
    reg.get("generator", "toy.circle")
    reg.available()
    reg.describe("gate_generator", "toy.ring")
    reg.add(_spec("acme.late"))
    assert builtins == [1] and calls == [ENTRY_POINT_GROUP]


def test_two_threads_load_builtins_once(monkeypatch):
    install_entry_points(monkeypatch)
    builtins = []
    real = builtin_plugins.register

    def slow_register(reg):
        builtins.append(threading.get_ident())
        time.sleep(0.05)                             # widen the race window
        real(reg)

    monkeypatch.setattr(builtin_plugins, "register", slow_register)
    reg = Registry()
    barrier = threading.Barrier(2)
    got, errors = [], []

    def worker():
        try:
            barrier.wait()
            got.append(reg.get("generator", "bezier"))
        except BaseException as exc:  # noqa: BLE001 - reported below
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert len(builtins) == 1
    assert len(got) == 2 and got[0] is got[1]


def test_plugin_register_may_query_the_registry(fresh):
    seen = []

    def register(reg):
        seen.append(reg.get("generator", "bezier").name)   # re-entrant lookup
        reg.add(TOY_CIRCLE)

    fresh(FakeEntryPoint("toy", register, TOY_DIST))
    assert "toy.circle" in registry.available("generator")
    assert seen == ["bezier"]


# ---------------------------------------------------------------------------------------
# Deprecated _src shims
# ---------------------------------------------------------------------------------------

def test_generator_registry_shim(plugin_registry):
    assert generator_registry.available() == plugin_registry.available("generator")
    assert generator_registry.get("bezier") is plugin_registry.get("generator", "bezier")
    with warnings.catch_warnings():
        warnings.simplefilter("error")               # a new name: no warning at all
        generator_registry.register(_spec("legacy_new"))
    assert "legacy_new" in generator_registry.available()
    replacement = _spec("legacy_new", module="other")
    with pytest.warns(DeprecationWarning, match="plugin.registry.add"):
        generator_registry.register(replacement)    # old semantics: overwrite
    assert generator_registry.get("legacy_new") is replacement
    view = generator_registry.GENERATORS
    assert isinstance(view, types.MappingProxyType)
    assert view["legacy_new"] is replacement and "bezier" in view
    with pytest.raises(TypeError):
        view["x"] = replacement
    from track_gen._src.generator_registry import GENERATORS  # noqa: F401 - importable
    generator_registry._ensure_loaded()


def test_gate_generator_registry_shim(plugin_registry):
    assert gate_generator_registry.available() == plugin_registry.available("gate_generator")
    assert gate_generator_registry.get("hull") is plugin_registry.get("gate_generator", "hull")
    gate_generator_registry.register(TOY_RING)
    assert "toy.ring" in gate_generator_registry.GATE_GENERATORS
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        gate_generator_registry.register(TOY_RING)     # the identical spec: a no-op
    same_module = dataclasses.replace(TOY_RING, max_gates=lambda config: TOY_GATES)
    assert same_module.generate.__module__ == TOY_RING.generate.__module__
    with pytest.warns(DeprecationWarning):
        gate_generator_registry.register(same_module)  # same module: overwrites
    assert gate_generator_registry.get("toy.ring") is same_module
    # The old gate rule: another module's spec under a taken name raises (ValueError).
    hull = gate_generator_registry.get("hull")
    other = dataclasses.replace(hull, generate=_spec("x", module="other_gates").generate)
    with pytest.raises(DuplicateSpecError, match="'hull'.*already registered") as e:
        gate_generator_registry.register(other)
    assert isinstance(e.value, ValueError)
    assert gate_generator_registry.get("hull") is hull
    with pytest.raises(TypeError):
        gate_generator_registry.GATE_GENERATORS["x"] = TOY_RING
    with pytest.raises(ValueError, match="unknown gate generator 'nope'"):
        gate_generator_registry.get("nope")


def test_spec_types_are_reexported():
    import track_gen.plugin as plugin

    assert plugin.GeneratorSpec is generator_registry.GeneratorSpec
    assert plugin.GateGeneratorSpec is gate_generator_registry.GateGeneratorSpec
    assert GeneratorSpec.registry_kind == "generator"
    assert GateGeneratorSpec.registry_kind == "gate_generator"
    assert "registry_kind" not in {f.name for f in dataclasses.fields(GeneratorSpec)}
    assert set(plugin.__all__) >= {"Registry", "registry", "GeneratorSpec",
                                   "GateGeneratorSpec", "DuplicateSpecError",
                                   "PluginOverrideWarning", "PluginLoadError",
                                   "API_VERSION", "SpecInfo"}
    # Roadmap PR8: per-generator Params and the generator subset of Stage.
    assert set(plugin.__all__) >= {"Params", "NoParams", "static", "dynamic", "CaptureClass",
                                   "Layout", "Stage", "StageBuilder", "ParamTable",
                                   "StageParams"}
    for name in plugin.__all__:
        assert getattr(plugin, name) is not None, name
    assert [f.name for f in dataclasses.fields(GeneratorSpec)][-1] == "params"
    assert [f.name for f in dataclasses.fields(GateGeneratorSpec)][-1] == "params"
    assert issubclass(DuplicateSpecError, ValueError)
    assert issubclass(PluginOverrideWarning, UserWarning)
    assert issubclass(PluginLoadError, RuntimeError)


def test_documented_contract_constants():
    # Pinned literals: renaming the env var or the group, or bumping the API, is a
    # breaking change for every installed plugin and must fail here.
    assert POLICY_ENV == "TRACK_GEN_PLUGINS"
    assert ENTRY_POINT_GROUP == "track_gen.plugins"
    assert API_VERSION == 1


# ---------------------------------------------------------------------------------------
# Robustness: bad metadata, shared entry-point names, attribution, odd inputs
# ---------------------------------------------------------------------------------------

class _UnreadableDist:
    """A distribution whose metadata cannot be read (invalid UTF-8, missing Name)."""

    @property
    def name(self):
        raise KeyError("Name")

    @property
    def metadata(self):
        raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")

    @property
    def version(self):
        raise KeyError("Version")


def test_unreadable_distribution_metadata_does_not_break_lookups(fresh):
    # Neither the sort key nor the load may raise out of discovery: the built-ins (and
    # the bad distribution's own plugin, with an unknown distribution) still load.
    fresh(_toy_ep(), FakeEntryPoint("bad", lambda reg: reg.add(_spec("bad.ok")),
                                    _UnreadableDist()))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        track = TrackGenerator(TrackGenConfig(generator="bezier", device="cpu",
                                              num_envs=E), _rng())
    assert track is not None
    assert {"toy.circle", "bad.ok"} <= set(registry.available("generator"))
    assert registry.describe("generator", "bad.ok").distribution is None
    assert registry.quarantined == {}


def test_broken_plugins_sharing_an_entry_point_name_are_both_quarantined(fresh):
    alpha_err, beta_err = ImportError("alpha broken"), ImportError("beta broken")
    fresh(FakeEntryPoint("plugin", alpha_err, FakeDist("alpha-tracks", "1.0"),
                         value="alpha_tracks:register"),
          FakeEntryPoint("plugin", beta_err, FakeDist("beta-gates", "2.0"),
                         value="beta_gates:register"))
    with pytest.warns(RuntimeWarning):
        quarantined = dict(registry.quarantined)
    assert quarantined == {"alpha-tracks:plugin": "ImportError: alpha broken",
                           "beta-gates:plugin": "ImportError: beta broken"}
    with pytest.raises(PluginLoadError, match="alpha-tracks") as e:
        registry.get("generator", "alpha_tracks.x")
    assert e.value.__cause__ is alpha_err
    with pytest.raises(PluginLoadError, match="beta-gates") as e:
        registry.get("gate_generator", "beta_gates.y")
    assert e.value.__cause__ is beta_err
    with pytest.raises(ValueError) as e:
        registry.get("generator", "nope")
    assert "alpha broken" in str(e.value) and "beta broken" in str(e.value)


def test_quarantine_attribution_uses_the_full_entry_point_module(fresh):
    # A namespace package: a broken nvidia.tracks plugin must not claim every nvidia.*
    # name, in any kind.
    boom = ImportError("nvidia.tracks is broken")
    fresh(FakeEntryPoint("nv_tracks", boom, FakeDist("nvidia-tracks", "1.0"),
                         value="nvidia.tracks:register"))
    with pytest.warns(RuntimeWarning):
        registry.available()
    for kind, name in (("generator", "nvidia.gates.ring"),
                       ("gate_generator", "nvidia.anything")):
        with pytest.raises(ValueError) as e:
            registry.get(kind, name)
        assert not isinstance(e.value, PluginLoadError)
        assert "nvidia.tracks is broken" in str(e.value)   # still listed
    for name in ("nvidia.tracks.ring", "nv_tracks.x", "nvidia_tracks.x"):
        with pytest.raises(PluginLoadError, match="may be provided by.*available") as e:
            registry.get("generator", name)
        assert e.value.__cause__ is boom


def test_exact_quarantined_name_is_attributed_with_certainty(fresh):
    def register(reg):
        reg.add(_spec("odd_prefix.thing", module="half"))
        raise RuntimeError("crashed")

    fresh(FakeEntryPoint("half", register, FakeDist("half")))
    with pytest.warns(RuntimeWarning):
        registry.available()
    with pytest.raises(PluginLoadError, match="is provided by.*'half'"):
        registry.get("generator", "odd_prefix.thing")


@pytest.mark.parametrize("name", [None, 3])
def test_non_string_name_raises_value_error(plugin_registry, name):
    with pytest.raises(ValueError, match="unknown generator"):
        plugin_registry.get("generator", name)
    with pytest.raises(ValueError, match="unknown generator"):
        plugin_registry.describe("generator", name)
    with pytest.raises(ValueError, match=f"unknown generator {name!r}"):
        TrackGenerator(TrackGenConfig(generator=name, device="cpu", num_envs=E), _rng())


def test_plugin_cannot_unregister_a_builtin_to_bypass_replace(fresh):
    def register(reg):
        reg.unregister("generator", "hull")
        reg.add(GeneratorSpec(name="hull", alloc_scratch=TOY_CIRCLE.alloc_scratch,
                              generate=TOY_CIRCLE.generate))

    fresh(FakeEntryPoint("sneaky", register, FakeDist("sneaky")))
    with pytest.warns(RuntimeWarning, match="'sneaky'.*may only unregister"):
        info = registry.describe("generator", "hull")
    assert info.builtin                                # the built-in stays
    assert "PluginLoadError" in registry.quarantined["sneaky"]


def test_plugin_may_unregister_its_own_names(fresh):
    def register(reg):
        reg.add(_spec("own.tmp", module="own"))
        reg.unregister("generator", "own.tmp")
        reg.add(TOY_CIRCLE)

    fresh(FakeEntryPoint("own", register, FakeDist("own")))
    names = registry.available("generator")
    assert "toy.circle" in names and "own.tmp" not in names
    assert registry.quarantined == {}


def test_swallowed_unregister_refusal_still_quarantines(fresh):
    def register(reg):
        reg.add(TOY_CIRCLE)
        try:
            reg.unregister("generator", "bezier")
        except PluginLoadError:
            pass

    fresh(FakeEntryPoint("toy", register, TOY_DIST))
    with pytest.warns(RuntimeWarning, match="may only unregister"):
        assert "toy.circle" not in registry.available("generator")
    assert registry.describe("generator", "bezier").builtin


def test_adding_the_identical_spec_again_is_a_noop(fresh):
    # E.g. a user calls acme.register(registry) by hand while the package is also
    # installed with an entry point, or re-runs a notebook cell.
    fresh(_toy_ep())
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        register_toys(registry)
        registry.add(TOY_CIRCLE)
        registry.add(TOY_CIRCLE, replace=True)
        generator_registry.register(TOY_CIRCLE)
    assert registry.get("generator", "toy.circle") is TOY_CIRCLE
    assert registry.describe("generator", "toy.circle").entry_point == "toy"
    with pytest.raises(DuplicateSpecError):
        registry.add(dataclasses.replace(TOY_CIRCLE))  # an equal but distinct object


def test_patching_importlib_metadata_entry_points_takes_effect(monkeypatch):
    # The usual way plugin authors fake discovery, after track_gen.plugin is imported.
    import importlib.metadata

    monkeypatch.delenv(POLICY_ENV, raising=False)
    ep = _toy_ep()
    monkeypatch.setattr(importlib.metadata, "entry_points",
                        lambda **params: [ep] if params.get("group") == ep.group else [])
    with registry.scoped(fresh=True):
        assert "toy.circle" in registry.available("generator")
    assert ep.loads == 1


def test_real_distribution_is_discovered_through_importlib_metadata(tmp_path,
                                                                   monkeypatch):
    # No fake: a module plus a <name>-<version>.dist-info with entry_points.txt on
    # sys.path, found by the real importlib.metadata (EntryPoint.load, ep.dist).
    module = "realdist_toy_plugin"
    (tmp_path / f"{module}.py").write_text(
        "import dataclasses\n"
        "from tests._plugin_toys import TOY_CIRCLE\n"
        "track_gen_api = 1\n"
        "SPEC = dataclasses.replace(TOY_CIRCLE, name='realdist.circle')\n"
        "def register(reg):\n"
        "    reg.add(SPEC)\n")
    dist_info = tmp_path / "realdist_toy_plugin-1.2.3.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: realdist-toy-plugin\nVersion: 1.2.3\n")
    (dist_info / "entry_points.txt").write_text(
        f"[{ENTRY_POINT_GROUP}]\nrealdist = {module}:register\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delenv(POLICY_ENV, raising=False)
    monkeypatch.delitem(sys.modules, module, raising=False)
    try:
        with registry.scoped(fresh=True):
            track = TrackGenerator(TrackGenConfig(generator="realdist.circle",
                                                  device="cpu", num_envs=E),
                                   _rng(3)).generate()
            assert track.valid.numpy().tolist() == [1] * E
            info = registry.describe("generator", "realdist.circle")
            assert info.distribution == ("realdist-toy-plugin", "1.2.3")
            assert info.entry_point == "realdist" and not info.builtin
    finally:
        sys.modules.pop(module, None)
