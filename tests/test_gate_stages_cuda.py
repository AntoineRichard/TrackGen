"""CUDA-only: the gate pipeline as stages under caller capture (roadmap PR11).

``GateGenerator`` runs a gate ``Program`` (``track_gen.pipeline.presets.gates`` or
``legacy.to_gate_pipeline_spec(GateGenConfig)``): an anchors generator (or a centerline
generator + ``stations``), ``order.*``, ``normalize``, ``carry.*``, ``gate_relax``,
``gate_arclen``, ``elevation.*`` (rewired to ``gates.*``), ``lift``, ``tangents``,
``gate_frames``, ``validity.*``, then the hold/restore commit. This module records those
Programs into a CALLER's CUDA graph:

- every built-in gate generator with every ordering it supports (flat and 3D);
- the toys of ``tests/_gate_toys.py`` (public plugin surface, installed through a fake
  entry point): ``acme.superformula`` (a third-party TRACK generator) through
  ``stations`` / ``relaxed_stations``, ``acme.helix`` / ``acme.helix_posed`` (anchors
  generators that provide z / tangent [/ half_size / roll]), ``acme.scatter`` +
  ``acme.reverse`` (a third-party order), ``acme.max_height`` (a gate validity check with
  its own fail bit), and ``acme.sinusoid_hills`` from ``tests/_stage_toys.py`` (a track
  elevation plugin reused on gates through the planner's port rewiring);
- the legacy ``GateGeneratorSpec`` adapter (``toy.ring``);
- masked regeneration (the gate commit) and ``on_invalid="poison"``.

Method (as in tests/test_caller_capture_cuda.py and tests/test_stages_cuda.py; the
helpers are :mod:`track_gen.testing.capture`'s): record on a NON-current side stream,
poison the outputs with FINITE, dtype-aware sentinels (gate outputs are NaN-padded,
poisoned envs are NaN, and ``assert_array_equal`` treats NaN == NaN, so a NaN poison
would hide a missed write), replay, and compare with an eager twin: a fresh
``GateGenerator`` of the same config / spec run with ``enqueue()`` (never a graph). Every
recording runs with the Warp mempool disabled (an allocation inside the capture fails
loudly) and ``wp.capture_launch`` replaced by a tripwire (a graph launch cannot be
recorded into another capture). Construction prepares the default body; the masked body
is warmed up with an all-zero mask (an exact no-op) and the test's own kernels on scratch
buffers, so no module loads while capturing. Seeds are rewritten between replays
(outside the graph): the graph reads the live seeds, so a replay after a reseed must equal
a twin built with the new seeds.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

pytestmark = pytest.mark.cuda

import warp as wp  # noqa: E402

from tests._gate_toys import acme_gates_entry_point  # noqa: E402
from tests._plugin_toys import TOY_RING, install_entry_points  # noqa: E402
from tests._stage_toys import acme_entry_point  # noqa: E402
from track_gen import GateGenConfig, GateGenerator, PerEnvSeededRNG  # noqa: E402
from track_gen._src.warp_gate import _safe_normalize3  # noqa: E402
from track_gen.pipeline import GATE_FAIL_BITS, presets  # noqa: E402
from track_gen.plugin import POLICY_ENV, registry  # noqa: E402
from track_gen.testing import capture as tg_capture  # noqa: E402
from track_gen.testing import check_gate_generator, check_stage  # noqa: E402

DEV = "cuda:0"
E = 8
G_ST = 12                       # stations per env (presets.gates(gates=...))

HELIX = "acme.helix"
HELIX_POSED = "acme.helix_posed"
SCATTER = "acme.scatter"
REVERSE = "acme.reverse"
SUPERFORMULA = "acme.superformula"
MAX_HEIGHT = "acme.max_height"
HILLS = "acme.sinusoid_hills"
_TOYS = (("gate_generator", HELIX), ("gate_generator", HELIX_POSED),
         ("gate_generator", SCATTER), ("order", REVERSE), ("generator", SUPERFORMULA),
         ("validity", MAX_HEIGHT), ("elevation", HILLS))

# Every built-in gate generator with each ordering it supports while only the built-in
# orders are registered (HEAD's lists; test_builtin_ordering_matrix_is_complete pins it).
BUILTIN_ORDERINGS = (
    ("bezier", "ccw"), ("bezier", "random_pairs"),
    ("hull", "ccw"), ("hull", "random_pairs"),
    ("polar", "ccw"), ("polar", "raw"),
    ("checkpoint", "ccw"), ("checkpoint", "raw"),
    ("voronoi", "ccw"), ("voronoi", "raw"),
)
_NORMALIZING = ("polar", "checkpoint", "voronoi")   # preferred["normalize"] = "extent"
# The default 2D path (sphere relaxation, flat elevation), and a 3D one (relaxation
# skipped, gate_arclen + elevation.noise, full-tangent frames): between them every
# built-in gate stage is recorded.
_VARIANTS = {
    "flat": {"gate_width": 0.05},
    "3d": {"gate_width": 0.05, "z_profile": "noise", "z_base": 0.1, "z_max": 1.5,
           "z_noise_amplitude": 0.5, "z_noise_harmonics": 4,
           "gate_align": "full_tangent", "gate_solve_iters": 0},
}

GATE_POINT_FIELDS = ("position", "tangent", "forward", "left", "right", "orientation",
                     "half_size")
GATE_ENV_FIELDS = ("count", "valid", "generation", "fail_mask")

_ZERO = np.zeros(E, np.int32)
_ONES = np.ones(E, np.int32)
_M1 = np.array([1, 0, 1, 0, 0, 1, 0, 1], np.int32)
# Gates lifted into [0.5, 1.5] (elevation.uniform on gates), so acme.max_height's bit is
# set for z_cap = 0 and clear for z_cap = 10 on every non-empty env.
_UPLIFT = ("uniform", {"z_min": 0.5, "z_max": 1.5})
_ZCAP_HIGH = 10.0
_ZCAP_SPLIT = np.array([0.0, _ZCAP_HIGH] * (E // 2), np.float32)


@pytest.fixture
def acme(monkeypatch):
    """The ``acme-gates`` and ``acme-stages`` toy distributions, discovered through fake
    entry points in a fresh registry scope (restored afterwards, as is
    ``TRACK_GEN_PLUGINS``)."""
    monkeypatch.delenv(POLICY_ENV, raising=False)
    install_entry_points(monkeypatch, acme_gates_entry_point(), acme_entry_point())
    with registry.scoped(fresh=True):
        for kind, name in _TOYS:
            assert not registry.describe(kind, name).builtin, (kind, name)
        yield registry


# ---------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------

def _snapshot(obj) -> dict:
    return {k: np.array(a.numpy(), copy=True)
            for k, a in tg_capture.result_arrays(obj).items()}


def _rows(host: dict, envs) -> dict:
    return {k: v.reshape(E, -1)[envs] for k, v in host.items()}


def _assert_host(got: dict, ref: dict, label: str) -> None:
    assert ref and got.keys() == ref.keys(), f"{label}: {sorted(set(got) ^ set(ref))}"
    for name in ref:
        np.testing.assert_array_equal(got[name], ref[name], err_msg=f"{label}: {name}")


def _differs(a: dict, b: dict) -> bool:
    return any(not np.array_equal(a[k], b[k], equal_nan=True) for k in a)


def _stitch(per_env: list) -> dict:
    """The batch whose env ``e`` is env ``e`` of ``per_env[e]`` (host snapshots)."""
    return {name: np.stack([s[name].reshape(E, -1)[e] for e, s in enumerate(per_env)])
            .reshape(per_env[0][name].shape) for name in per_env[0]}


def _side_stream() -> wp.Stream:
    return tg_capture.side_stream(DEV)


def _recording(stream):
    """The caller's Warp capture on ``stream``: mempool disabled, capture_launch
    forbidden."""
    return tg_capture.recording(DEV, stream)


def _replay(graph, stream) -> None:
    wp.capture_launch(graph, stream=stream)
    wp.synchronize()


def _dev(values, dtype=wp.float32) -> wp.array:
    """A device copy of ``values``, complete before any other stream reads it."""
    np_dtype = np.float32 if dtype is wp.float32 else np.int32
    arr = wp.array(np.asarray(values, np_dtype), dtype=dtype, device=DEV)
    wp.synchronize()
    return arr


def _write(dst: wp.array, values) -> None:
    """Write host values into a device buffer, outside any capture (read at replay)."""
    dst.assign(np.asarray(values, dst.numpy().dtype))
    wp.synchronize()


def _per_env(values) -> np.ndarray:
    v = np.asarray(values, np.float32)
    return np.resize(v, E).astype(np.float32) if v.ndim else np.full(E, v, np.float32)


def _reseed(rng, base: int) -> None:
    """Reseed outside any capture (same expansion as ``PerEnvSeededRNG(seeds=base)``)."""
    rng.set_seeds_warp(wp.array((base + np.arange(E)).astype(np.int32), dtype=wp.int32,
                                device=DEV), None)
    wp.synchronize()


def _make(source, seeds=11):
    """``GateGenerator(source, rng)`` for a ``GateGenConfig`` or a gate ``PipelineSpec``;
    returns ``(gen, rng)``."""
    rng = PerEnvSeededRNG(seeds=seeds, num_envs=E, device=DEV)
    return GateGenerator(source, rng), rng


def _eager(source, seeds=11) -> dict:
    """A fresh twin of ``source`` run eagerly (``enqueue`` never self-graphs)."""
    twin, _ = _make(source, seeds)
    out = twin.enqueue()
    wp.synchronize()
    assert twin.graph is None, "enqueue() must never self-graph"
    return _snapshot(out)


def _gates(**kw):
    return presets.gates(num_envs=E, device=DEV, **kw)


def _ref(spec, stage_id: str):
    refs = [r for r in spec.stages if r.slot_id == stage_id]
    assert len(refs) == 1, f"{stage_id!r} in {spec.ids}"
    return refs[0]


def _stage_id(spec, kind: str, name: str) -> str:
    ids = [r.slot_id for r in spec.stages if r.kind == kind and r.name == name]
    assert len(ids) == 1, f"{kind}.{name} in {spec.ids}"
    return ids[0]


def _set(gen, stage_id: str, field: str, values, *, stream=None) -> None:
    """Write a dynamic field's live array: a scalar ``set`` for uniform values, an
    ``[E]`` array ``set`` otherwise."""
    v = _per_env(values)
    sp = gen.params[stage_id]
    src = float(v[0]) if np.all(v == v[0]) else _dev(v)
    sp.set(field, src, stream=stream)
    wp.synchronize()                       # src stays alive until the copy completed


def _assert_valid_is_fail_mask_zero(host: dict, label: str) -> None:
    assert "fail_mask" in host, f"{label}: GateSequence.fail_mask is None"
    np.testing.assert_array_equal(host["valid"], (host["fail_mask"] == 0).astype(np.int32),
                                  err_msg=f"{label}: valid != (fail_mask == 0)")


def _poisoned(keep: dict) -> dict:
    """The ``on_invalid="poison"`` GateSequence of a ``"keep"`` snapshot, derived
    independently of the commit kernel: every env with ``fail_mask != 0`` gets
    ``count = 0``, NaN on every per-gate field (quaternion and ``half_size`` included)
    and ``valid = 0``; ``fail_mask`` and ``generation`` keep their values; valid envs
    are unchanged."""
    assert set(keep) == set(GATE_POINT_FIELDS + GATE_ENV_FIELDS), sorted(keep)
    out = {k: v.copy() for k, v in keep.items()}
    bad = keep["fail_mask"] != 0
    for name in GATE_POINT_FIELDS:
        out[name].reshape(E, -1)[bad] = np.nan
    out["count"][bad] = 0
    out["valid"][bad] = 0
    return out


def _count_split(**kw) -> int:
    """A ``min_gates`` strictly inside the spread of the per-env gate counts of
    ``presets.gates(**kw)`` (seeds 11..): the built-in count check (bit ``count``) then
    fails some envs and passes others."""
    counts = _eager(_gates(**kw))["count"]
    uniq = np.unique(counts)
    assert uniq.size >= 2, f"every env has the same gate count: {counts.tolist()}"
    return int(uniq[uniq.size // 2])


def _record_replay_reseed(source, label: str, *, seeds=(11, 101), check=None):
    """Build ``GateGenerator(source)``, record ONE ``generate(stream=s)`` into the
    caller's capture (no allocation, no capture_launch, no self-graph; the recording
    executes nothing), then replay it into poisoned outputs three times: at the build
    seeds, after a reseed to ``seeds[1]``, and back. Each replay equals the eager twin
    of those seeds and has ``valid == (fail_mask == 0)``; ``check(gen, host, label)``
    runs after each. Returns ``(gen, out, cap, stream, rng)``."""
    gen, rng = _make(source, seeds[0])
    refs = {s: _eager(source, s) for s in seeds}
    assert _differs(refs[seeds[0]], refs[seeds[1]]), \
        f"{label}: reseeding must change the batch (vacuous test otherwise)"
    base = _snapshot(gen.program.result)
    _assert_host(base, refs[seeds[0]], f"{label}: construction batch vs eager twin")

    s = _side_stream()
    with _recording(s) as cap:
        out = gen.generate(stream=s)
    assert out is gen.program.result, label
    assert gen.graph is None, f"{label}: recording into the caller's graph must not self-capture"
    _assert_host(_snapshot(out), base, f"{label}: the recording executed the pipeline")

    for k, seed in enumerate((seeds[0], seeds[1], seeds[0])):
        if k:
            _reseed(rng, seed)
        step = f"{label}: replay {k} (seeds {seed}..)"
        tg_capture.poison_all(out)
        _replay(cap.graph, s)
        got = _snapshot(out)
        _assert_host(got, refs[seed], step)
        _assert_valid_is_fail_mask_zero(got, step)
        if check is not None:
            check(gen, got, step)
    assert gen.graph is None
    return gen, out, cap, s, rng


def _assert_kit_capturable(gen, label: str) -> None:
    """The conformance kit's recording (Warp capture, plus the torch modes when torch
    with CUDA is available) replays to the eager twin before and after a reseed."""
    status = tg_capture.assert_caller_capturable(gen)
    assert status["warp"] == "ok", f"{label}: {status}"
    assert gen.graph is None, f"{label}: caller recordings never self-capture"


# ---------------------------------------------------------------------------------------
# Built-in gate generators x orderings
# ---------------------------------------------------------------------------------------

def test_builtin_ordering_matrix_is_complete():
    """:data:`BUILTIN_ORDERINGS` (the static parametrization below) lists every built-in
    gate generator with exactly the orderings it supports while only the built-in
    orders are registered."""
    from track_gen.stages.gates import _supported_orderings  # noqa: PLC0415

    builtins = sorted(n for n in registry.available("gate_generator")
                      if registry.describe("gate_generator", n).builtin)
    assert builtins == sorted({g for g, _ in BUILTIN_ORDERINGS})
    for name in builtins:
        assert sorted(_supported_orderings(registry, name)) == \
            sorted(o for g, o in BUILTIN_ORDERINGS if g == name), name


@pytest.mark.parametrize("variant", list(_VARIANTS))
@pytest.mark.parametrize("generator,ordering", BUILTIN_ORDERINGS,
                         ids=[f"{g}-{o}" for g, o in BUILTIN_ORDERINGS])
def test_builtin_gate_generator_and_ordering_record_into_caller_capture(
        generator, ordering, variant):
    """Each built-in gate generator with each supported ordering, through the config
    path (``GateGenerator(GateGenConfig(...))`` -> ``legacy.to_gate_pipeline_spec``):
    the Program holds the anchors stage, ``order.<ordering>``, ``normalize.extent``
    exactly for the normalizing generators, and ``gate_relax.spheres`` / ``.none``;
    ``generate(stream=s)`` records into the caller's capture and every replay into
    poisoned outputs equals the eager twin, before and after a reseed. On the flat
    variant the conformance kit's recording (torch modes included) agrees."""
    cfg = GateGenConfig(device=DEV, num_envs=E, generator=generator,
                        gate_ordering=ordering, **_VARIANTS[variant])
    label = f"{generator}/{ordering}/{variant}"
    gen, *_ = _record_replay_reseed(cfg, label)
    spec = gen.program.spec
    assert (_ref(spec, "generator").kind, _ref(spec, "generator").name) == \
        ("gate_generator", generator), spec.ids
    assert _ref(spec, "order").name == ordering, spec.ids
    assert ("normalize" in spec.ids) == (generator in _NORMALIZING), spec.ids
    assert _ref(spec, "gate_relax").name == ("spheres" if variant == "flat" else "none")
    assert _ref(spec, "elevation").name == ("flat" if variant == "flat" else "noise")
    if variant == "flat":
        _assert_kit_capturable(gen, label)


def test_legacy_gate_generator_adapter_records_into_caller_capture():
    """A legacy ``GateGeneratorSpec`` (``toy.ring``: ``accepts_ctx``, capturable,
    orders its own gates) runs through the adapter slot (no ``order`` / ``normalize``
    stage) and records into the caller's capture like a native generator."""
    registry.add(TOY_RING)
    cfg = GateGenConfig(device=DEV, num_envs=E, generator="toy.ring", gate_ordering="raw",
                        gate_width=0.05)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)   # bind-time deprecation
        gen, *_ = _record_replay_reseed(cfg, "toy.ring (legacy adapter)")
    ids = gen.program.spec.ids
    assert "order" not in ids and "normalize" not in ids, ids


# ---------------------------------------------------------------------------------------
# Toy gate pipelines
# ---------------------------------------------------------------------------------------

def _height_spec(on_invalid: str = "keep", z_cap=None):
    """Bezier gates lifted into [0.5, 1.5] with ``acme.max_height`` after
    ``gate_default``."""
    check = MAX_HEIGHT if z_cap is None else (MAX_HEIGHT, {"z_cap": float(z_cap)})
    return _gates(elevation=_UPLIFT, checks=("gate_default", check), on_invalid=on_invalid)


def _superformula(source: str = "auto", **kw):
    return _gates(generator=SUPERFORMULA, gates=G_ST, source=source, **kw)


# label -> (source builder, the (kind, name) the recorded Program must hold)
_TOY_PIPELINES = {
    "helix": (lambda: _gates(generator=HELIX), ("gate_generator", HELIX)),
    "helix_posed": (lambda: _gates(generator=HELIX_POSED), ("gate_generator", HELIX_POSED)),
    "scatter-reverse": (lambda: GateGenConfig(device=DEV, num_envs=E, generator=SCATTER,
                                              gate_ordering=REVERSE, gate_width=0.05),
                        ("order", REVERSE)),
    "bezier-reverse": (lambda: GateGenConfig(device=DEV, num_envs=E, generator="bezier",
                                             gate_ordering=REVERSE, gate_width=0.05),
                       ("order", REVERSE)),
    "superformula-stations": (lambda: _superformula(), ("stations", "raw")),
    "superformula-relaxed": (lambda: _superformula("relaxed_stations"),
                             ("stations", "relaxed")),
    "hills-on-gates": (lambda: _gates(elevation=HILLS), ("elevation", HILLS)),
    "max_height": (lambda: _height_spec(), ("validity", MAX_HEIGHT)),
}


@pytest.mark.parametrize("case", list(_TOY_PIPELINES))
def test_toy_gate_pipeline_records_into_caller_capture(acme, case):
    """Each toy pipeline (a provider, a third-party order on a toy and on a built-in
    generator through the CONFIG path, a third-party track generator through
    ``stations`` / ``relaxed_stations``, a track elevation plugin rewired onto gates, a
    gate validity plugin): the toy is in the recorded Program, every replay of the
    caller's graph into poisoned outputs equals the eager twin before and after a
    reseed, and the conformance kit's recording agrees (torch modes included)."""
    build, (kind, name) = _TOY_PIPELINES[case]
    gen, *_ = _record_replay_reseed(build(), case)
    _stage_id(gen.program.spec, kind, name)
    _assert_kit_capturable(gen, case)


@wp.kernel
def _normalize3_k(src: wp.array(dtype=wp.vec3f), dst: wp.array(dtype=wp.vec3f)):
    t = wp.tid()
    dst[t] = _safe_normalize3(src[t])


_PROVIDES = {HELIX: ("z", "tangent"), HELIX_POSED: ("z", "tangent", "half_size", "roll")}


def _check_provided(attrs):
    """After a replay: each provided attribute, carried by anchor index (order.raw),
    reaches the GateSequence of THAT replay: ``position.z == anchors.z`` and (posed)
    ``half_size == anchors.half_size`` bit for bit, ``tangent ==
    _safe_normalize3(anchors.tangent)`` bit for bit (a Warp reference on the same
    device), on every real gate."""
    def check(gen, host, label):
        tap = gen.program.tap
        A = int(tap("anchors.xy").shape[0]) // E
        G = int(host["half_size"].size) // E
        count = host["count"]
        assert np.all((count >= 1) & (count <= min(A, G))), f"{label}: {count.tolist()}"
        z = tap("anchors.z").numpy().reshape(E, A)
        pos = host["position"].reshape(E, G, 3)
        at = tap("anchors.tangent")
        want_t = wp.empty_like(at)
        wp.launch(_normalize3_k, int(at.shape[0]), inputs=[at, want_t], device=DEV)
        wp.synchronize()
        want_t = want_t.numpy().reshape(E, A, 3)
        tan = host["tangent"].reshape(E, G, 3)
        hs_a = tap("anchors.half_size").numpy().reshape(E, A) if "half_size" in attrs \
            else None
        for e in range(E):
            m = int(count[e])
            np.testing.assert_array_equal(pos[e, :m, 2], z[e, :m],
                                          err_msg=f"{label}: env {e}: position.z")
            np.testing.assert_array_equal(tan[e, :m], want_t[e, :m],
                                          err_msg=f"{label}: env {e}: tangent")
            if hs_a is not None:
                np.testing.assert_array_equal(host["half_size"].reshape(E, G)[e, :m],
                                              hs_a[e, :m],
                                              err_msg=f"{label}: env {e}: half_size")
        assert np.any(np.nan_to_num(pos[..., 2]) != 0.0), f"{label}: the helix is flat"
    return check


@pytest.mark.parametrize("name", list(_PROVIDES))
def test_provided_attributes_reach_the_sequence_under_capture(acme, name):
    """An anchors generator that provides z / tangent (``acme.helix``; ``_posed`` also
    half_size / roll): ``presets.gates`` drops ``gate_arclen``, ``elevation`` and
    ``tangents``, inserts ``carry.<attr>`` per provided attribute after ``order.raw``,
    and (posed) selects ``gate_frames.posed``. Recorded into the caller's graph, every
    replay (also after a reseed) carries the provided values of THAT replay into the
    GateSequence unchanged (``carry.*`` and ``_finalize_frame[_posed]_k`` on CUDA)."""
    attrs = _PROVIDES[name]
    spec = _gates(generator=name)
    for dropped in ("gate_arclen", "elevation", "tangents"):
        assert dropped not in spec.ids, spec.ids
    for a in attrs:
        assert f"carry_{a}" in spec.ids, spec.ids
    assert _ref(spec, "order").name == "raw"
    assert _ref(spec, "gate_frames").name == ("posed" if "half_size" in attrs else "default")
    _record_replay_reseed(spec, name, check=_check_provided(attrs))


def test_superformula_stations_emit_their_gate_count(acme):
    """``presets.gates(generator="acme.superformula")`` (a third-party TRACK generator)
    resolves to the stations source: ``stations.raw`` (auto) or ``stations.relaxed``
    after the track pipeline's resample + relax; every env gets exactly ``gates``
    stations and some env is valid (the recorded comparisons are not on empty
    sequences)."""
    for source, station, extra in (("auto", "raw", ()),
                                   ("relaxed_stations", "relaxed",
                                    ("resample", "width", "relax"))):
        spec = _superformula(source)
        assert _ref(spec, "generator").name == SUPERFORMULA, spec.ids
        assert _ref(spec, "stations").name == station, spec.ids
        for sid in extra:
            assert sid in spec.ids, spec.ids
        host = _eager(spec)
        np.testing.assert_array_equal(host["count"], G_ST, err_msg=source)
        assert host["valid"].any(), f"{source}: no valid env {host['fail_mask'].tolist()}"


# ---------------------------------------------------------------------------------------
# Dynamic parameters written between replays
# ---------------------------------------------------------------------------------------

# label -> (spec of a value (None: the default), stage kind, stage name, dynamic field,
#           a uniform value, per-env values)
_DYNAMIC = {
    "superformula-stations": (
        lambda v: _superformula() if v is None else _superformula(generator_params={"m": v}),
        "generator", SUPERFORMULA, "m", 3.0, [4.0, 8.0]),
    "superformula-relaxed": (
        lambda v: _superformula("relaxed_stations") if v is None else
        _superformula("relaxed_stations", generator_params={"m": v}),
        "generator", SUPERFORMULA, "m", 3.0, [4.0, 8.0]),
    "hills-on-gates": (
        lambda v: _gates(elevation=HILLS if v is None else (HILLS, {"amplitude": v})),
        "elevation", HILLS, "amplitude", 1.0, [0.5, 2.0]),
    "max_height": (
        lambda v: _height_spec(z_cap=v),
        # The default z_cap (1.0) fails nearly every uplifted env; _ZCAP_HIGH none.
        "validity", MAX_HEIGHT, "z_cap", _ZCAP_HIGH, [0.0, _ZCAP_HIGH]),
}


@pytest.mark.parametrize("case", list(_DYNAMIC))
def test_dynamic_param_written_between_replays_reaches_the_next_replay(acme, case):
    """ONE recording of ``generate(stream=s)`` on a gate Program with a dynamic field
    (the superformula's ``m`` feeding the stations, the hills' ``amplitude`` on gates,
    the gate check's ``z_cap``). Between replays the field is written in place
    (``gen.params[stage_id].set``: a scalar, an ``[E]`` array, then the default). The
    recording executed nothing; each write lands in the live array (stable pointer) and
    the next replay latches it and equals the eager twin(s) built with those values
    (per-env values: rows stitched from scalar twins)."""
    make, kind, name, field, uniform, per_env = _DYNAMIC[case]
    spec = make(None)
    gen, _ = _make(spec)
    sid = _stage_id(spec, kind, name)
    live = gen.params[sid][field]
    ptr = live.ptr
    default = float(getattr(gen.params[sid].params, field))
    cache: dict = {}

    def twin(v) -> dict:
        key = float(np.float32(v))
        if key not in cache:
            cache[key] = _eager(make(key))
        return cache[key]

    def rows(values) -> dict:
        return _stitch([twin(v) for v in _per_env(values)])

    base = _snapshot(gen.program.result)
    _assert_host(base, twin(default), f"{case}: construction batch")

    s = _side_stream()
    with _recording(s) as cap:
        out = gen.generate(stream=s)
    assert out is gen.program.result and gen.graph is None
    np.testing.assert_array_equal(live.numpy(), _per_env(default),
                                  err_msg="the recording executed a write")
    _assert_host(_snapshot(out), base, f"{case}: the recording executed the pipeline")

    schedule = (_per_env(uniform), _per_env(per_env), _per_env(default))
    assert _differs(rows(schedule[0]), twin(default)), \
        f"{field}={uniform} must change the batch (vacuous otherwise)"
    assert _differs(rows(schedule[1]), rows(schedule[0])), \
        f"per-env {field} must change the batch (vacuous otherwise)"
    for k, values in enumerate(schedule):
        label = f"{case}: replay {k} {field}={values.tolist()}"
        _set(gen, sid, field, values)
        np.testing.assert_array_equal(live.numpy(), values, err_msg=f"{label}: live")
        tg_capture.poison_all(out)
        _replay(cap.graph, s)
        got = _snapshot(out)
        _assert_host(got, rows(values), label)
        _assert_valid_is_fail_mask_zero(got, label)
    assert gen.params[sid][field] is live and live.ptr == ptr, "stable live array"
    assert gen.graph is None


# ---------------------------------------------------------------------------------------
# on_invalid="poison"
# ---------------------------------------------------------------------------------------

def test_poison_records_into_caller_capture():
    """Bezier gates with ``min_gates`` inside the spread of the per-env counts (the
    built-in count bit fails some envs, not all) and ``on_invalid="poison"``: eagerly
    the batch is exactly the ``"keep"`` batch with the invalid envs poisoned (``count =
    0``, NaN per-gate fields, ``valid = 0``, ``fail_mask`` / ``generation`` kept), the
    count bit is set exactly where the keep count is below ``min_gates``; recorded into
    a caller's graph, every replay into poisoned outputs equals the poisoned keep twin,
    before and after a reseed."""
    split = _count_split()
    spec = _gates(min_gates=split, on_invalid="poison")
    keep_spec = _gates(min_gates=split)
    snap = _eager(spec)
    keep = _eager(keep_spec)
    bad = snap["fail_mask"] != 0
    assert bad.any() and not bad.all(), f"vacuous poison split: {snap['fail_mask'].tolist()}"
    count_bit = (keep["fail_mask"] >> GATE_FAIL_BITS["count"]) & 1
    np.testing.assert_array_equal(count_bit, (keep["count"] < split).astype(count_bit.dtype),
                                  err_msg="count bit where count < min_gates")
    np.testing.assert_array_equal(snap["count"][bad], 0)
    assert np.isnan(snap["position"].reshape(E, -1)[bad]).all()
    _assert_host(snap, _poisoned(keep), "poison == keep with the invalid envs poisoned")

    s = _side_stream()
    gen, rng = _make(spec)
    with _recording(s) as cap:
        out = gen.generate(stream=s)
    assert gen.graph is None
    for k, seed in enumerate((11, 101, 11)):
        if k:
            _reseed(rng, seed)
        label = f"replay {k} (seeds {seed}..)"
        tg_capture.poison_all(out)
        _replay(cap.graph, s)
        got = _snapshot(out)
        _assert_host(got, _poisoned(_eager(keep_spec, seed)), f"{label}: vs keep, poisoned")
        _assert_host(got, _eager(spec, seed), f"{label}: vs poison twin")
        _assert_valid_is_fail_mask_zero(got, label)
    _assert_kit_capturable(gen, "poison")


def test_gate_check_param_written_inside_the_caller_graph_with_poison(acme):
    """ONE caller graph holds ``[set z_cap from a device source] -> generate()`` on a
    gate Program with ``acme.max_height`` and ``on_invalid="poison"``. Recording
    executes nothing. Before each replay the source is rewritten outside the graph; the
    replay copies it into the live array, latches it, and equals the poison twins built
    with those values (per-env stitched), which are the keep twins with the invalid
    envs poisoned: the check's own bit, and the poisoned set, follow ``z_cap`` on
    device replay by replay."""
    spec = _height_spec("poison")
    gen, _ = _make(spec)
    sid = _stage_id(spec, "validity", MAX_HEIGHT)
    bit = gen.program.fail_bits[sid]
    assert 16 <= bit <= 30, bit
    default = float(getattr(gen.params[sid].params, "z_cap"))
    src = _dev(_per_env(default))
    gen.params[sid].set("z_cap", src)               # warm-up: a no-op write
    wp.synchronize()
    base = _snapshot(gen.program.result)

    s = _side_stream()
    with _recording(s) as cap:
        gen.params[sid].set("z_cap", src, stream=s)
        out = gen.generate(stream=s)
    assert gen.graph is None
    np.testing.assert_array_equal(gen.params[sid]["z_cap"].numpy(), _per_env(default),
                                  err_msg="the recording executed the z_cap write")
    _assert_host(_snapshot(out), base, "the recording executed the pipeline")

    cache: dict = {}

    def twin(on_invalid, z_cap) -> dict:
        key = (on_invalid, float(z_cap))
        if key not in cache:
            cache[key] = _eager(_height_spec(on_invalid, z_cap=z_cap))
        return cache[key]

    schedule = (_ZCAP_SPLIT, _per_env(_ZCAP_HIGH), _ZCAP_SPLIT[::-1].copy())
    seen = []
    for k, values in enumerate(schedule):
        label = f"replay {k} z_cap={values.tolist()}"
        _write(src, values)
        tg_capture.poison_all(out)
        _replay(cap.graph, s)
        np.testing.assert_array_equal(gen.params[sid]["z_cap"].numpy(), values,
                                      err_msg=f"{label}: live z_cap")
        got = _snapshot(out)
        _assert_host(got, _stitch([twin("poison", v) for v in values]), f"{label}: vs twin")
        _assert_host(got, _poisoned(_stitch([twin("keep", v) for v in values])),
                     f"{label}: vs keep twin, poisoned")
        _assert_valid_is_fail_mask_zero(got, label)
        own = (got["fail_mask"] >> bit) & 1
        np.testing.assert_array_equal(own, (values == 0.0).astype(own.dtype),
                                      err_msg=f"{label}: bit set where max z > z_cap")
        np.testing.assert_array_equal(got["count"][got["fail_mask"] != 0], 0, err_msg=label)
        seen.append(got)
    assert _differs(seen[0], seen[1]) and _differs(seen[1], seen[2])


# ---------------------------------------------------------------------------------------
# Masked regeneration (the gate commit) under capture
# ---------------------------------------------------------------------------------------

def _prepared(spec) -> GateGenerator:
    """A generator of ``spec`` with its masked body warmed up with an all-zero mask,
    which must be an exact no-op (nothing committed, no epoch advanced)."""
    g, _ = _make(spec)
    base = _snapshot(g.program.result)
    m = g.inputs["regen_mask"]
    np.testing.assert_array_equal(m.numpy(), _ZERO, err_msg="regen_mask starts zeroed")
    assert g.enqueue(advance=True, mask=m) is g.program.result
    wp.synchronize()
    _assert_host(_snapshot(g.program.result), base, "an all-zero mask must change nothing")
    np.testing.assert_array_equal(g.seeds.epoch.numpy(), _ZERO)
    return g


def _new_batch(cur: dict, old: dict, e: int) -> bool:
    """Whether env ``e``'s gates changed (an env poisoned in both counts as new: its NaN
    rows cannot differ)."""
    both_poisoned = cur["count"][e] == 0 == old["count"][e] and \
        cur["fail_mask"][e] != 0 and old["fail_mask"][e] != 0 and \
        np.isnan(cur["position"].reshape(E, -1)[e]).all()
    return both_poisoned or not np.array_equal(cur["position"].reshape(E, -1)[e],
                                               old["position"].reshape(E, -1)[e],
                                               equal_nan=True)


def _masked_spec(source: str, on_invalid: str):
    if source == "bezier":
        return _gates(min_gates=_count_split(), on_invalid=on_invalid)
    assert source == "superformula" and on_invalid == "keep", (source, on_invalid)
    return _superformula()


@pytest.mark.parametrize("source,on_invalid",
                         [("bezier", "keep"), ("bezier", "poison"), ("superformula", "keep")])
def test_masked_regeneration_in_caller_graph_writes_only_the_masked_envs(
        acme, source, on_invalid):
    """ONE caller graph holds ``[copy mask_src -> inputs["regen_mask"]] ->
    generate(advance=True, mask=regen_mask)`` (the mask produced in-graph), on an
    anchors source (bezier, ``min_gates`` splitting the counts) and a stations source
    (superformula). Before every replay the WHOLE GateSequence is poisoned
    (``generation`` and ``fail_mask`` too); after it:

    - envs with ``mask[e] == 0`` still hold the poison byte for byte: the gate commit
      left them exactly as the replay found them (an all-zero mask changes no byte);
    - masked envs equal an eager twin driven with the same masks (same seed and epoch),
      with ``generation == epoch``, and got a new batch;
    - with ``"poison"`` the masked invalid envs are poisoned exactly as the ``"keep"``
      twin with its invalid envs poisoned (some masked envs poisoned, some kept);
    - after an all-ones mask every env is committed and the batch equals the twin."""
    spec = _masked_spec(source, on_invalid)
    gen, twin = _prepared(spec), _prepared(spec)
    keep_twin = _prepared(_masked_spec(source, "keep")) if on_invalid == "poison" else None
    out = gen.program.result
    m = gen.inputs["regen_mask"]
    twins = [t for t in (twin, keep_twin) if t is not None]
    base = _snapshot(out)
    mask_src = _dev(_ZERO, wp.int32)

    s = _side_stream()
    with _recording(s) as cap:
        wp.copy(m, mask_src, stream=s)
        assert gen.generate(stream=s, advance=True, mask=m) is out
    assert gen.graph is None, "recording into the caller's graph must not self-capture"
    np.testing.assert_array_equal(gen.seeds.epoch.numpy(), _ZERO,
                                  err_msg="the recording executed the epoch advance")
    _assert_host(_snapshot(out), base, "the recording executed the pipeline")

    schedule = [_M1, _M1, _ZERO, 1 - _M1, _ONES]
    epoch = np.zeros(E, np.int32)
    committed = {k: v.copy() for k, v in base.items()}
    poisoned_masked = kept_masked = 0
    for k, mk in enumerate(schedule):
        label = f"{source}/{on_invalid}: replay {k} mask {mk.tolist()}"
        _write(mask_src, mk)
        tg_capture.poison_all(out)
        poison = _snapshot(out)
        _replay(cap.graph, s)
        np.testing.assert_array_equal(m.numpy(), mk, err_msg=f"{label}: in-graph mask copy")
        for t in twins:
            _write(t.inputs["regen_mask"], mk)
            t.enqueue(advance=True, mask=t.inputs["regen_mask"])
        wp.synchronize()
        ref = _snapshot(twin.program.result)
        epoch += mk
        cur = _snapshot(out)
        masked, unmasked = np.flatnonzero(mk), np.flatnonzero(mk == 0)
        _assert_host(_rows(cur, unmasked), _rows(poison, unmasked),
                     f"{label}: an unmasked env's GateSequence changed")
        _assert_host(_rows(cur, masked), _rows(ref, masked), f"{label}: masked envs vs twin")
        np.testing.assert_array_equal(gen.seeds.epoch.numpy(), epoch, err_msg=label)
        np.testing.assert_array_equal(cur["generation"][masked], epoch[masked], err_msg=label)
        _assert_valid_is_fail_mask_zero(_rows(cur, masked), label)
        if keep_twin is not None:
            keep = _poisoned(_snapshot(keep_twin.program.result))
            _assert_host(_rows(cur, masked), _rows(keep, masked),
                         f"{label}: masked envs vs keep twin, poisoned")
            bad = cur["fail_mask"][masked] != 0
            np.testing.assert_array_equal(cur["count"][masked][bad], 0, err_msg=label)
            poisoned_masked += int(bad.sum())
            kept_masked += int((~bad).sum())
        for e in masked:
            assert _new_batch(cur, committed, e), f"{label}: masked env {e}: no new batch"
            for name, v in cur.items():
                committed[name].reshape(E, -1)[e] = v.reshape(E, -1)[e]
    _assert_host(cur, ref, "after an all-ones mask every env is committed")
    if on_invalid == "poison":
        assert poisoned_masked and kept_masked, (poisoned_masked, kept_masked)


@wp.kernel
def _bump_k(counter: wp.array(dtype=wp.int32)):
    counter[0] = counter[0] + 1


@wp.kernel
def _shift_k(counter: wp.array(dtype=wp.int32), xy: wp.array(dtype=wp.vec2f)):
    t = wp.tid()
    xy[t] = xy[t] + wp.vec2f(1.0e-3 * float(counter[0]), 0.0)


def _perturb_relax(gen) -> wp.array:
    """Wrap the ``gate_relax`` stage's enqueue (test-only): after it, bump a device
    counter and shift its version of ``gates.xy`` (the final one: nothing later writes
    it, and the GateSequence does not export it) of EVERY env by ``1e-3 * counter``, so
    each generation recomputes every env differently. Returns the counter."""
    program = gen.program
    assert program._writers("gates.xy")[-1] == "gate_relax", program._writers("gates.xy")
    i = list(program._stage_ids).index("gate_relax")
    base = program._enqueues[i]
    xy = program.tap("gates.xy", after="gate_relax")
    counter = wp.zeros(1, dtype=wp.int32, device=DEV)
    wp.synchronize()

    def enqueue(ctx):
        base(ctx)
        ctx.launch(_bump_k, 1, [counter])
        ctx.launch(_shift_k, int(xy.shape[0]), [counter, xy])

    program._enqueues[i] = enqueue
    return counter


def test_masked_regeneration_under_capture_does_not_rely_on_recomputing_unmasked_envs():
    """The ``gate_relax`` output is perturbed differently on every generation (a device
    counter bumped in-graph), so the pipeline's recomputation of an unmasked env DIFFERS
    from replay to replay (its ``tap("gates.xy")`` row changes). A recorded masked
    regeneration still leaves every unmasked env's GateSequence rows bitwise unchanged
    (the gate commit restores their held bytes, it never relies on recomputing them
    identically), while the masked envs equal an eager twin perturbed the same way (the
    in-graph counter is read at replay time)."""
    spec = _gates(gate_width=0.05)
    assert _ref(spec, "gate_relax").name == "spheres", spec.ids
    gen, twin = _prepared(spec), _prepared(spec)
    out = gen.program.result
    m = gen.inputs["regen_mask"]
    scratch = wp.zeros(4, dtype=wp.vec2f, device=DEV)         # load the test kernels
    c0 = wp.zeros(1, dtype=wp.int32, device=DEV)
    wp.launch(_bump_k, 1, inputs=[c0], device=DEV)
    wp.launch(_shift_k, 4, inputs=[c0, scratch], device=DEV)
    wp.synchronize()
    counter, counter_t = _perturb_relax(gen), _perturb_relax(twin)
    mask_src = _dev(_ZERO, wp.int32)

    s = _side_stream()
    with _recording(s) as cap:
        wp.copy(m, mask_src, stream=s)
        gen.generate(stream=s, advance=True, mask=m)
    assert int(counter.numpy()[0]) == 0, "the recording executed the pipeline"

    tap = gen.program.tap
    for k, mk in enumerate([_M1, _M1, 1 - _M1]):
        label = f"replay {k} mask {mk.tolist()}"
        before = _snapshot(out)
        tap_before = tap("gates.xy").numpy().reshape(E, -1).copy()
        _write(mask_src, mk)
        _replay(cap.graph, s)
        _write(twin.inputs["regen_mask"], mk)
        twin.enqueue(advance=True, mask=twin.inputs["regen_mask"])
        wp.synchronize()
        assert int(counter.numpy()[0]) == int(counter_t.numpy()[0]) == k + 1, label
        cur = _snapshot(out)
        tap_now = tap("gates.xy").numpy().reshape(E, -1)
        masked, unmasked = np.flatnonzero(mk), np.flatnonzero(mk == 0)
        recomputed = [e for e in unmasked
                      if not np.array_equal(tap_now[e], tap_before[e], equal_nan=True)]
        live = [e for e in unmasked if before["count"][e] > 0]
        assert live and recomputed == live, \
            f"{label}: the unmasked recomputation did not change (vacuous): {recomputed}"
        _assert_host(_rows(cur, unmasked), _rows(before, unmasked),
                     f"{label}: an unmasked env's GateSequence changed")
        _assert_host(_rows(cur, masked), _rows(_snapshot(twin.program.result), masked),
                     f"{label}: masked envs vs the perturbed twin")


# ---------------------------------------------------------------------------------------
# The conformance kit on CUDA
# ---------------------------------------------------------------------------------------

_KIT_STAGES = (("order", REVERSE), ("stations", "raw"), ("stations", "relaxed"),
               ("validity", MAX_HEIGHT))


def _assert_kit_captured(report, label: str) -> None:
    report.raise_for_failures()
    caller = report.result("capture.caller", DEV)
    assert not caller.skipped and caller.details.get("warp") == "ok", f"{label}: {report}"
    assert not report.result("capture.self_graph", DEV).skipped, f"{label}: {report}"


@pytest.mark.parametrize("kind,name", _KIT_STAGES, ids=[f"{k}.{n}" for k, n in _KIT_STAGES])
def test_gate_stage_passes_check_stage_on_cuda(acme, kind, name):
    """``check_stage`` on ``cuda:0`` for the new kinds (a third-party order, both
    built-in stations stages) and a gate validity plugin: every check passes, the stage
    recorded into a Warp caller capture (and the torch modes when available) replays to
    the eager twin, and the Program's own self-graph replays the eager batch."""
    report = check_stage(kind, name, devices=[DEV])
    _assert_kit_captured(report, f"{kind}.{name}")
    if kind == "validity":
        assert not report.result("validity.bits", DEV).skipped, str(report)


@pytest.mark.parametrize("name", [HELIX, HELIX_POSED, SCATTER])
def test_toy_gate_generator_passes_check_gate_generator_on_cuda(acme, name):
    """``check_gate_generator`` on ``cuda:0`` for the anchors toys (providers and a plain
    cloud): every check passes and the caller / self-graph recordings replay to the
    eager batch."""
    _assert_kit_captured(check_gate_generator(name, devices=[DEV]), name)
