"""The conformance kit (``track_gen.testing``) over TrackGen's own generators, the plugin
toys and deliberately broken toys, plus the kit's building blocks.

- Every built-in track and gate generator passes ``check_generator`` /
  ``check_gate_generator`` (CPU here; the same checks on ``cuda:0`` under the ``cuda``
  marker), and keeps the stronger promises the built-ins make on top of the contract
  (every env generator-valid with a finite raw centerline, a finite ``Track`` before
  ``count`` for every env). These replace the per-generator contract tests that the
  generator test modules used to duplicate (registration, raw shape / finiteness, NaN
  padding, count range, closed loop, determinism, diversity, repeated-call stability).
- The plugin toys (:mod:`tests._plugin_toys`, public plugin surface only) pass, through
  the opt-in pytest plugin's ``track_gen_conformance`` fixture.
- Each broken toy (:mod:`tests._broken_toys`) fails exactly the check it breaks, with a
  message that names the defect; each edge toy (unusual but conforming) passes.
"""
from __future__ import annotations

import contextvars
import functools
import os
import re
import subprocess
import sys
import threading

import numpy as np
import pytest
import warp as wp
import warp._src.context as _warp_context
from warp import launch as _launch_bound_at_import
from warp import synchronize as _sync_bound_at_import

from tests._broken_toys import BROKEN, EDGE, FIVE, register_broken
from tests._plugin_toys import register_stage_toys, register_toys
from tests.conftest import CUDA_AVAILABLE
from track_gen import PerEnvSeededRNG, TrackGenConfig, TrackGenerator
from track_gen.plugin import GeneratorSpec, registry
from track_gen.testing import (
    CHECKS,
    BufferSet,
    CheckResult,
    HotPathViolation,
    Report,
    Site,
    assert_caller_capturable,
    assert_pointers_within,
    assert_same_topology,
    buffers_of,
    check_gate_generator,
    check_generator,
    check_stage,
    hot_path_guard,
    no_allocation,
    no_host_sync,
    stream_explicit,
    trace_launches,
)
from track_gen.testing import capture as tg_capture

# The kit's opt-in pytest plugin (the track_gen_conformance fixture), loaded the way a
# plugin author's conftest.py loads it. Not imported here before pytest loads it, so
# pytest can rewrite its asserts.
pytest_plugins = ["track_gen.testing.pytest_plugin"]

CPU = ["cpu"]
CUDA = ["cuda:0"]

# The repulsive generator's host-scheduled growth loop is O(N^2) per step: run the kit on
# the small config tests/test_generate_repulsive.py uses (the default config is covered
# by its CUDA yield test), on 8 envs.
# repulsive: a small config with a loose stall tolerance, so the kit's batches freeze and
# its loop stops early (the bounded.absorbing probe is exercised) on every platform.
_CONFIG = {"repulsive": dict(num_points=64, repulsive_stages=(16, 32, 64), N_max=384,
                             half_width=0.1, repulsive_stall_area_tol=0.5)}
# The batch the per-generator contract tests used to check (count range, finite prefix,
# NaN padding on 64 envs); repulsive keeps the kit's default of 8.
_NUM_ENVS = 64
_NUM_ENVS_SMALL = {"repulsive": 8}

# Built-ins only: tests/conftest.py sets TRACK_GEN_PLUGINS=off before collection.
TRACK = sorted(registry.available("generator"))
GATES = sorted(registry.available("gate_generator"))
# Every track generator at its default config, plus config variants the generator test
# modules pin (voronoi at 64 sites, as tests/test_generate_voronoi.py's cell-count test;
# hull and voronoi on the configs their end-to-end tests used).
TRACK_CASES = [(n, _CONFIG.get(n)) for n in TRACK] + [
    ("voronoi", {"voronoi_num_sites": 64, "num_points": 96, "N_max": 192}),
    ("hull", {"num_points": 128, "N_max": 256}),
    ("voronoi", {"num_points": 128, "N_max": 256, "half_width": 0.1})]
# Every gate generator in its default ordering, plus the random_pairs ordering of the
# point-family generators (its own RNG stream).
GATE_CASES = [(n, None) for n in GATES] + [
    (n, {"gate_ordering": "random_pairs"}) for n in ("bezier", "hull") if n in GATES]
TOYS = [("generator", "toy.circle"), ("generator", "toy.lobed"),
        ("generator", "toy.wobble"), ("generator", "toy.wobble_host"),
        ("generator", "toy.wobble_input"), ("gate_generator", "toy.ring")]

# A fragment each broken toy's failure message must contain (the defect, located).
_DEFECT = {
    "broken.host_sync": r"wp\.array\.numpy\(\) from tests\._broken_toys\.host_sync_generate:",
    "broken.alloc": r"wp\.empty\(\) from tests\._broken_toys\.alloc_generate:",
    "broken.nan": r"env 0 is marked valid \(out_valid=1\) but its raw centerline has 1 "
                  r"non-finite point",
    "broken.topology": r"tests\._broken_toys\._circle_flat_k \(from "
                       r"tests\._broken_toys\.topology_generate:\d+\): dim \(\d+,\) vs "
                       r"\(\d+,\)",
    "broken.batch": r"env 0 \(seed 11\) changes when the other envs' seeds are permuted "
                    r"\(raw\.centerline",
    "broken.legacy_stream": r"wp\.launch without stream= from "
                            r"tests\._broken_toys\.legacy_stream_generate:.*accepts_ctx=False",
    "broken.stateful": r"generate\(\) #1 with unchanged seeds differs from the first batch",
    "broken.constant": r"env 0 is identical for seeds 11 and 1009",
    "broken.open": r"exceeds 3x the median step and 1\.5x the longest step: env 0: "
                   r"\d+\.\dx",
    "broken.global_buffer": r"tests\._broken_toys\._circle_k argument 3 \(from "
                            r"tests\._broken_toys\.global_buffer_generate:",
    "broken.shared_scratch": r"2 launch argument\(s\) use the same buffer in two generator "
                             r"instances: tests\._broken_toys\._circle_k argument 3 \(from "
                             r"tests\._broken_toys\.shared_scratch_generate:",
    "broken.unwritten": r"env 0 is marked valid \(out_valid=1\) but 1 raw point\(s\) were "
                        r"not written, first at index 255 \(it still holds the kit's poison\)",
    "broken.no_valid": r"out_valid\[e\] was not written for env\(s\) \[0, 1, 2, 3, 4, 5\] "
                       r"\.\.\. \(it still holds the kit's poison 7\)",
    "broken.tiled": r"wp\.launch_tiled without stream= from "
                    r"tests\._broken_toys\.tiled_generate:",
    "broken.assign": r"wp\.array\.assign\(\) from tests\._broken_toys\.assign_generate:",
}


# The BOUNDED / HOST tier checks (roadmap PR12): skipped "not applicable" for a unit
# without a bounded loop / a HOST stage.
_TIERS = ("bounded.absorbing", "bounded.lowering", "host.require_caller_capturable")
_CUDA_ONLY = ("capture.caller", "capture.self_graph", "bounded.lowering")


def _cpu_names(kind: str) -> list:
    """The checks the kit runs on a CPU device, in order."""
    return [n for n in CHECKS[kind] if n not in _CUDA_ONLY]


# Roadmap PR13a: the attribute checks are skipped ("no attributes") for a generator that
# writes no per-point attribute (every built-in).
_ATTRS = ("attributes.written", "attributes.carried")


def _seeded(report: Report) -> bool:
    """Whether the checked generator's output depends on its seed (roadmap PR13b:
    ``Stage.seeded``; ``generator/external`` declares ``False``)."""
    return registry.describe(report.kind, report.name).seeded


# Roadmap PR13c: params.per_env is skipped ("no dynamic parameters") for a unit whose
# stage has no dynamic field nobody draws (external, most toys; roadmap PR13d: repulsive
# and the bezier / hull gate generators have a dynamic size, so it runs for them).
_PER_ENV = "params.per_env"
_NO_DYNAMIC = "skipped: no dynamic parameters"


def _skipped_besides_tiers(report: Report) -> list:
    return [r.name for r in report if r.skipped and r.name not in _TIERS
            and not (r.name in _ATTRS and r.message == "skipped: no attributes")
            and not (r.name == _PER_ENV and r.message == _NO_DYNAMIC)
            and not (r.name == "diversity" and not _seeded(report))]


# The built-ins whose stage has dynamic params (roadmap PR13c; PR13d: + repulsive and the
# bezier / hull gate generators, their size): params.per_env runs.
_DYNAMIC_TRACK = {"bezier", "hull", "polar", "voronoi", "checkpoint", "repulsive"}
_DYNAMIC_GATES = {"bezier", "hull", "polar", "voronoi", "checkpoint"}


def _assert_builtin_promises(report: Report, device: str) -> None:
    """What every built-in guarantees beyond the contract: every env is generator-valid
    with a finite raw output, the result is finite before count for EVERY env (the
    contract asks it of valid envs only), every track has at least one point, and a
    different seed moves the raw output by more than rounding (a seed-dependent
    generator: ``seeded=False`` reports ``diversity`` skipped)."""
    finite = report.result("output.finite", device).details
    assert finite["final_finite_all"], finite
    diversity = report.result("diversity", device)
    if not _seeded(report):
        assert diversity.skipped and "seeded=False" in diversity.message, diversity
    else:
        assert diversity.details["compared"] >= 1, diversity
        assert diversity.details["min_raw_delta"] > 1e-3, diversity
        assert diversity.details["min_pair_delta"] > 1e-3, diversity
    if report.kind == "generator":
        assert finite["raw_valid_fraction"] == 1.0, finite
        assert finite["raw_finite_all"], finite
        count = report.result("output.count", device)
        assert count.details["min"] >= 1, count


# ---------------------------------------------------------------------------------------
# Built-in generators
# ---------------------------------------------------------------------------------------

def test_the_builtins_are_all_checked():
    assert {"bezier", "checkpoint", "hull", "polar", "repulsive", "voronoi"} <= set(TRACK)
    assert {"bezier", "checkpoint", "hull", "polar", "voronoi"} <= set(GATES)


@pytest.mark.parametrize("kind,names", [("generator", TRACK), ("gate_generator", GATES)])
def test_builtins_are_capturable_and_stream_explicit(kind, names):
    for name in names:
        info = registry.describe(kind, name)
        assert info.builtin and info.capturable and info.accepts_ctx, info


def _case_id(name, config):
    labels = [f"{k}={config[k]}" for k in ("gate_ordering", "voronoi_num_sites",
                                           "num_points")
              if k in (config or {})]
    if config and not labels:
        labels = ["small"]
    return "-".join([name] + labels)


@pytest.mark.parametrize("name,config", TRACK_CASES,
                         ids=[_case_id(n, c) for n, c in TRACK_CASES])
def test_builtin_generator_conforms_cpu(name, config):
    report = check_generator(name, devices=CPU, config=config,
                             num_envs=_NUM_ENVS_SMALL.get(name, _NUM_ENVS))
    report.raise_for_failures()
    assert [r.name for r in report] == _cpu_names("generator")   # no capture.*
    assert not _skipped_besides_tiers(report)
    per_env = report.result(_PER_ENV)
    assert per_env.skipped == (name not in _DYNAMIC_TRACK), per_env
    # No built-in has a HOST stage; repulsive is the BOUNDED one (its loop stops early for
    # at least one of the kit's batches, so the absorbing probe is exercised; which one is
    # platform-dependent).
    assert report.result("host.require_caller_capturable").skipped
    absorbing = report.result("bounded.absorbing")
    assert absorbing.skipped == (name != "repulsive"), absorbing
    if name == "repulsive":
        assert absorbing.passed and sum(absorbing.details["skipped"].values()) >= 1, \
            absorbing
        assert report.result("hot_path.host_sync").details["sanctioned"] >= 1
    _assert_builtin_promises(report, "cpu")


@pytest.mark.parametrize("name,config", GATE_CASES,
                         ids=[_case_id(n, c) for n, c in GATE_CASES])
def test_builtin_gate_generator_conforms_cpu(name, config):
    report = check_gate_generator(name, devices=CPU, config=config, num_envs=_NUM_ENVS)
    report.raise_for_failures()
    assert not _skipped_besides_tiers(report)
    per_env = report.result(_PER_ENV)
    assert per_env.skipped == (name not in _DYNAMIC_GATES), per_env
    _assert_builtin_promises(report, "cpu")


@pytest.mark.cuda
@pytest.mark.parametrize("name,config", TRACK_CASES,
                         ids=[_case_id(n, c) for n, c in TRACK_CASES])
def test_builtin_generator_conforms_cuda(name, config):
    report = check_generator(name, devices=CUDA, config=config,
                             num_envs=_NUM_ENVS_SMALL.get(name, _NUM_ENVS))
    report.raise_for_failures()
    assert report.result("capture.caller", "cuda:0").details.get("warp") == "ok"
    _assert_builtin_promises(report, "cuda:0")


@pytest.mark.cuda
@pytest.mark.parametrize("name,config", GATE_CASES,
                         ids=[_case_id(n, c) for n, c in GATE_CASES])
def test_builtin_gate_generator_conforms_cuda(name, config):
    report = check_gate_generator(name, devices=CUDA, config=config, num_envs=_NUM_ENVS)
    report.raise_for_failures()
    _assert_builtin_promises(report, "cuda:0")


# ---------------------------------------------------------------------------------------
# Plugin toys (public plugin surface only), through the opt-in pytest plugin
# ---------------------------------------------------------------------------------------

@pytest.fixture
def toys(plugin_registry):
    register_toys(plugin_registry)
    register_stage_toys(plugin_registry)
    return plugin_registry


@pytest.mark.parametrize("kind,name", TOYS)
def test_plugin_toy_conforms_cpu(kind, name, toys, track_gen_conformance):
    report = track_gen_conformance(name, kind=kind, devices=CPU)
    assert report.ok and report.kind == kind and report.name == name
    if name == "toy.wobble_host":      # capture = HOST: topology / buffers are waived
        for check in ("hot_path.topology", "hot_path.buffers"):
            r = report.result(check)
            assert r.skipped and "HOST stage(s) ['generator'] run eagerly" in r.message
        for check in ("hot_path.host_sync", "hot_path.allocation", "hot_path.stream"):
            assert not report.result(check).skipped     # a HOST stage may not sync itself


@pytest.mark.cuda
@pytest.mark.parametrize("kind,name", TOYS)
def test_plugin_toy_conforms_cuda(kind, name, toys, track_gen_conformance):
    track_gen_conformance(name, kind=kind, devices=CUDA)


def test_plugin_toy_params_are_passed_through(toys):
    report = check_generator("toy.lobed", {"lobes": 7, "amp": 0.2}, devices=CPU)
    report.raise_for_failures()
    with pytest.raises(AssertionError, match="construct"):
        check_generator("toy.lobed", {"lobes": 99}, devices=CPU).raise_for_failures()


# ---------------------------------------------------------------------------------------
# Broken toys: each fails exactly the check it breaks
# ---------------------------------------------------------------------------------------

def test_the_spec_toys_are_covered():
    assert set(FIVE) <= set(BROKEN)
    assert [BROKEN[n][1] for n in FIVE] == ["hot_path.host_sync", "hot_path.allocation",
                                            "output.finite", "hot_path.topology",
                                            "batch_independence"]


@pytest.mark.parametrize("name", sorted(BROKEN))
def test_broken_toy_fails_exactly_its_check(name, plugin_registry):
    register_broken(plugin_registry)
    check = BROKEN[name][1]
    report = check_generator(name, devices=CPU)
    assert report.failed_names == [check], str(report)
    failure = report.result(check, "cpu")
    assert not failure.passed and not failure.skipped
    assert re.search(_DEFECT[name], failure.message), failure.message
    with pytest.raises(AssertionError, match=re.escape(f"[cpu] {check}: FAIL")):
        report.raise_for_failures()


@pytest.mark.parametrize("name", sorted(EDGE))
def test_edge_toy_conforms(name, plugin_registry):
    """A closing straight, NaN-rejected envs, a wp.struct scratch: unusual, conforming."""
    register_broken(plugin_registry)
    report = check_generator(name, devices=CPU)
    assert report.ok and not _skipped_besides_tiers(report), str(report)
    if name == "edge.rejecting":
        assert 0 < report.result("diversity").details["compared"] < 8


@pytest.mark.cuda
@pytest.mark.parametrize("name", ["broken.host_sync", "broken.alloc"])
def test_broken_toy_cannot_be_recorded_on_cuda(name, plugin_registry):
    """On a GPU the readback / the allocation breaks the caller's capture itself."""
    register_broken(plugin_registry)
    report = check_generator(name, devices=CUDA, capture_modes=("warp",))
    assert "capture.caller" in report.failed_names, str(report)


def test_the_pytest_fixture_fails_with_every_failed_check(plugin_registry,
                                                          track_gen_conformance):
    register_broken(plugin_registry)
    with pytest.raises(pytest.fail.Exception,
                       match=r"generator 'broken\.batch' failed 1 conformance check"
                             r"\(s\):\n  \[cpu\] batch_independence: FAIL"):
        track_gen_conformance("broken.batch", devices=CPU)
    with pytest.raises(ValueError, match="kind"):
        track_gen_conformance("toy.circle", kind="banking")


# ---------------------------------------------------------------------------------------
# check_generator: arguments, skips, construct failures, the Report
# ---------------------------------------------------------------------------------------

def test_skip_reports_checks_as_skipped(toys):
    report = check_generator("toy.circle", devices=CPU, skip=("diversity", "hot_path"))
    assert report.ok
    skipped = [r.name for r in report if r.skipped and "skip=" in r.message]
    assert skipped == ["diversity", "hot_path.host_sync", "hot_path.allocation",
                       "hot_path.stream", "hot_path.capture", "hot_path.topology",
                       "hot_path.buffers"]
    assert "skip=" in report.result("diversity").message


def test_construct_failure_is_one_failed_check(plugin_registry):
    def alloc(config):
        raise RuntimeError("no scratch for you")

    plugin_registry.add(GeneratorSpec(name="broken.alloc_scratch", alloc_scratch=alloc,
                                      generate=lambda *a, **k: None, accepts_ctx=True))
    report = check_generator("broken.alloc_scratch", devices=CPU)
    assert report.failed_names == ["construct"]
    assert "RuntimeError: no scratch for you" in report.result("construct").message
    assert all(r.skipped and "construct failed" in r.message
               for r in report if r.name != "construct")


@pytest.mark.parametrize("kwargs,match", [
    (dict(num_envs=1), "num_envs"),
    (dict(seeds=(0, 3)), "overlap"),
    (dict(seeds=(1,)), "two base seeds"),
    (dict(skip=("closeness",)), "unknown check"),
    (dict(skip=("construct",)), "unknown check"),
    (dict(capture_modes=("jax",)), "capture mode"),
    (dict(config={"device": "cpu"}), "must not set"),
    (dict(params={"lobes": 3}, config={"generator_params": {"lobes": 3}}), "not both"),
])
def test_bad_arguments_raise(toys, kwargs, match):
    name = "toy.lobed" if "params" in kwargs else "toy.circle"
    params = kwargs.pop("params", None)
    with pytest.raises(ValueError, match=match):
        check_generator(name, params, devices=CPU, **kwargs)


def test_the_origin_of_a_wrapped_generate_is_its_function(plugin_registry):
    """A functools.partial generate counts as the function it wraps: its package is
    probed for hot_path.stream (here: this test module, through the partial)."""
    from tests._broken_toys import legacy_stream_generate

    spec = GeneratorSpec(name="broken.partial", alloc_scratch=lambda config: None,
                         generate=functools.partial(legacy_stream_generate))
    plugin_registry.add(spec)
    assert registry.describe("generator", "broken.partial").origin == "tests._broken_toys"
    report = check_generator("broken.partial", devices=CPU, skip=("diversity",))
    assert report.failed_names == ["hot_path.stream"], str(report)


def test_modules_adds_probed_packages(toys):
    """modules= extends the stream check to a helper package with another top-level
    name (here the plugin toys' own package is probed anyway; "acme" is extra)."""
    report = check_generator("toy.circle", devices=CPU, modules=("acme",),
                             skip=("batch_independence", "determinism", "diversity",
                                   "statelessness"))
    assert report.ok


def test_unknown_generator_raises_the_registry_error():
    with pytest.raises(ValueError, match="acme.nope"):
        check_generator("acme.nope", devices=CPU)


def test_default_devices(toys):
    report = check_generator("toy.circle",
                             skip=("batch_independence", "hot_path", "capture"))
    want = {"cpu", "cuda:0"} if CUDA_AVAILABLE else {"cpu"}
    assert {r.device for r in report} == want


def test_report_api():
    results = [CheckResult("construct", "cpu", True),
               CheckResult("diversity", "cpu", False, "envs [0, 1] produced identical "
                                                       "tracks"),
               CheckResult("hot_path.topology", "cpu", True, "waived", skipped=True),
               CheckResult("diversity", "cuda:0", False, "again")]
    report = Report("generator", "acme.x", results)
    assert not report.ok and len(report) == 4 and list(report) == results
    assert report.failed_names == ["diversity"]
    assert [r.device for r in report.failures] == ["cpu", "cuda:0"]
    assert report.result("diversity").device == "cpu"
    assert report.result("diversity", "cuda:0").message == "again"
    with pytest.raises(KeyError):
        report.result("determinism")
    text = str(report)
    assert text.splitlines()[0] == "generator 'acme.x': 1/3 checks passed, 1 skipped"
    assert re.search(r"cpu\s+diversity\s+FAIL\s+envs \[0, 1\]", text)
    assert re.search(r"hot_path\.topology\s+SKIP\s+waived", text)
    with pytest.raises(AssertionError) as info:
        report.raise_for_failures()
    msg = str(info.value)
    assert msg.startswith("generator 'acme.x' failed 2 conformance check(s):")
    assert "[cpu] diversity: FAIL - envs [0, 1]" in msg and "[cuda:0] diversity" in msg
    passing = Report("gate_generator", "acme.g", results[:1])
    assert passing.ok and passing.raise_for_failures() is passing


def test_check_names():
    assert CHECKS["generator"][0] == "construct"
    assert "output.closed" in CHECKS["generator"]
    assert "output.accepted" in CHECKS["gate_generator"]
    assert set(CHECKS["generator"]) - {"output.closed"} == \
        set(CHECKS["gate_generator"]) - {"output.accepted"}
    # the track-pipeline stage kinds (roadmap PR10b): no diversity / output.closed /
    # attributes (the generator's business); a width stage adds width.values (PR13a)
    for kind in ("relax", "elevation"):
        assert set(CHECKS[kind]) == set(CHECKS["generator"]) - {
            "output.closed", "diversity", *_ATTRS} | {"output.accepted", "stage.isolation"}
    assert set(CHECKS["width"]) == set(CHECKS["relax"]) | {"width.values"}
    assert set(CHECKS["validity"]) == set(CHECKS["relax"]) - {"output.accepted"} | {
        "validity.bits"}
    assert set(_ATTRS) <= set(CHECKS["generator"]) and \
        set(_ATTRS) <= set(CHECKS["gate_generator"])


# ---------------------------------------------------------------------------------------
# check_stage over TrackGen's own relax / width / elevation / validity stages (PR10b)
# ---------------------------------------------------------------------------------------

_STAGE_KINDS = ("relax", "width", "elevation", "validity")


# width.from_channel reads a channel the user wires (a required read port): it runs
# with a toy generator's carried attribute (test_channel_toys_conform below).
_NEEDS_PORTS = {("width", "from_channel")}


@pytest.mark.parametrize("kind,name", [(k, n) for k in _STAGE_KINDS
                                       for n in registry.available(k)
                                       if (k, n) not in _NEEDS_PORTS])
def test_every_builtin_stage_passes_check_stage(kind, name):
    report = check_stage(kind, name, devices=CPU, num_envs=4)
    report.raise_for_failures()
    assert [r.name for r in report] == _cpu_names(kind)
    assert registry.describe(kind, name).builtin


@pytest.mark.cuda
@pytest.mark.parametrize("kind,name", [("relax", "xpbd"), ("elevation", "noise"),
                                       ("validity", "default")])
def test_builtin_stages_pass_check_stage_on_cuda(kind, name):
    check_stage(kind, name, devices=CUDA, num_envs=4,
                capture_modes=("warp",)).raise_for_failures()


def test_the_fixture_runs_check_stage(track_gen_conformance):
    report = track_gen_conformance("default", kind="validity", devices=CPU, num_envs=4,
                                   skip=("hot_path",))
    assert report.kind == "validity" and "validity.bits" in [r.name for r in report]
    assert str(report).startswith("validity stage 'default': ")


def test_parametrize_helpers():
    from track_gen.testing.pytest_plugin import generator_names, parametrize_generators

    assert generator_names("generator", builtin=True) == TRACK
    assert generator_names("generator", builtin=False) == []
    assert generator_names("gate_generator", exclude=("bezier",)) == \
        [n for n in GATES if n != "bezier"]
    mark = parametrize_generators("gate_generator", argname="g")
    assert mark.mark.name == "parametrize" and mark.mark.args == ("g", GATES)
    with pytest.raises(ValueError, match="kind"):
        generator_names("banking")


# ---------------------------------------------------------------------------------------
# contract: the hot-path guards
# ---------------------------------------------------------------------------------------

def _arr(n=4, dtype=wp.float32):
    return wp.zeros(n, dtype=dtype, device="cpu")


@wp.kernel
def _scale_k(x: wp.array(dtype=wp.float32), s: float):
    i = wp.tid()
    x[i] = x[i] * s


def test_no_host_sync_raises_at_the_call():
    a = _arr()
    with pytest.raises(HotPathViolation,
                       match=r"host sync on a hot path: wp\.array\.numpy\(\) from "
                             r"tests\.test_conformance\.test_no_host_sync_raises_at_the_"
                             r"call:\d+"):
        with no_host_sync():
            a.numpy()
    with pytest.raises(HotPathViolation, match=r"wp\.synchronize_device\(\)"):
        with no_host_sync():
            wp.synchronize_device("cpu")
    with pytest.raises(HotPathViolation, match=r"wp\.array\.list\(\)"):
        with no_host_sync():
            a.list()


def test_a_swallowed_violation_still_fails_the_block():
    a = _arr()
    with pytest.raises(HotPathViolation, match="host sync"):
        with no_host_sync():
            try:
                a.numpy()
            except AssertionError:
                pass


def test_allow_lets_a_known_readback_through():
    a = _arr()
    key = ("wp.array.numpy", __name__, "test_allow_lets_a_known_readback_through")
    with no_host_sync(allow={key}) as guard:
        assert a.numpy().shape == (4,)
    assert guard.allowed == [key] and guard.ok


def test_record_mode_sees_every_category():
    a = _arr()
    with hot_path_guard(modules=(__name__,), strict=False) as guard:
        a.numpy()
        wp.zeros(3, dtype=wp.int32, device="cpu")
        wp.launch(_scale_k, dim=4, inputs=[a, 2.0], device="cpu")
        wp.launch(_scale_k, dim=4, inputs=[a, 2.0], device="cpu", stream=None)  # fine
        wp.copy(a, _arr())       # allocates (the source) and has no stream=
        a.zero_()
        with wp.ScopedStream(None):
            pass
    got = [(v.category, v.api) for v in guard.violations]
    assert got == [("host_sync", "wp.array.numpy()"), ("allocation", "wp.zeros()"),
                   ("stream", "wp.launch without stream="), ("allocation", "wp.zeros()"),
                   ("stream", "wp.copy without stream="), ("stream", "wp.array.zero_()"),
                   ("stream", "wp.ScopedStream")]
    assert all(v.site.module == __name__ for v in guard.violations)
    assert [v.category for v in guard.of("stream")] == ["stream"] * 4
    report = guard.report()
    assert report.splitlines()[0].startswith("host sync (1).")
    with pytest.raises(HotPathViolation):
        guard.raise_for_violations()


def test_capture_calls_are_refused_before_they_run():
    with pytest.raises(HotPathViolation, match=r"wp\.capture_launch\(\)"):
        with hot_path_guard():
            wp.capture_launch(None)
    with pytest.raises(HotPathViolation, match=r"wp\.ScopedCapture\(\)"):
        with hot_path_guard():
            wp.ScopedCapture(device="cpu")


@pytest.mark.parametrize("make,api", [
    (lambda a: wp.empty(3, dtype=wp.float32, device="cpu"), "wp.empty()"),
    (lambda a: wp.zeros(3, dtype=wp.float32, device="cpu"), "wp.zeros()"),
    (lambda a: wp.ones(3, dtype=wp.float32, device="cpu"), "wp.ones()"),
    (lambda a: wp.full(3, 2.0, dtype=wp.float32, device="cpu"), "wp.full()"),
    (lambda a: wp.clone(a), "wp.clone()"),
    (lambda a: wp.empty_like(a), "wp.empty_like()"),
    (lambda a: wp.zeros_like(a), "wp.zeros_like()"),
    (lambda a: wp.array(np.arange(3, dtype=np.float32), device="cpu"), "wp.array()"),
])
def test_no_allocation_sees_every_allocation_path(make, api):
    a = _arr()
    with no_allocation(strict=False) as guard:
        make(a)
    assert [v.api for v in guard.violations] == [api]
    assert guard.violations[0].site.module == __name__
    assert len(guard.allocated()) == 1


def test_views_do_not_allocate():
    a = _arr(8)
    with no_allocation():
        a.reshape((2, 4))
        a[2:]
        a.flatten()


def test_stream_explicit_probes_only_the_listed_modules():
    a = _arr()
    with stream_explicit(("track_gen",)) as guard:
        wp.launch(_scale_k, dim=4, inputs=[a, 2.0], device="cpu")   # not track_gen code
    assert guard.ok
    with pytest.raises(HotPathViolation, match="wp.launch without stream="):
        with stream_explicit(("tests",)):
            wp.launch(_scale_k, dim=4, inputs=[a, 2.0], device="cpu")
    with stream_explicit(("tests.test_conf",)) as guard:    # a dotted prefix, not a string one
        wp.launch(_scale_k, dim=4, inputs=[a, 2.0], device="cpu")
    assert guard.ok


def test_guards_are_context_local_and_restore_warp():
    real = (wp.launch, wp.copy, wp.synchronize, wp.array.numpy, wp.array.zero_)
    a = _arr()
    seen = []
    with no_host_sync():
        assert wp.array.numpy is not real[3]
        t = threading.Thread(target=lambda: seen.append(a.numpy().shape))
        t.start()
        t.join()
        with pytest.raises(HotPathViolation, match="allocation"):
            with no_allocation():               # nested guards compose
                wp.empty(1, device="cpu")
    assert seen == [(4,)]                       # the other thread was not guarded
    assert (wp.launch, wp.copy, wp.synchronize, wp.array.numpy, wp.array.zero_) == real
    guard = no_host_sync()
    with guard:
        with pytest.raises(RuntimeError, match="already active"):
            guard.__enter__()


def test_no_host_sync_switches_the_debug_sync_off(monkeypatch):
    monkeypatch.setenv("TRACKGEN_DEBUG_SYNC", "1")
    with no_host_sync():
        assert "TRACKGEN_DEBUG_SYNC" not in os.environ
    assert os.environ["TRACKGEN_DEBUG_SYNC"] == "1"
    with no_allocation():
        assert os.environ["TRACKGEN_DEBUG_SYNC"] == "1"


@wp.kernel
def _scale_tiled_k(x: wp.array(dtype=wp.float32), s: float):
    i, _lane = wp.tid()
    x[i] = x[i] * s


def test_stream_check_sees_every_launch_path():
    """launch_tiled, a recorded command, assign, a launch bound at import time, Warp's
    defining module, and a Warp utility launching on the caller's behalf."""
    a, b = _arr(), _arr()
    i32 = _arr(4, wp.int32)
    cmd = wp.launch(_scale_k, dim=4, inputs=[a, 2.0], device="cpu", record_cmd=True)
    with stream_explicit((__name__,), strict=False) as guard:
        wp.launch_tiled(_scale_tiled_k, dim=[4], inputs=[a, 2.0], block_dim=1,
                        device="cpu")
        wp.launch_tiled(_scale_tiled_k, dim=[4], inputs=[a, 2.0], block_dim=1,
                        device="cpu", stream=None)                       # fine
        wp.launch(_scale_k, dim=4, inputs=[a, 2.0], device="cpu", record_cmd=True)  # fine
        cmd.launch()
        cmd.launch(None)                                                 # fine
        b.assign(a)
        _launch_bound_at_import(_scale_k, dim=4, inputs=[a, 2.0], device="cpu")
        _warp_context.launch(_scale_k, dim=4, inputs=[a, 2.0], device="cpu")
        wp.utils.array_cast(a, i32)          # launches inside Warp, without stream=
    got = [(v.api, v.site.module) for v in guard.violations]
    assert got == [("wp.launch_tiled without stream=", __name__),
                   ("wp.Launch.launch without stream=", __name__),
                   ("wp.array.assign()", __name__),
                   ("wp.launch without stream=", __name__),
                   ("wp.launch without stream=", __name__),
                   ("wp.launch without stream=", __name__)], got
    assert _launch_bound_at_import is wp.launch          # restored on exit
    assert _warp_context.launch is wp.launch


def test_trace_sees_tiled_recorded_and_assigned_work():
    a, b = _arr(), _arr()
    cmd = wp.launch(_scale_k, dim=4, inputs=[a, 2.0], device="cpu", record_cmd=True)
    with trace_launches() as t:
        wp.launch_tiled(_scale_tiled_k, dim=[4], inputs=[a, 2.0], block_dim=1,
                        device="cpu")
        wp.launch(_scale_k, dim=4, inputs=[b, 3.0], device="cpu", record_cmd=True)
        cmd.launch()
        b.assign(a)
    kernels = t.kernels()
    assert kernels[0].endswith("._scale_tiled_k") and kernels[1].endswith("._scale_k")
    assert kernels[2:] == ["wp.array.assign"]
    assert t[1].arrays == ((0, a.ptr),) and t[1].scalars == ((1, 2.0),)
    assert t[2].op == "copy" and t[2].arrays == ((0, b.ptr), (1, a.ptr))


def test_host_sync_sees_array_interface_and_bound_names():
    a = _arr()
    with no_host_sync(strict=False) as guard:
        np.asarray(a)
        _sync_bound_at_import()
        a.numpy()                         # numpy()'s own use of the interface: once
    assert [v.api for v in guard.violations] == ["wp.array.__array_interface__",
                                                 "wp.synchronize()", "wp.array.numpy()"]


@pytest.mark.cuda
def test_a_copy_across_devices_is_a_host_sync():
    a = wp.zeros(4, dtype=wp.float32, device="cuda:0")
    host = wp.zeros(4, dtype=wp.float32, device="cpu")
    with no_host_sync(strict=False) as guard:
        wp.copy(host, a)
    assert [v.api for v in guard.violations] == ["wp.copy from cuda:0 to cpu"]


def test_native_objects_are_allocations():
    with no_allocation(strict=False) as guard:
        grid = wp.HashGrid(4, 4, 4, device="cpu")
        grid.reserve(16)
    assert [v.api for v in guard.violations] == ["wp.HashGrid()", "wp.HashGrid.reserve()"]
    assert all(v.site.module == __name__ for v in guard.violations)
    assert guard.allocated() == []                      # only wp.arrays are listed


def test_classes_stay_classes_while_a_guard_is_active():
    """The guards patch __init__, not the class: isinstance and subclassing keep working,
    in every thread."""
    seen = []

    def other_thread():
        seen.append(isinstance(None, wp.ScopedStream))

        class Mine(wp.ScopedCapture):
            pass

        seen.append(issubclass(Mine, wp.ScopedCapture))

    classes = (wp.ScopedStream, wp.ScopedCapture, wp.HashGrid)
    with no_host_sync():
        assert (wp.ScopedStream, wp.ScopedCapture, wp.HashGrid) == classes
        t = threading.Thread(target=other_thread)
        t.start()
        t.join()
    assert seen == [False, True]


def test_warp_is_restored_when_a_guard_exits_in_another_context():
    real = wp.launch
    a = _arr()
    guard = no_host_sync()
    guard.__enter__()
    contextvars.copy_context().run(guard.__exit__, None, None, None)
    assert wp.launch is real and not hasattr(wp.array.numpy, "_track_gen_testing_real")
    with stream_explicit(("nothing",)):      # the exited guard no longer listens here
        a.numpy()


def test_a_wrapper_left_by_a_foreign_patch_is_cleared_by_the_next_guard():
    """A guard exits inside a foreign patch of the same attribute, which then restores the
    guard's (pass-through) wrapper: the next guard replaces it instead of wrapping it."""
    real = wp.capture_launch
    guard = hot_path_guard()
    guard.__enter__()
    forbidden = tg_capture.capture_launch_forbidden()
    forbidden.__enter__()
    guard.__exit__(None, None, None)
    forbidden.__exit__(None, None, None)
    stale = wp.capture_launch
    assert stale is not real and stale._track_gen_testing_real is real
    with hot_path_guard():
        assert wp.capture_launch is not stale
        assert wp.capture_launch._track_gen_testing_real is real   # not a double wrap
    assert wp.capture_launch is real


def test_site_is_public():
    with no_host_sync(strict=False) as guard:
        _arr().numpy()
    site = guard.violations[0].site
    assert isinstance(site, Site) and site.module == __name__
    assert str(site) == f"{site.module}.{site.function}:{site.line}"


def test_a_real_generator_hot_path_is_clean():
    gen = TrackGenerator(TrackGenConfig(generator="polar", device="cpu", num_envs=2),
                         PerEnvSeededRNG(seeds=5, num_envs=2, device="cpu"))
    with hot_path_guard() as guard:
        gen.generate()
        gen.enqueue(advance=True)
    assert guard.ok


# ---------------------------------------------------------------------------------------
# trace: launch topology and buffer ownership
# ---------------------------------------------------------------------------------------

def test_trace_records_launches_and_copies():
    a, b = _arr(), _arr()
    with trace_launches() as t:
        wp.launch(_scale_k, dim=4, inputs=[a, 2.5], device="cpu")
        wp.copy(b, a, count=3)
        wp.clone(a)                              # Warp's internal copy is not recorded
    assert len(t) == 2 and t.kernels()[1] == "wp.copy"
    k, c = t[0], t[1]
    assert k.op == "launch" and k.kernel.endswith("._scale_k") and k.dim == (4,)
    assert k.arrays == ((0, a.ptr),) and k.scalars == ((1, 2.5),) and k.stream is None
    assert k.site.module == __name__
    assert c.arrays == ((0, b.ptr), (1, a.ptr)) and c.scalars == ((2, 0), (3, 0), (4, 3))
    assert c.dim == (3,)
    with trace_launches(include_warp=True) as t:
        wp.clone(a)
    assert t.kernels() == ["wp.copy"]


def test_topology_diff():
    a, b = _arr(), _arr()

    def run(x, s, dim=4):
        with trace_launches() as t:
            wp.launch(_scale_k, dim=dim, inputs=[x, s], device="cpu")
            wp.launch(_scale_k, dim=4, inputs=[x, float("nan")], device="cpu")
        return t

    ref = run(a, 2.0)
    assert ref.same_topology(run(a, 2.0))           # NaN scalars compare equal
    assert_same_topology(ref, run(a, 2.0))
    assert "dim (3,) vs (4,)" in run(a, 2.0, dim=3).diff(ref)[0]
    assert "scalar args" in ref.diff(run(a, 3.0))[0]
    assert "buffers" in ref.diff(run(b, 2.0))[0]
    owned = BufferSet().add(a, "a").add(b, "b")
    assert ref.topology(owned)[0][3] == ((0, ("a", 0)),)
    with pytest.raises(AssertionError, match="launch topology differs"):
        assert_same_topology(ref, run(b, 2.0))
    longer = run(a, 2.0)
    longer.records.append(longer.records[0])
    assert ref.diff(longer)[0] == "2 launches/copies vs 3"


class _Slotted:
    __slots__ = ("arr", "other")

    def __init__(self, arr) -> None:
        self.arr = arr
        self.other = 3


def test_buffers_of_walks_objects_and_flags_foreign_pointers():
    import dataclasses

    @dataclasses.dataclass
    class Result:
        x: wp.array
        nested: dict

    a, b, c, d = _arr(), _arr(8), _arr(), _arr()
    obj = Result(x=a, nested={"k": [b, (_Slotted(c),)], "n": 1})
    owned = buffers_of(obj, root="res")
    assert sorted(owned.labels()) == ["res.nested['k'][0]", "res.nested['k'][1][0].arr",
                                      "res.x"]
    assert owned.locate(b.ptr + 8) == ("res.nested['k'][0]", 8)
    assert d.ptr not in owned and a.ptr in owned
    with trace_launches() as t:
        wp.launch(_scale_k, dim=4, inputs=[a, 1.0], device="cpu")
        wp.launch(_scale_k, dim=4, inputs=[d, 1.0], device="cpu")
    foreign = t.foreign_pointers(owned)
    assert [(r.kernel.split(".")[-1], i, p) for r, i, p in foreign] == [("_scale_k", 0, d.ptr)]
    assert t.foreign_pointers(owned, ignore=[d]) == []
    with pytest.raises(AssertionError, match="1 pointer"):
        assert_pointers_within(t, owned)


@wp.struct
class _Pair:
    x: wp.array(dtype=wp.float32)
    scale: float


@wp.kernel
def _pair_k(p: _Pair, y: wp.array(dtype=wp.float32)):
    i = wp.tid()
    y[i] = p.x[i] * p.scale


def test_struct_arguments_are_flattened_and_walked():
    a, b = _arr(), _arr()
    pair = _Pair()
    pair.x = a
    pair.scale = 2.0
    with trace_launches() as t:
        wp.launch(_pair_k, dim=4, inputs=[pair, b], device="cpu")
    assert t[0].arrays == (("0.x", a.ptr), (1, b.ptr))
    assert t[0].scalars == (("0.scale", 2.0),)
    pair.scale = 3.0
    with trace_launches() as u:
        wp.launch(_pair_k, dim=4, inputs=[pair, b], device="cpu")
    assert "scalar args" in t.diff(u)[0]
    owned = buffers_of({"pair": pair}, root="s")
    assert owned.locate(a.ptr) == ("s['pair'].x", 0)


def test_buffers_of_does_not_enter_a_registry_spec(plugin_registry):
    """A spec serves every generator built from it: what its callables close over is
    shared, not owned."""
    shared = _arr(16)

    def alloc(config):
        return None

    def generate(*args, **kwargs):
        return shared

    spec = GeneratorSpec(name="toy.spec_closure", alloc_scratch=alloc, generate=generate,
                         accepts_ctx=True)
    assert len(buffers_of(generate)) == 1
    assert len(buffers_of({"spec": spec})) == 0


def test_buffers_of_a_generator_covers_its_launches():
    gen = TrackGenerator(TrackGenConfig(generator="bezier", device="cpu", num_envs=2),
                         PerEnvSeededRNG(seeds=5, num_envs=2, device="cpu"))
    with trace_launches() as t:
        gen.generate()
    owned = buffers_of(gen, root="gen")
    assert len(t) > 10 and len(owned) > 20
    assert_pointers_within(t, owned)
    assert any(label.startswith("gen._track.") for label in owned.labels())


# ---------------------------------------------------------------------------------------
# capture: the CPU-checkable parts (the recordings themselves need a GPU)
# ---------------------------------------------------------------------------------------

def test_the_gate_raw_anchors_are_named():
    from track_gen import GateGenConfig, GateGenerator

    gen = GateGenerator(GateGenConfig(generator="polar", device="cpu", num_envs=3),
                        PerEnvSeededRNG(seeds=5, num_envs=3, device="cpu"))
    anchors = gen._raw_anchors
    # PR11: the raw output is the anchors channel (polar: 12 knots per env), not a
    # positional scratch slot of max_gates rows.
    assert anchors is gen.program.tap("anchors.xy")
    assert anchors.dtype is wp.vec2f and anchors.shape == (3 * 12,)


def test_poison_is_finite_and_dtype_aware():
    i32, u32, f32 = _arr(3, wp.int32), _arr(3, wp.uint32), _arr(3)
    v3, q = _arr(2, wp.vec3f), _arr(2, wp.quatf)
    tg_capture.poison_all({"i": i32, "u": u32, "f": f32, "v": v3, "q": q, "n": None})
    assert (i32.numpy() == -7).all() and (u32.numpy() == 7).all()
    assert (f32.numpy() == -12345.0).all() and (v3.numpy() == -12345.0).all()
    assert (q.numpy() == -12345.0).all()


def test_result_arrays():
    gen = TrackGenerator(TrackGenConfig(device="cpu", num_envs=2),
                         PerEnvSeededRNG(seeds=5, num_envs=2, device="cpu"))
    arrays = tg_capture.result_arrays(gen.generate())
    assert {"center", "valid", "count", "truncated", "generation"} <= set(arrays)
    assert tg_capture.result_arrays(None) == {}


def test_assert_caller_capturable_needs_a_cuda_producer():
    gen = TrackGenerator(TrackGenConfig(device="cpu", num_envs=2),
                         PerEnvSeededRNG(seeds=5, num_envs=2, device="cpu"))
    with pytest.raises(ValueError, match="CUDA producer"):
        assert_caller_capturable(gen)
    with pytest.raises(ValueError, match="capture mode"):
        assert_caller_capturable(gen, modes=("jax",))


@pytest.mark.cuda
@pytest.mark.parametrize("kind", ["track", "gate"])
def test_assert_caller_capturable_on_a_builtin(kind):
    from track_gen import GateGenConfig, GateGenerator

    rng = PerEnvSeededRNG(seeds=11, num_envs=4, device="cuda:0")
    if kind == "track":
        gen = TrackGenerator(TrackGenConfig(generator="polar", device="cuda:0", num_envs=4),
                             rng)
    else:
        gen = GateGenerator(GateGenConfig(generator="polar", device="cuda:0", num_envs=4,
                                          gate_width=0.05), rng)
    status = assert_caller_capturable(gen)
    assert status["warp"] == "ok"
    assert set(status) == {"warp", "external", "torch"}
    assert gen.graph is None                   # recordings never self-capture


# ---------------------------------------------------------------------------------------
# Roadmap PR13a: attributes and widths (tests/_channel_toys.py)
# ---------------------------------------------------------------------------------------

@pytest.fixture
def channel_toys(plugin_registry):
    from tests import _channel_toys

    _channel_toys.register(plugin_registry)
    return plugin_registry


@pytest.mark.parametrize("name", ["acme.ribbon", "acme.helix_loop", "acme.pinch",
                                  "acme.ribbon_metres"])
def test_channel_toy_generators_conform(name, channel_toys):
    report = check_generator(name, devices=CPU)
    report.raise_for_failures()
    assert not report.result("attributes.written").skipped, str(report)
    carried = report.result("attributes.carried")
    # helix_loop / pinch provide raw.z / raw.half_width (written, carried into the
    # Track's own fields): no exported attribute to check
    assert carried.skipped == (name in ("acme.helix_loop", "acme.pinch")), str(report)
    if name == "acme.ribbon":
        assert report.result("attributes.carried").details["exported"] == sorted(
            ["acme.grip", "acme.lane", "acme.surface_id", "acme.theta", "acme.xy_copy"])
    if name == "acme.helix_loop":                 # raw.z: written, none exported
        assert report.result("attributes.written").passed
    # half_width is an optional Track field the kit checks (shape, padding)
    assert report.result("output.padding").passed


def test_a_generator_that_does_not_write_gen_valid_conforms(channel_toys):
    """``gen.valid`` is an optional write (the binding sets it to 1 once, at build): the
    kit does not poison it for a generator that leaves it out (the HelixLoop example of
    writing-a-generator.rst writes only ``raw.xy`` and ``raw.z``)."""
    from tests import _channel_toys as toys

    class HelixNoValid(toys.HelixLoop):
        name = "acme.helix_novalid"
        writes = {"xy": "raw.xy", "z": "raw.z"}

        def build(self, b, p):
            xy, z = b.write("xy", wp.vec2f, per="raw"), b.write("z")
            ok = b.scratch("ok", wp.int32, (b.E,))    # the kernel's flag, not gen.valid
            salt, seed, n = b.salt("phase"), b.seed, toys.RAW
            height, base, waves = b.param("height"), b.param("base"), float(p.waves)
            r = 1.0 * b.scale

            def enqueue(ctx):
                ctx.launch(toys._circle_attr_k, b.E * n,
                           [seed, salt, n, r, base, height, waves, xy, ok, z])
            return enqueue

    channel_toys.add(HelixNoValid)
    report = check_generator("acme.helix_novalid", devices=CPU)
    report.raise_for_failures()
    assert report.result("output.shape").passed


def test_an_anchors_attribute_generator_conforms(channel_toys):
    report = check_gate_generator("acme.tagged", devices=CPU)
    report.raise_for_failures()
    for check in _ATTRS:
        assert not report.result(check).skipped, str(report)


def test_a_partially_written_raw_attribute_fails_attributes_written(channel_toys):
    report = check_generator("acme.ribbon_partial", devices=CPU,
                             skip=("hot_path", "statelessness", "batch_independence"))
    assert "attributes.written" in report.failed_names, str(report)
    msg = report.result("attributes.written").message
    assert "raw.acme.grip" in msg and "not written" in msg, msg


@pytest.mark.parametrize("name,kw", [
    ("constant", {}),
    ("from_channel", dict(ports={"src": "cl.acme.hw"},
                          upstream={"generator": "acme.ribbon"})),
    ("acme.rescale", {}),
    ("acme.wave", {}),
])
def test_width_stages_conform(name, kw, channel_toys):
    report = check_stage("width", name, devices=CPU, num_envs=4, **kw)
    report.raise_for_failures()
    values = report.result("width.values")
    assert not values.skipped and values.details["min"] >= 0.0, values
    assert [r.name for r in report] == _cpu_names("width")


def test_a_negative_width_fails_width_values(channel_toys):
    report = check_stage("width", "acme.negative", devices=CPU, num_envs=4,
                         skip=("hot_path", "capture"))
    assert "width.values" in report.failed_names, str(report)
    assert "is not a finite value >= 0" in report.result("width.values").message


# ---------------------------------------------------------------------------------------
# Import hygiene
# ---------------------------------------------------------------------------------------

def test_track_gen_does_not_import_the_kit_and_the_kit_does_not_import_torch():
    code = "\n".join([
        "import sys, track_gen",
        "assert 'track_gen.testing' not in sys.modules",
        "import track_gen.testing",
        "assert 'pytest' not in sys.modules       # only the opt-in plugin needs it",
        "import track_gen.testing.pytest_plugin",
        "leaked = sorted(m for m in sys.modules if m == 'torch' or m.startswith('torch.'))",
        "assert not leaked, leaked",
    ])
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
