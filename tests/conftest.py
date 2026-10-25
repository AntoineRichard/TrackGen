"""Shared pytest configuration: Warp init and CUDA gating.

CUDA gating: a test that needs a CUDA device carries the ``cuda`` marker (a module
``pytestmark``, a decorator, or the ``"cuda"`` entry of :data:`DEVS`). It is skipped
when ``wp.is_cuda_available()`` is False — Warp's view, not torch's, so a CPU-only
torch build no longer hides a usable GPU. The CPU CI lane deselects these tests with
``-m "not cuda"``; the GPU lane selects them with ``-m cuda``.

``TRACKGEN_REQUIRE_CUDA=1`` (GPU lane) turns silent skips into failures: the session
errors out at startup when Warp sees no CUDA device, and fails at the end when any
``cuda`` test was skipped or when no ``cuda`` test ran at all.
"""
from __future__ import annotations

# track_gen is installed editable (`pip install -e .`), so `import track_gen` works
# from anywhere — no sys.path manipulation needed, and the package's types module does
# not shadow the stdlib `types` module.
import os

# The library's own suite sees only the built-in generators: several tests parametrize
# over registry.available(...) at collection time, and a third-party plugin installed in
# the environment must not leak into them. Set before any test module is collected (and
# before the registry loads). The plugin tests clear it (monkeypatch.delenv) to exercise
# discovery; export TRACK_GEN_PLUGINS=warn to run the suite with installed plugins.
os.environ.setdefault("TRACK_GEN_PLUGINS", "off")

import pytest

# Warp >= 1.14 requires wp.init() before interop entry points like wp.from_torch;
# tests that call them directly (e.g. test_warp_relax) would otherwise fail with
# 'NoneType' has no attribute 'cuda_devices'. Library code paths init themselves.
import warp as wp

wp.init()

CUDA_AVAILABLE = wp.is_cuda_available()
REQUIRE_CUDA = os.environ.get("TRACKGEN_REQUIRE_CUDA") == "1"

# Device parametrization for kernels tested on both devices:
# ``@pytest.mark.parametrize("dev", DEVS)``. The cuda entry carries the ``cuda``
# marker, so it is reported as skipped without a GPU instead of silently missing.
DEVS = ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]

_SKIP_REASON = "CUDA not available (wp.is_cuda_available() is False)"
_cuda_ran = 0
_cuda_skipped: list[str] = []


@pytest.fixture(autouse=True)
def _isolate_plugin_registry():
    """Every test runs inside ``track_gen.plugin.registry.scoped()``: whatever a test (or
    a helper it calls) registers, overrides or unregisters is undone afterwards, so no
    registration leaks into later tests (or into registry-wide enumerations)."""
    from track_gen.plugin import registry

    with registry.scoped():
        yield


@pytest.fixture
def plugin_registry():
    """The process-global ``track_gen.plugin.registry``, snapshot on entry and restored
    after the test (``registry.scoped()``): add/override/unregister specs freely."""
    from track_gen.plugin import registry

    with registry.scoped():
        yield registry


def override_spec(registry, spec, *, kind=None):
    """Replace an already-registered spec (``replace=True``), asserting the
    ``PluginOverrideWarning`` it emits. Use inside a ``plugin_registry`` scope."""
    from track_gen.plugin import PluginOverrideWarning

    with pytest.warns(PluginOverrideWarning):
        registry.add(spec, kind=kind, replace=True)
    return spec


def pytest_configure(config):
    if REQUIRE_CUDA and not CUDA_AVAILABLE:
        raise pytest.UsageError(
            "TRACKGEN_REQUIRE_CUDA=1 but wp.is_cuda_available() is False")


def pytest_collection_modifyitems(config, items):
    if CUDA_AVAILABLE:
        return
    skip = pytest.mark.skip(reason=_SKIP_REASON)
    for item in items:
        if item.get_closest_marker("cuda") is not None:
            item.add_marker(skip)


def pytest_runtest_logreport(report):
    global _cuda_ran
    if "cuda" not in report.keywords:
        return
    if report.skipped and not hasattr(report, "wasxfail"):
        _cuda_skipped.append(report.nodeid)
    elif report.when == "call":
        _cuda_ran += 1


def _require_cuda_failures() -> list[str]:
    if not REQUIRE_CUDA:
        return []
    problems = [f"skipped cuda test: {nodeid}" for nodeid in _cuda_skipped]
    if _cuda_ran == 0:
        problems.append("no cuda test ran")
    return problems


def pytest_sessionfinish(session, exitstatus):
    if _require_cuda_failures() and session.exitstatus in (
            pytest.ExitCode.OK, pytest.ExitCode.NO_TESTS_COLLECTED):
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    problems = _require_cuda_failures()
    if problems:
        terminalreporter.section("TRACKGEN_REQUIRE_CUDA=1", red=True)
        for line in problems:
            terminalreporter.line(line, red=True)
