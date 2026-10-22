"""Conformance kit for generator and pipeline-stage plugins (and TrackGen's own).

A plugin author runs one call in their own CI and learns whether their generator (or
relax / width / elevation / validity stage) honours the contract::

    from track_gen.testing import check_generator, check_stage

    report = check_generator("acme.superformula", params={"m": 7.0})  # cpu (+ cuda:0)
    report.raise_for_failures()
    check_stage("relax", "acme.taubin").raise_for_failures()

or, in pytest, loads the opt-in plugin (``pytest_plugins =
["track_gen.testing.pytest_plugin"]``) and calls the ``track_gen_conformance`` fixture.
TrackGen's own suite runs the same checks over every registered generator.

Modules:

- :mod:`~track_gen.testing.conformance`: :func:`check_generator`,
  :func:`check_gate_generator`, :func:`check_stage`, :class:`Report`,
  :class:`CheckResult`, :data:`CHECKS`;
- :mod:`~track_gen.testing.contract`: hot-path guards (:func:`hot_path_guard`,
  :func:`no_host_sync`, :func:`no_allocation`, :func:`stream_explicit`; each returns
  a :class:`HotPathGuard`, whose breaches are :class:`Violation` s located by a
  :class:`Site`);
- :mod:`~track_gen.testing.trace`: :func:`trace_launches` / :class:`LaunchTrace`
  (launch topology and buffer ownership);
- :mod:`~track_gen.testing.capture`: :func:`assert_caller_capturable` and the capture
  helpers (CUDA);
- :mod:`~track_gen.testing.pytest_plugin`: the opt-in pytest plugin.

``import track_gen`` does not import this package, and importing it does not import
torch (only the torch capture modes do, inside the mode).
"""
from .capture import CAPTURE_MODES, assert_caller_capturable
from .conformance import (
    CHECKS,
    CheckResult,
    Report,
    check_gate_generator,
    check_generator,
    check_stage,
)
from .contract import (
    CATEGORIES,
    HotPathGuard,
    HotPathViolation,
    Site,
    Violation,
    hot_path_guard,
    no_allocation,
    no_host_sync,
    stream_explicit,
)
from .trace import (
    BufferSet,
    LaunchRecord,
    LaunchTrace,
    assert_pointers_within,
    assert_same_topology,
    buffers_of,
    trace_launches,
)

__all__ = [
    "BufferSet",
    "CAPTURE_MODES",
    "CATEGORIES",
    "CHECKS",
    "CheckResult",
    "HotPathGuard",
    "HotPathViolation",
    "LaunchRecord",
    "LaunchTrace",
    "Report",
    "Site",
    "Violation",
    "assert_caller_capturable",
    "assert_pointers_within",
    "assert_same_topology",
    "buffers_of",
    "check_gate_generator",
    "check_generator",
    "check_stage",
    "hot_path_guard",
    "no_allocation",
    "no_host_sync",
    "stream_explicit",
    "trace_launches",
]
