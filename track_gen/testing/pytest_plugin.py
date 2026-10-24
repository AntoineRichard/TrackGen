"""Opt-in pytest plugin for the conformance kit.

Not registered as a ``pytest11`` entry point (it would load into every pytest run of
every environment that installs track_gen). Load it from your top-level ``conftest.py``
(or a test module)::

    pytest_plugins = ["track_gen.testing.pytest_plugin"]

It provides:

- the ``track_gen_conformance`` fixture: a factory
  ``track_gen_conformance(name, params=None, *, kind="generator", **kwargs) -> Report``
  that runs :func:`~track_gen.testing.check_generator` (``kind="gate_generator"``:
  :func:`~track_gen.testing.check_gate_generator`; ``"relax"`` / ``"width"`` /
  ``"elevation"`` / ``"validity"`` / ``"order"`` / ``"stations"``:
  :func:`~track_gen.testing.check_stage`) and fails
  the test, listing every failed check, unless all pass;
- the ``track_gen_check_component`` fixture: a factory
  ``track_gen_check_component(factory, **kwargs) -> Report`` that runs
  :func:`~track_gen.testing.check_component` and fails the test, listing every failed
  check, unless all pass;
- :func:`generator_names` and :func:`parametrize_generators`, to run a test over the
  registered generators::

      from track_gen.testing.pytest_plugin import parametrize_generators

      @parametrize_generators("generator", builtin=False)
      def test_my_plugins_conform(generator, track_gen_conformance):
          track_gen_conformance(generator)
"""
from __future__ import annotations

from typing import Iterable

import pytest

from .components import check_component
from .conformance import Report, check_gate_generator, check_generator, check_stage

__all__ = ["generator_names", "parametrize_generators", "track_gen_check_component",
           "track_gen_conformance"]

_KINDS = ("generator", "gate_generator", "relax", "width", "elevation", "validity",
          "order", "stations")


def _check_kind(kind: str) -> None:
    if kind not in _KINDS:
        raise ValueError(f"kind must be one of {_KINDS}, got {kind!r}")


def generator_names(kind: str = "generator", *, builtin: "bool | None" = None,
                    exclude: Iterable[str] = ()) -> list[str]:
    """The names registered under ``kind`` (``"generator"``, ``"gate_generator"``,
    ``"relax"``, ``"width"``, ``"elevation"``, ``"validity"``, ``"order"`` or
    ``"stations"``), sorted;
    ``builtin=True`` / ``False`` keeps only the built-ins / only the others."""
    from ..plugin import registry  # noqa: PLC0415

    _check_kind(kind)
    skip = set(exclude)
    names = [n for n in registry.available(kind) if n not in skip]
    if builtin is not None:
        names = [n for n in names if registry.describe(kind, n).builtin == builtin]
    return sorted(names)


def parametrize_generators(kind: str = "generator", *, argname: str = "generator",
                           builtin: "bool | None" = None, exclude: Iterable[str] = ()):
    """``pytest.mark.parametrize(argname, generator_names(kind, ...))``, evaluated at
    collection time (so the registry, and the plugins its entry points name, load
    then)."""
    return pytest.mark.parametrize(argname, generator_names(kind, builtin=builtin,
                                                            exclude=exclude))


@pytest.fixture
def track_gen_conformance():
    """Factory: ``track_gen_conformance(name, params=None, *, kind="generator",
    **kwargs)`` runs the conformance checks (``kwargs`` as
    :func:`~track_gen.testing.check_generator`, or
    :func:`~track_gen.testing.check_stage` for a pipeline-stage kind), fails the test
    when one fails, and returns the :class:`~track_gen.testing.Report`."""

    def run(name: str, params=None, *, kind: str = "generator", **kwargs) -> Report:
        _check_kind(kind)
        if kind == "generator":
            report = check_generator(name, params, **kwargs)
        elif kind == "gate_generator":
            report = check_gate_generator(name, params, **kwargs)
        else:
            report = check_stage(kind, name, params, **kwargs)
        if not report.ok:
            lines = "\n".join(f"  {r}" for r in report.failures)
            pytest.fail(f"{kind.replace('_', ' ')} {name!r} failed "
                        f"{len(report.failures)} conformance check(s):\n{lines}",
                        pytrace=False)
        return report

    return run


@pytest.fixture
def track_gen_check_component():
    """Factory: ``track_gen_check_component(factory, **kwargs)`` runs
    :func:`~track_gen.testing.check_component` (``kwargs`` as there: ``source``,
    ``upstream``, ``devices``, ...), fails the test when a check fails, and returns the
    :class:`~track_gen.testing.Report`."""

    def run(factory, **kwargs) -> Report:
        report = check_component(factory, **kwargs)
        if not report.ok:
            lines = "\n".join(f"  {r}" for r in report.failures)
            pytest.fail(f"component {report.name!r} failed {len(report.failures)} "
                        f"check(s):\n{lines}", pytrace=False)
        return report

    return run
