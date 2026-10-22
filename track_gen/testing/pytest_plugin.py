"""Opt-in pytest plugin for the conformance kit.

Not registered as a ``pytest11`` entry point (it would load into every pytest run of
every environment that installs track_gen). Load it from your top-level ``conftest.py``
(or a test module)::

    pytest_plugins = ["track_gen.testing.pytest_plugin"]

It provides:

- the ``track_gen_conformance`` fixture: a factory
  ``track_gen_conformance(name, params=None, *, kind="generator", **kwargs) -> Report``
  that runs :func:`~track_gen.testing.check_generator` (``kind="gate_generator"``:
  :func:`~track_gen.testing.check_gate_generator`) and fails the test, listing every
  failed check, unless all pass;
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

from .conformance import Report, check_gate_generator, check_generator

__all__ = ["generator_names", "parametrize_generators", "track_gen_conformance"]

_KINDS = ("generator", "gate_generator")


def _check_kind(kind: str) -> None:
    if kind not in _KINDS:
        raise ValueError(f"kind must be one of {_KINDS}, got {kind!r}")


def generator_names(kind: str = "generator", *, builtin: "bool | None" = None,
                    exclude: Iterable[str] = ()) -> list[str]:
    """The names registered under ``kind`` (``"generator"`` or ``"gate_generator"``),
    sorted; ``builtin=True`` / ``False`` keeps only the built-ins / only the others."""
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
    :func:`~track_gen.testing.check_generator`), fails the test when one fails, and
    returns the :class:`~track_gen.testing.Report`."""

    def run(name: str, params=None, *, kind: str = "generator", **kwargs) -> Report:
        _check_kind(kind)
        check = check_generator if kind == "generator" else check_gate_generator
        report = check(name, params, **kwargs)
        if not report.ok:
            lines = "\n".join(f"  {r}" for r in report.failures)
            pytest.fail(f"{kind.replace('_', ' ')} {name!r} failed "
                        f"{len(report.failures)} conformance check(s):\n{lines}",
                        pytrace=False)
        return report

    return run
