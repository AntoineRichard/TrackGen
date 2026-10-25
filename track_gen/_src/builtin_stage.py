"""Shared static views, parameter validation and scratch allocation for built-in Stages."""
from __future__ import annotations

import dataclasses
import types

import warp as wp


def run_view(params, b) -> types.SimpleNamespace:
    """The ``config`` a built-in generator's private core reads in its Stage: every field
    of the static view ``params`` (a dynamic name holds the raising sentinel of
    :func:`track_gen.plugin.params._static_view`) plus ``num_envs``, ``num_points`` (the
    raw capacity), ``scale`` and ``device`` from the builder."""
    view = types.SimpleNamespace(num_envs=int(b.E), num_points=int(b.raw),
                                 scale=float(b.scale), device=str(b.device))
    for f in dataclasses.fields(params):
        setattr(view, f.name, getattr(params, f.name))
    return view


def check_dyn(dyn, names, what: str) -> None:
    """``dyn`` must be ``None`` or hold exactly the dynamic fields ``names``."""
    if dyn is None:
        return
    keys, want = set(dyn), set(names)
    if keys != want:
        raise ValueError(f"{what}: dyn must map exactly the dynamic fields {sorted(want)} "
                         f"to their [E] arrays (or be None); missing "
                         f"{sorted(want - keys)}, extra {sorted(map(str, keys - want))}")




def default_alloc(config):
    """Allocate private scratch with ``wp.empty`` on ``config.device``."""
    dev = str(config.device)

    def alloc(name, dtype, n):
        return wp.empty(n, dtype=dtype, device=dev)
    return alloc
