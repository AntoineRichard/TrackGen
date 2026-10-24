"""Shared pieces of the built-in track generator Stages (roadmap PR13c).

``bezier``, ``hull``, ``polar``, ``voronoi`` and ``checkpoint`` are ``kind="generator"``
Stages (like ``repulsive`` since PR12). Each module keeps its frozen legacy entry points
(``generate_*_warp(seeds_wp, config, out_centerline, out_valid_wp, scratch, ctx=None)``
and ``*_alloc_scratch(config)``) on a private core shared with its Stage:
``_enqueue_<gen>(seeds, config, xy, valid, scratch, ctx, dyn)`` and
``_alloc_<gen>_scratch(config, alloc=None)``. ``dyn`` is ``None`` (the legacy path: host
scalars from ``config``, the scalar kernels) or exactly the generator's dynamic fields
mapped to their ``[E]`` float32 arrays (the Stage: ``b.param(field)``, the
array-reading ``_dyn`` twins).
"""
from __future__ import annotations

import dataclasses
import types
import warnings

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
    """``alloc(name, dtype, n)`` of the legacy ``*_alloc_scratch(config)``:
    ``wp.empty(n, dtype=dtype, device=config.device)``."""
    dev = str(config.device)

    def alloc(name, dtype, n):
        return wp.empty(n, dtype=dtype, device=dev)
    return alloc


class DeprecatedParamsAlias:
    """``registry.get("generator", <built-in>).params``: the 0.2.0.dev4 accessor of a
    built-in track generator's Params class (its ``GeneratorSpec.params``). Warns
    ``DeprecationWarning`` (attributed to the caller) and returns ``.Params``; removed in
    1.0 (roadmap PR16)."""

    def __get__(self, obj, owner=None):
        from .generator_binding import _outside_track_gen_stacklevel  # noqa: PLC0415

        owner = type(obj) if owner is None else owner
        warnings.warn(f"registry.get('generator', {owner.name!r}).params is deprecated: "
                      f"use .Params (removed in 1.0)", DeprecationWarning,
                      stacklevel=_outside_track_gen_stacklevel())
        return owner.Params
