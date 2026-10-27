"""Simple, context-aware track generator plugin specification."""
from __future__ import annotations

import dataclasses
from typing import Callable, ClassVar


@dataclasses.dataclass(frozen=True)
class GeneratorSpec:
    """A simple track generator whose device work goes through ``ctx``.

    name:          the ``config.generator`` string that selects it. Third-party names
                   should be dotted (``"package.name"``).
    alloc_scratch: ``(config) -> scratch`` — allocate this generator's PRIVATE working
                   buffers ONCE (fixed shapes from config, on config.device). The
                   generation OUTPUT buffers (centerline, valid) are owned by the
                   orchestrator and passed to ``generate``; they are NOT part of this.
    generate:      ``(seeds_wp, config, out_centerline, out_valid_wp, scratch, *, ctx) -> None`` —
                   write the closed centerline ([E*num_points] vec2f) into out_centerline
                   and per-env validity ([E] int32) into out_valid_wp, using scratch.
                   Every launch must use the supplied LaunchContext ``ctx``. Pure Warp,
                   in-place, zero-alloc, no host sync WHEN ``capturable=True``.
    capturable:    whether ``generate`` obeys the CUDA-graph-capture contract (pure Warp,
                   no host sync, no per-call allocation). Defaults ``True`` so existing
                   registrations stay on the capture-then-replay
                   CUDA path. A ``capturable=False`` generator may use host-side loop control
                   and per-call allocation; ``TrackGenerator`` runs it EAGERLY on CUDA every
                   call (the same code path as ``cpu``), never building a ``wp.Graph`` for it.
    params:        optional :class:`track_gen.plugin.Params` subclass (STATIC fields
                   only) declaring this generator's own parameters. Users set them with
                   ``TrackGenConfig(generator=name, generator_params=...)`` (an instance
                   or a mapping of its fields; ``None`` resolves
                   ``params.from_legacy(config)``). ``generate`` / ``alloc_scratch`` read
                   the resolved instance as ``config.generator_params``, and
                   ``registry.describe`` reports its JSON schema. ``None`` (the default):
                   the generator takes no ``generator_params``.
    """

    #: Registry kind ``track_gen.plugin.registry.add(spec)`` infers for this spec type.
    registry_kind: ClassVar[str] = "generator"

    name: str
    alloc_scratch: Callable
    generate: Callable
    capturable: bool = True
    params: "type | None" = None

