"""Public device primitives for generator authors (``track_gen.ops``).

These are the building blocks the built-in generators are made of, so a plugin
generator can be assembled from the same documented pieces instead of private
``track_gen._src`` helpers. Conventions shared by every op:

* ``op(ctx, ...)``: ``ctx`` is a :class:`track_gen.exec.LaunchContext` (anything with
  ``.launch`` and ``.device`` works; the device is canonicalised, so ``"cuda"`` matches
  arrays on the current CUDA device). Every launch goes through ``ctx.launch``, so
  the work lands on the context's stream and can be recorded into a caller's CUDA
  graph. An op never calls ``ctx.sync()``, never reads device data back to the host
  and never allocates device memory: working buffers are passed in (e.g. an
  :class:`ArcScratch` allocated once in ``alloc_scratch``).
* Flat per-env layout: env ``e`` owns ``[e*stride, (e+1)*stride)`` of a flat array of
  ``E*stride`` elements; per-env scalars are ``[E]`` arrays.
* Argument names: ``src`` is the input point array and ``stride`` its points per
  env; ``num`` is an exact number of output points per env and ``out_stride`` an
  output row capacity (NaN-padded past the env's count); outputs are ``out`` (plus
  ``count_out``/``truncated_out`` for per-env results).
* ``E`` is ``src.shape[0] // stride`` (which must divide exactly) for every op that
  takes ``src``, and ``seeds.shape[0]`` for the samplers; every other per-env array
  must then be exactly ``[E]``.
* Count semantics: ``count`` is always keyword-only. Where it is given, only the first
  ``count[e]`` points of a row are real (a count above the stride is clamped to it);
  slots past it are written ``(nan, nan)`` by ops that produce points. Masks are the
  keyword-only ``active`` (``active[e] > 0`` selects the env).
* Validation is host-side and happens before any launch: data arrays must match their
  exact shape, dtype and ``ctx.device``; scratch arrays need the dtype, the device
  and at least the stated length. Mismatches raise ``ValueError`` naming the argument,
  except that ``fill`` raises ``TypeError`` for an unsupported dtype and ``any_set`` for
  an argument that is not an ``int32`` ``wp.array``.

Device functions for your own kernels live in :mod:`track_gen.ops.funcs`; shared
numeric conventions in :mod:`track_gen.ops.constants`.
"""
from . import constants, funcs
from .check import check_array
from .curves import closed_catmull_rom, closed_polyline
from .fill import fill
from .geometry import angle_sort, normalize_extent, self_intersections, thickness, turning_number
from .hash import hash32, hash32_host
from .reduce import any_set
from .resample import ArcScratch, arc_resample, resample_constant_spacing, resample_uniform
from .sampling import grid_cells, sample_counts, sample_grid_points

__all__ = [
    "ArcScratch",
    "angle_sort",
    "any_set",
    "arc_resample",
    "check_array",
    "closed_catmull_rom",
    "closed_polyline",
    "constants",
    "fill",
    "funcs",
    "grid_cells",
    "hash32",
    "hash32_host",
    "normalize_extent",
    "resample_constant_spacing",
    "resample_uniform",
    "sample_counts",
    "sample_grid_points",
    "self_intersections",
    "thickness",
    "turning_number",
]
