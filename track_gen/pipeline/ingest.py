"""Host loader of the ingestion inputs: :func:`load_centerlines`.

Loads numpy centerlines into the ``centerline.*`` inputs of any generator that declares
them (``generator/external``, or a third-party ingestion generator built with
``StageBuilder.input(..., per="raw")`` under the same names), with host checks and the
sizing check, on the stream resolved like ``generate``'s (``stream=``, an ambient
``LaunchContext.scope``, the generator's constructor stream, torch's capture stream,
Warp's current stream). Cold: it allocates host staging and blocks;
call it outside the hot path (in-graph ingestion writes ``program.inputs`` directly, on
the generation's stream).
"""
from __future__ import annotations

import math
import numbers
from collections.abc import Mapping, Sequence

import numpy as np
import warp as wp

__all__ = ["load_centerlines"]

_PREFIX = "centerline."
_NP = {wp.float32: np.float32, wp.int32: np.int32, wp.vec2f: np.float32,
       wp.vec3f: np.float32}
_WIDTH = {wp.float32: 1, wp.int32: 1, wp.vec2f: 2, wp.vec3f: 3}


def _program_of(target):
    from .program import Program  # noqa: PLC0415

    program = target if isinstance(target, Program) else getattr(target, "program", None)
    if not isinstance(program, Program):
        raise TypeError(f"load_centerlines: target must be a TrackGenerator, a "
                        f"GateGenerator or a Program, got {type(target).__name__}")
    return program


def _envs(envs, E: int) -> "list[int]":
    if envs is None:
        return list(range(E))
    if isinstance(envs, (str, bytes)) or not isinstance(envs, Sequence) and \
            not isinstance(envs, np.ndarray):
        raise TypeError(f"load_centerlines: envs must be a sequence of env indices, got "
                        f"{type(envs).__name__}")
    out = []
    for e in list(envs):
        if isinstance(e, bool) or not isinstance(e, numbers.Integral) or not 0 <= e < E:
            raise ValueError(f"load_centerlines: envs holds {e!r}; env indices are ints "
                             f"in [0, {E})")
        out.append(int(e))
    if len(set(out)) != len(out) or not out:
        raise ValueError(f"load_centerlines: envs must be distinct and non-empty, got "
                         f"{list(envs)}")
    return out


def _rows(name: str, data, n_env: int, N: int, width: int) -> "list[np.ndarray]":
    """``data`` as ``n_env`` float64 rows ``[m_e, width]`` (``[m_e]`` for width 1):
    ``[n_env, m, width]``, ``[n_env * N, width]`` or a sequence of ``n_env`` arrays."""
    shape_pt = () if width == 1 else (width,)
    if isinstance(data, np.ndarray) or (hasattr(data, "shape") and hasattr(data, "dtype")
                                        and not isinstance(data, wp.array)):
        a = np.asarray(data)
        if a.ndim == 2 + len(shape_pt) and tuple(a.shape[2:]) == shape_pt \
                and a.shape[0] == n_env:
            return [a[e] for e in range(n_env)]
        if a.ndim == 1 + len(shape_pt) and tuple(a.shape[1:]) == shape_pt \
                and a.shape[0] == n_env * N:
            return [a[e * N:(e + 1) * N] for e in range(n_env)]
        raise ValueError(f"load_centerlines: {name} has shape {a.shape}; expected "
                         f"[{n_env}, m{', ' + str(width) if width > 1 else ''}], "
                         f"[{n_env}*{N}{', ' + str(width) if width > 1 else ''}] or a "
                         f"sequence of {n_env} rows")
    if isinstance(data, wp.array):
        return _rows(name, data.numpy(), n_env, N, width)
    if isinstance(data, (str, bytes)) or not isinstance(data, Sequence):
        raise TypeError(f"load_centerlines: {name} must be a numpy array or a sequence of "
                        f"per-env arrays, got {type(data).__name__}")
    if len(data) != n_env:
        raise ValueError(f"load_centerlines: {name} holds {len(data)} rows; {n_env} envs "
                         f"are loaded")
    out = []
    for e, r in enumerate(data):
        a = np.asarray(r)
        if a.ndim != 1 + len(shape_pt) or tuple(a.shape[1:]) != shape_pt:
            raise ValueError(f"load_centerlines: {name}[{e}] has shape {a.shape}; expected "
                             f"[m{', ' + str(width) if width > 1 else ''}]")
        out.append(a)
    return out


def _check_values(label: str, rows, envs, dtype=None) -> None:
    """Each row's NumPy dtype fits the input: real numbers (``xy`` / ``z`` /
    ``half_width``, a float attribute), integers within int32 (an ``int32`` attribute:
    a float would be truncated)."""
    for e, r in zip(envs, rows):
        kind = r.dtype.kind
        if dtype is wp.int32:
            if kind not in "iu":
                raise ValueError(f"load_centerlines: env {e}: {label} is an int32 "
                                 f"attribute; got {r.dtype} values (pass integers)")
            info = np.iinfo(np.int32)
            if r.size and (int(r.min()) < info.min or int(r.max()) > info.max):
                raise ValueError(f"load_centerlines: env {e}: {label} is an int32 "
                                 f"attribute; its values exceed the int32 range")
        elif kind not in "iuf":
            what = f"a {getattr(dtype, '__name__', dtype)} attribute" \
                if dtype is not None else "real numbers"
            raise ValueError(f"load_centerlines: env {e}: {label} holds {r.dtype} values; "
                             f"expected {what}")


def _sizing(program) -> "tuple[float, int] | None":
    """``(spacing, n_max)`` of the Program's ``resample.constant_spacing`` (``None``
    without one: no sizing check)."""
    for ref in program.spec.stages:
        if (ref.kind, ref.name) == ("resample", "constant_spacing"):
            spacing = getattr(ref.params, "spacing", None)
            if spacing is None and isinstance(ref.params, Mapping):
                spacing = ref.params.get("spacing")
            if spacing is None:
                return None
            return float(spacing), int(program.spec.batch.n_max)
    return None


def _perimeter(xy: np.ndarray, z: "np.ndarray | None") -> float:
    p = xy.astype(np.float64)
    if z is not None:
        p = np.concatenate([p, z.astype(np.float64)[:, None]], axis=1)
    d = np.diff(np.concatenate([p, p[:1]], axis=0), axis=0)
    return float(np.sqrt((d * d).sum(axis=1)).sum())


def load_centerlines(target, xy, *, count=None, z=None, half_width=None,
                     attributes=None, envs=None, stream=None) -> None:
    """Load centerlines into a generator's ``centerline.*`` inputs (cold, blocking).

    The rows reach the device inputs the generator reads (``program.inputs``): every
    generation after this call (on the generator's stream) ingests them; a masked
    regeneration (``generate(advance=True, mask=m)``) rebuilds only the masked envs, so
    load new rows for those envs (``envs=``) before it.

    Args:
        target: A ``TrackGenerator``, ``GateGenerator`` or ``Program`` whose generator
            declares the ``centerline.*`` inputs (``generator="external"``).
        xy: The plan-view points of each env: ``[E', m, 2]`` (every row ``m`` points),
            ``[E' * N, 2]`` (``N`` = the raw capacity, ``num_points``), or a sequence of
            ``E'`` arrays ``[m_e, 2]`` (``E'``: the number of loaded envs). A closed
            loop: the last point connects to the first (do not repeat it).
        count: Optional ``[E']`` real point counts (default: each row's length); a row
            longer than its count is cut.
        z: The altitude of each point, as ``xy`` without its last axis: required when
            the generator was built ``with_z``, refused otherwise.
        half_width: The half-width of each point, alike; ``> 0``. In the data's units,
            like the pipeline's ``spacing``. Required with ``with_half_width``.
        attributes: ``{key: rows}`` for the ingested attributes (``(key, dtype,
            interp)`` of ``ExternalParams.attributes``): every one is required, with
            values of its kind (integers for ``int32``, real numbers otherwise).
        envs: The env indices of the rows (default: every env, in order).
        stream: The stream of the upload. Default: resolved like ``generate``'s (an
            ambient ``LaunchContext.scope``, the generator's constructor ``stream=``,
            torch's capture stream, Warp's current stream, in that order), so it is
            ordered after in-flight generations on that stream. Inside a scope on
            another stream than the generator's, or after a ``generate(stream=t)``,
            pass ``stream=`` (or order the streams with events).

    The sizing check (with a ``resample.constant_spacing`` stage): each env's closed
    perimeter ``L`` (3D with ``z``) must need ``floor(L / spacing) + 1 <= n_max``
    points at the pipeline's spacing, and at least 3: set ``spacing`` / ``n_max``
    (``presets.track(spacing=..., n_max=...)``), or rescale the data. Exact without
    relax (the ``external`` default); a close pre-check with relax on (the device
    ``truncated`` bit stays authoritative).

    Raises:
        TypeError: a target without a Program; an input of the wrong type.
        ValueError: a generator without ``centerline.xy``; a shape / env-count mismatch;
            ``count`` outside ``[3, N]``; a non-finite point, altitude or half-width; a
            half-width ``<= 0``; values of the wrong kind (non-numeric data; non-integer
            or out-of-range data for an ``int32`` attribute); an input the generator was
            not built with; an input it WAS built with left out (``z=`` for
            ``with_z``, ``half_width=`` for ``with_half_width``, each ingested
            attribute: every per-point input is loaded with the rows); an unknown
            attribute; the sizing check.
        track_gen.exec.CaptureError: called inside a capture (before any work).
    """
    from .._src.types import _refuse_capture  # noqa: PLC0415
    from ..exec import _compat  # noqa: PLC0415
    from ..exec.context import LaunchContext  # noqa: PLC0415

    program = _program_of(target)
    info = program.input_info
    inputs = program.inputs
    name = program._binding.name
    if _PREFIX + "xy" not in info:
        raise ValueError(f"load_centerlines: this generator has no 'centerline.xy' input: "
                         f"its generator is {name!r} (ingestion: generator='external')")
    E = int(program.layout.num_envs)
    N = int(info[_PREFIX + "xy"].length) // E
    dev = wp.get_device(program.layout.device)
    # Refuse a capture before any work (the checks below may read device arrays back).
    ctx = LaunchContext.for_call(dev, stream, default=program.runner.default_stream)
    _refuse_capture(dev, (ctx,), "load_centerlines()", message=(
        "load_centerlines() stages host data and blocks: call it outside any capture (in "
        "a graph, write program.inputs['centerline.*'] directly, on the generation's "
        "stream)"))
    rows_envs = _envs(envs, E)
    n_env = len(rows_envs)
    xy_rows = _rows("xy", xy, n_env, N, 2)
    _check_values("xy", xy_rows, rows_envs)
    if count is None:
        counts = [int(r.shape[0]) for r in xy_rows]
    else:
        c = np.asarray(count)
        if c.shape != (n_env,) or not np.issubdtype(c.dtype, np.integer):
            raise ValueError(f"load_centerlines: count must be [{n_env}] ints, got "
                             f"{c.shape} {c.dtype}")
        counts = [int(v) for v in c]
    for k, (e, m) in enumerate(zip(rows_envs, counts)):
        if not 3 <= m <= N:
            raise ValueError(f"load_centerlines: env {e}: {m} points; a row holds 3 to "
                             f"{N} points (the raw capacity num_points={N})")
        if xy_rows[k].shape[0] < m:
            raise ValueError(f"load_centerlines: env {e}: count {m} but the row holds "
                             f"{xy_rows[k].shape[0]} points")
        if not np.isfinite(xy_rows[k][:m]).all():
            bad = int(np.flatnonzero(~np.isfinite(xy_rows[k][:m]).all(axis=1))[0])
            raise ValueError(f"load_centerlines: env {e}: point {bad} is not finite")
    channels = {}                             # input name -> rows
    for label, data in (("z", z), ("half_width", half_width)):
        if data is None:
            continue
        if _PREFIX + label not in info:
            flag = "with_z" if label == "z" else "with_half_width"
            raise ValueError(f"load_centerlines: {label}= given, but generator {name!r} "
                             f"was built without it (generator_params={{{flag!r}: True}})")
        rows = _rows(label, data, n_env, N, 1)
        _check_values(label, rows, rows_envs)
        for k, (e, m) in enumerate(zip(rows_envs, counts)):
            r = rows[k]
            if r.shape[0] < m:
                raise ValueError(f"load_centerlines: env {e}: {label} holds {r.shape[0]} "
                                 f"values for {m} points")
            if not np.isfinite(r[:m]).all():
                raise ValueError(f"load_centerlines: env {e}: {label} is not finite")
            if label == "half_width" and not (r[:m] > 0.0).all():
                raise ValueError(f"load_centerlines: env {e}: half_width must be > 0")
        channels[_PREFIX + label] = rows
    for key, data in dict(attributes or {}).items():
        iname = _PREFIX + str(key)
        if iname not in info or key in ("xy", "count", "z", "half_width"):
            known = sorted(k[len(_PREFIX):] for k in info if k.startswith(_PREFIX)
                           and k[len(_PREFIX):] not in ("xy", "count", "z", "half_width"))
            raise ValueError(f"load_centerlines: generator {name!r} has no attribute "
                             f"{key!r}; its attributes: {known}")
        dtype = info[iname].dtype
        rows = _rows(f"attributes[{key!r}]", data, n_env, N, _WIDTH[dtype])
        _check_values(f"attributes[{key!r}]", rows, rows_envs, dtype)
        for k, (e, m) in enumerate(zip(rows_envs, counts)):
            if rows[k].shape[0] < m:
                raise ValueError(f"load_centerlines: env {e}: attributes[{key!r}] holds "
                                 f"{rows[k].shape[0]} values for {m} points")
        channels[iname] = rows
    # Every per-point input the generator declares must be given: an omitted one would
    # keep the previous rows (or the placeholders) under the new xy.
    for iname in sorted(k for k, i in info.items() if k.startswith(_PREFIX)
                        and i.per == "raw" and k != _PREFIX + "xy"):
        if iname in channels:
            continue
        label = iname[len(_PREFIX):]
        if label in ("z", "half_width"):
            flag = "with_z" if label == "z" else "with_half_width"
            raise ValueError(f"load_centerlines: generator {name!r} was built {flag}: pass "
                             f"{label}= for the loaded rows (otherwise the input keeps the "
                             f"previous rows)")
        raise ValueError(f"load_centerlines: generator {name!r} was built with attribute "
                         f"{label!r}: pass attributes={{{label!r}: ...}} for the loaded "
                         f"rows (otherwise the input keeps the previous rows)")
    sizing = _sizing(program)
    if sizing is not None:
        spacing, n_max = sizing
        zr = channels.get(_PREFIX + "z")
        for k, (e, m) in enumerate(zip(rows_envs, counts)):
            L = _perimeter(xy_rows[k][:m], None if zr is None else zr[k][:m])
            need = math.floor(L / spacing) + 1
            if need > n_max:
                raise ValueError(f"load_centerlines: env {e}: perimeter {L:.1f} needs "
                                 f"{need} points at spacing {spacing:g} > n_max {n_max}; "
                                 f"set spacing / n_max, or rescale the data (half_width "
                                 f"and spacing are in the data's units)")
            if need < 3:
                raise ValueError(f"load_centerlines: env {e}: perimeter {L:.4g} is too "
                                 f"short for spacing {spacing:g} (fewer than 3 points); "
                                 f"set spacing, or rescale the data")
    # Host staging (pinned on CUDA), one row per loaded env, padded with the row's last
    # real point (the device pads past count[e] alike).
    uploads = [(inputs[_PREFIX + "xy"], wp.vec2f, xy_rows)]
    for iname, rows in channels.items():
        uploads.append((inputs[iname], info[iname].dtype, rows))
    staged = []
    for dst, dtype, rows in uploads:
        width = _WIDTH[dtype]
        host = np.empty((n_env, N) + ((width,) if width > 1 else ()), dtype=_NP[dtype])
        for k, m in enumerate(counts):
            host[k, :m] = rows[k][:m]
            host[k, m:] = rows[k][m - 1]
        flat = host.reshape((n_env * N,) + ((width,) if width > 1 else ()))
        staged.append((dst, wp.array(flat, dtype=dtype, device="cpu",
                                     pinned=dev.is_cuda), N))
    cnt = np.asarray(counts, dtype=np.int32)
    staged.append((inputs[_PREFIX + "count"],
                   wp.array(cnt, dtype=wp.int32, device="cpu", pinned=dev.is_cuda), 1))
    whole = rows_envs == list(range(E))
    for dst, src, n in staged:
        if whole:
            ctx.copy(dst, src)
        else:
            for k, e in enumerate(rows_envs):
                ctx.copy(dst, src, dst_offset=e * n, src_offset=k * n, count=n)
    if ctx.stream is not None:
        _compat.stream_synchronize(ctx.stream)    # the staging must outlive the copies
