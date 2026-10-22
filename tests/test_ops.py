"""Public ``track_gen.ops`` primitives (CPU; roadmap PR7).

Per op: host-side validation (raised before any launch), hand-built semantics (shapes,
NaN padding, count semantics), BITWISE equivalence to the frozen HEAD-1077b18 kernel it
replaces (``tests/_legacy_kernels_pr7.py``), and zero allocation during the call.
Stream-explicitness and no-host-sync are covered by the ``ops[all]`` driver in
``tests/test_no_host_sync.py`` (shared with ``tests/test_launch_stream.py``).
"""
from __future__ import annotations

import ast
import contextlib
import functools
import importlib
import inspect
import math
import pathlib
import subprocess
import sys

import numpy as np
import pytest
import warp as wp

from tests import _legacy_kernels_pr7 as L
from track_gen import ops
from track_gen.exec import LaunchContext
from track_gen.ops import funcs
from track_gen.ops.geometry import (
    _angle_sort_launch,
    _self_intersections_func as _ops_selfx_func,
    _thickness_func as _ops_thickness_func,
    _turning_func as _ops_turning_func,
)

DEV = "cpu"
_INT32_OVERFLOW_SEEDS = [0, 1, 7, 99991, 2147483647, 1000003, 123456789, 46341, -5,
                         -2147483648]


@pytest.fixture
def ctx():
    return LaunchContext(DEV, None)


class _RecCtx:
    """A LaunchContext stand-in that records launches (``device`` is configurable so a
    device mismatch can be simulated on a CPU-only host)."""

    def __init__(self, device=DEV):
        self.device = device
        self.launches = []
        self._real = LaunchContext(DEV, None)

    def launch(self, kernel, dim, inputs=(), outputs=(), **kw):
        self.launches.append(kernel.key)
        self._real.launch(kernel, dim, inputs, outputs, **kw)


# ---------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------

def v2(a):
    return wp.array(np.asarray(a, np.float32).reshape(-1, 2), dtype=wp.vec2f, device=DEV)


def i32(a):
    return wp.array(np.asarray(a, np.int32).reshape(-1), dtype=wp.int32, device=DEV)


def f32(a):
    return wp.array(np.asarray(a, np.float32).reshape(-1), dtype=wp.float32, device=DEV)


def ev2(n, fill=None):
    if fill is None:
        return wp.empty(n, dtype=wp.vec2f, device=DEV)
    return v2(np.full((n, 2), fill, np.float32))


def ei32(n):
    return wp.empty(n, dtype=wp.int32, device=DEV)


def ef32(n):
    return wp.empty(n, dtype=wp.float32, device=DEV)


def same(a, b):
    """Bitwise equality of two wp.arrays (NaN payloads included)."""
    x, y = a.numpy(), b.numpy()
    return x.shape == y.shape and np.array_equal(x.view(np.uint8), y.view(np.uint8))


def launch(kernel, dim, args):
    wp.launch(kernel, dim, args, device=DEV)


def _circle(n, r=1.0, ccw=True, cx=0.0, cy=0.0):
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    if not ccw:
        t = -t
    return np.stack([cx + r * np.cos(t), cy + r * np.sin(t)], -1).astype(np.float32)


def _noisy_loops(rs, E, n, nan_rows=True):
    pts = np.stack([_circle(n, r) for r in rs.uniform(0.5, 2.0, E)])
    pts = (pts + 0.08 * rs.normal(size=pts.shape)).astype(np.float32)
    if nan_rows and E >= 5:
        pts[1, 3:6] = np.nan            # interior NaN
        pts[2, 1:] = np.nan             # R == 1
        pts[3, ::4] = np.nan            # scattered NaN
        pts[4, n // 2:] = np.nan        # NaN tail
    return pts


# ---------------------------------------------------------------------------------------
# public surface
# ---------------------------------------------------------------------------------------

_ALL = ["ArcScratch", "angle_sort", "any_set", "arc_resample", "check_array",
        "closed_catmull_rom", "constants", "fill", "funcs", "grid_cells", "hash32",
        "hash32_host", "normalize_extent", "closed_polyline",
        "resample_constant_spacing", "resample_uniform", "sample_counts",
        "sample_grid_points", "self_intersections", "thickness", "turning_number"]

_CTX_OPS = ["angle_sort", "any_set", "arc_resample", "closed_catmull_rom", "fill",
            "normalize_extent", "closed_polyline", "resample_constant_spacing",
            "resample_uniform", "sample_counts", "sample_grid_points",
            "self_intersections", "thickness", "turning_number"]


def test_public_surface():
    assert sorted(ops.__all__) == sorted(_ALL)
    for name in ops.__all__:
        assert hasattr(ops, name), name
    assert ops.constants.TARGET_EXTENT == 1.44
    assert sorted(ops.funcs.__all__) == ["catmull_rom", "ccw", "cross2", "hash32", "nan0",
                                         "safe_normalize2", "segments_cross"]


@pytest.mark.parametrize("name", _CTX_OPS)
def test_every_op_takes_ctx_first_and_documents_shapes(name):
    fn = getattr(ops, name)
    params = list(inspect.signature(fn).parameters)
    assert params[0] == "ctx", params
    doc = inspect.getdoc(fn) or ""
    assert "Args:" in doc and "ctx" in doc, name
    assert "[" in doc, f"{name} docstring does not state shapes"
    assert "Raises:" in doc, name


def test_arc_scratch_and_helpers_are_documented():
    for obj in (ops.ArcScratch, ops.ArcScratch.alloc, ops.check_array, ops.grid_cells):
        assert inspect.getdoc(obj), obj
    assert "[>= E*stride]" in inspect.getdoc(ops.ArcScratch)


def test_funcs_are_warp_functions_usable_in_user_kernels():
    for name in ops.funcs.__all__:
        assert isinstance(getattr(ops.funcs, name), wp.Function), name

    @wp.kernel
    def user_k(a: wp.array(dtype=wp.vec2f), out_f: wp.array(dtype=wp.float32),
               out_v: wp.array(dtype=wp.vec2f), out_i: wp.array(dtype=wp.int32),
               out_u: wp.array(dtype=wp.uint32)):
        p0 = a[0]
        p1 = a[1]
        p2 = a[2]
        p3 = a[3]
        out_f[0] = funcs.ccw(p0[0], p0[1], p1[0], p1[1], p2[0], p2[1])
        out_f[1] = funcs.cross2(p1 - p0, p2 - p0)
        out_f[2] = funcs.nan0(wp.nan)
        out_v[0] = funcs.safe_normalize2(p1 - p0)
        out_v[1] = funcs.catmull_rom(p0, p1, p2, p3, 0.0)
        out_i[0] = funcs.segments_cross(p0, p2, p1, p3)
        out_u[0] = funcs.hash32(wp.uint32(1))

    a = v2([[0, 0], [1, 0], [1, 1], [0, 1]])
    of, ov, oi, ou = ef32(3), ev2(2), ei32(1), wp.empty(1, dtype=wp.uint32, device=DEV)
    launch(user_k, 1, [a, of, ov, oi, ou])
    f = of.numpy()
    assert f[0] == pytest.approx(1.0) and f[1] == pytest.approx(1.0) and f[2] == 0.0
    np.testing.assert_allclose(ov.numpy(), [[1, 0], [1, 0]])
    assert oi.numpy()[0] == 1                       # diagonals of the unit square cross
    assert int(ou.numpy()[0]) == ops.hash32_host(1)


_OPS_DIR = pathlib.Path(ops.__file__).parent


def test_ops_modules_do_not_import_src_or_exec_at_module_level():
    bad = []
    for path in sorted(_OPS_DIR.glob("*.py")):
        tree = ast.parse(path.read_text())
        for node in tree.body:     # module level only (TYPE_CHECKING blocks are `if`s)
            mods = []
            if isinstance(node, ast.ImportFrom):
                prefix = "." * node.level
                mods = [prefix + (node.module or "")]
            elif isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            for m in mods:
                if m.startswith(("..", "track_gen._src", "track_gen.exec")):
                    bad.append(f"{path.name}: {m}")
    assert not bad, bad


_SRC_DIR = _OPS_DIR.parent / "_src"
# The only private track_gen.ops names a generator module may import: the unvalidated
# launchers behind the legacy wrappers (corner_*_inplace, ccw_sort_inplace,
# point_*_inplace) and the legacy count floor of the bezier/hull angle sort.
_PRIVATE_OPS_ALLOWED = {
    "warp_generate.py": {"_angle_sort_launch", "_sample_counts_launch",
                         "_sample_grid_points_launch"},
    "warp_generate_hull.py": {"_angle_sort_launch", "_sample_counts_launch",
                              "_sample_grid_points_launch"},
}


def _imports(path):
    """(module, imported names, bound module aliases) of every import in ``path``."""
    out, aliases = [], {}
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom):
            mod = "." * node.level + (node.module or "")
            out.append((mod, [a.name for a in node.names]))
            for a in node.names:
                aliases[a.asname or a.name] = f"{mod}.{a.name}"
        elif isinstance(node, ast.Import):
            for a in node.names:
                out.append((a.name, []))
                aliases[a.asname or a.name.split(".")[0]] = a.name
    return out, aliases


def test_generator_modules_use_public_ops_not_private_internals():
    gens = sorted(_SRC_DIR.glob("warp_generate*.py"))
    gates = [p for p in gens if p.name.endswith("_gates.py")] + [_SRC_DIR / "warp_gate.py"]
    assert len(gens) >= 10
    bad = []
    for path in gens:
        imps, aliases = _imports(path)
        for mod, names in imps:
            if mod.endswith("warp_pipeline") or "warp_pipeline" in names:
                bad.append(f"{path.name}: imports warp_pipeline")
            if mod.startswith(("..ops", "track_gen.ops")):
                private = {n for n in names if n.startswith("_")}
                extra = private - _PRIVATE_OPS_ALLOWED.get(path.name, set())
                if extra:
                    bad.append(f"{path.name}: private ops imports {sorted(extra)}")
    for path in gates:
        imps, aliases = _imports(path)
        for mod, names in imps:
            if "warp_generate" in mod:
                private = [n for n in names if n.startswith("_")]
                if private:
                    bad.append(f"{path.name}: imports {private} from {mod}")
        gen_mods = {k for k, v in aliases.items() if "warp_generate" in v}
        for node in ast.walk(ast.parse(path.read_text())):
            if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                    and node.value.id in gen_mods and node.attr.startswith("_")):
                bad.append(f"{path.name}: uses {node.value.id}.{node.attr}")
    assert not bad, bad


def test_import_ops_loads_no_generator_module_and_no_torch():
    code = "\n".join([
        "import sys",
        "import track_gen.ops",
        "gens = sorted(m for m in sys.modules if m.startswith('track_gen._src.warp_generate'))",
        "assert not gens, gens",
        "leaked = sorted(m for m in sys.modules if m == 'torch' or m.startswith('torch.'))",
        "assert not leaked, leaked",
    ])
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_runtime_check_arr_is_the_public_validator():
    from track_gen._src import runtime
    assert runtime._check_arr is ops.check_array


# ---------------------------------------------------------------------------------------
# validation (host-side, before any launch)
# ---------------------------------------------------------------------------------------

def _arc_args(E=3, M=6, num=5):
    return dict(src=ev2(E * M), stride=M, num=num, scratch=ops.ArcScratch.alloc(E, M, DEV),
                out=ev2(E * num), count_out=ei32(E))


def _bad_cases():
    """name -> thunk building (op, kwargs, substring of the expected message)."""
    E, M, num, n, P, K, ps = 3, 6, 5, 6, 4, 4, 3

    def sc():
        return ops.ArcScratch.alloc(E, n, DEV, compact=False)

    def arc(**over):
        return {**_arc_args(E, M, num), **over}

    def cs(**over):
        return {**dict(src=ev2(E * n), stride=n, spacing=0.1, out_stride=8, scratch=sc(),
                       out=ev2(E * 8), count_out=ei32(E), truncated_out=ei32(E)), **over}

    def grid(**over):
        return {**dict(seeds=ei32(E), salt=3, offset=0, num=P, num_cells=4, cell_size=0.25,
                       scale=1.0, used=ei32(E * P), out=ev2(E * P)), **over}

    def counts(**over):
        return {**dict(seeds=ei32(E), salt=3, offset=0, lo=1, hi=4, out=ei32(E)), **over}

    def poly(**over):
        return {**dict(src=ev2(E * n), count=ei32(E), stride=n), **over}

    def cr(**over):
        return {**dict(src=ev2(E * K), stride=K, per_seg=ps, out=ev2(E * K * ps)), **over}

    def alias(view):
        b = ev2(2 * E * n)
        src = b[0:E * n]
        out = b[3:3 + E * n] if view else src
        return dict(src=src, count=ei32(E), stride=n, scratch=sc(), out=out)

    def cs_alias(view):
        b = ev2(2 * E * n)
        src = b[0:E * n]
        out = b[3:3 + E * n] if view else src
        return cs(src=src, out_stride=n, out=out)

    def scratch_alias(op):
        # out shares memory with a scratch buffer the lookup reads (a float32 view).
        if op == "arc":
            s_ = ops.ArcScratch.alloc(E, M, DEV)
            out = wp.array(ptr=s_.seg.ptr, shape=(E * num,), dtype=wp.vec2f, device=DEV)
            return arc(scratch=s_, out=out)
        s_ = sc()
        out = wp.array(ptr=s_.s.ptr, shape=(E * n,), dtype=wp.vec2f, device=DEV)
        return dict(src=ev2(E * n), count=ei32(E), stride=n, scratch=s_, out=out)

    small = lambda: ops.ArcScratch(None, ef32(E * n - 1), ef32(E * (n + 1)), None)  # noqa: E731
    c = {
        # arc_resample
        "arc/src dtype": lambda: (ops.arc_resample, arc(src=wp.empty(E * M, dtype=wp.vec3f, device=DEV)), "src"),
        "arc/src shape": lambda: (ops.arc_resample, arc(src=ev2(E * M + 1)), "src"),
        "arc/src type": lambda: (ops.arc_resample, arc(src=np.zeros((E * M, 2), np.float32)), "src"),
        "arc/out shape": lambda: (ops.arc_resample, arc(out=ev2(E * num - 1)), "out"),
        "arc/count dtype": lambda: (ops.arc_resample, arc(count_out=ef32(E)), "count_out"),
        "arc/active shape": lambda: (ops.arc_resample, arc(active=i32([1] * (E + 1))), "active"),
        "arc/scratch small": lambda: (ops.arc_resample, arc(scratch=ops.ArcScratch.alloc(E, M - 1, DEV)), "scratch"),
        "arc/scratch no real": lambda: (ops.arc_resample, arc(scratch=ops.ArcScratch.alloc(E, M, DEV, compact=False)), "scratch.real"),
        "arc/scratch type": lambda: (ops.arc_resample, arc(scratch=object()), "scratch"),
        "arc/stride": lambda: (ops.arc_resample, arc(stride=0), "stride"),
        "arc/num": lambda: (ops.arc_resample, arc(num=0), "num"),
        "arc/stride float": lambda: (ops.arc_resample, arc(stride=6.0), "stride"),
        "arc/count_out shape": lambda: (ops.arc_resample, arc(count_out=ei32(E + 1)), "count_out"),
        "arc/out aliases scratch": lambda: (ops.arc_resample, scratch_alias("arc"), "out must not alias scratch.seg"),
        # resample_uniform
        "uni/alias": lambda: (ops.resample_uniform, alias(False), "out"),
        "uni/alias view": lambda: (ops.resample_uniform, alias(True), "out"),
        "uni/stride": lambda: (ops.resample_uniform, dict(src=ev2(0), count=ei32(E), stride=0, scratch=sc(), out=ev2(0)), "stride"),
        "uni/s small": lambda: (ops.resample_uniform, dict(src=ev2(E * n), count=ei32(E), stride=n, scratch=small(), out=ev2(E * n)), "scratch.seg"),
        "uni/count dtype": lambda: (ops.resample_uniform, dict(src=ev2(E * n), count=ef32(E), stride=n, scratch=sc(), out=ev2(E * n)), "count"),
        "uni/out aliases scratch": lambda: (ops.resample_uniform, scratch_alias("uni"), "out must not alias scratch.s"),
        # resample_constant_spacing
        "cs/spacing=0": lambda: (ops.resample_constant_spacing, cs(spacing=0.0), "spacing"),
        "cs/spacing<0": lambda: (ops.resample_constant_spacing, cs(spacing=-1.0), "spacing"),
        "cs/spacing=nan": lambda: (ops.resample_constant_spacing, cs(spacing=float("nan")), "spacing"),
        "cs/spacing=inf": lambda: (ops.resample_constant_spacing, cs(spacing=float("inf")), "spacing"),
        "cs/out_stride": lambda: (ops.resample_constant_spacing, cs(out_stride=0, out=ev2(0)), "out_stride"),
        "cs/alias": lambda: (ops.resample_constant_spacing, cs_alias(False), "out must not alias src"),
        "cs/alias view": lambda: (ops.resample_constant_spacing, cs_alias(True), "out must not alias src"),
        "cs/trunc dtype": lambda: (ops.resample_constant_spacing, cs(truncated_out=ef32(E)), "truncated_out"),
        # geometry
        "selfx/out dtype": lambda: (ops.self_intersections, poly(out=ef32(E)), "out"),
        "selfx/stride": lambda: (ops.self_intersections, poly(stride=0, out=ei32(E)), "stride"),
        "selfx/src shape": lambda: (ops.self_intersections, poly(src=ev2(E * n - 1), out=ei32(E)), "src"),
        "selfx/count shape": lambda: (ops.self_intersections, poly(count=ei32(E + 1), out=ei32(E)), "count"),
        "turn/out dtype": lambda: (ops.turning_number, poly(out=ei32(E)), "out"),
        "thick/band": lambda: (ops.thickness, poly(band=ei32(E + 1), out=ef32(E)), "band"),
        "norm/divide": lambda: (ops.normalize_extent, dict(src=ev2(E * n + 1), stride=n, target_extent=1.0), "stride"),
        "norm/count": lambda: (ops.normalize_extent, dict(src=ev2(E * n), stride=n, target_extent=1.0, count=ei32(E + 1)), "count"),
        "norm/extent": lambda: (ops.normalize_extent, dict(src=ev2(E * n), stride=n, target_extent=float("nan")), "target_extent"),
        "sort/keys": lambda: (ops.angle_sort, dict(src=ev2(E * P), count=ei32(E), stride=P, keys=ef32(E * P - 1), out=ev2(E * P)), "keys"),
        "sort/out_stride": lambda: (ops.angle_sort, dict(src=ev2(E * P), count=ei32(E), stride=P, keys=ef32(E * 9), out=ev2(E * P), out_stride=P + 1), "out"),
        # sampling
        "counts/lo>hi": lambda: (ops.sample_counts, counts(lo=5, hi=4), "lo"),
        "counts/out": lambda: (ops.sample_counts, counts(out=ei32(E + 1)), "out"),
        "counts/salt range": lambda: (ops.sample_counts, counts(salt=2 ** 31), "salt"),
        "counts/salt float": lambda: (ops.sample_counts, counts(salt=3.5), "salt"),
        "grid/scale shape": lambda: (ops.sample_grid_points, grid(scale=ef32(E + 1)), "scale"),
        "grid/used": lambda: (ops.sample_grid_points, grid(used=ei32(E * P - 1)), "used"),
        "grid/num": lambda: (ops.sample_grid_points, grid(num=0, used=ei32(0), out=ev2(0)), "num"),
        "grid/cells": lambda: (ops.sample_grid_points, grid(num_cells=0), "num_cells"),
        # curves
        "cr/endpoint per_seg": lambda: (ops.closed_catmull_rom, cr(per_seg=1, out=ev2(E * K), endpoint=True), "per_seg"),
        "cr/divide": lambda: (ops.closed_catmull_rom, cr(src=ev2(E * K + 1)), "stride"),
        "cr/out": lambda: (ops.closed_catmull_rom, cr(out=ev2(E * K * ps + 1)), "out"),
        "cr/count dtype": lambda: (ops.closed_catmull_rom, cr(count=ef32(E)), "count"),
        "polyline/active": lambda: (ops.closed_polyline, cr(active=ei32(E - 1)), "active"),
        "polyline/count": lambda: (ops.closed_polyline, cr(count=ei32(E + 1)), "count"),
        # fill (ValueError for a bad array; TypeError only for an unsupported dtype)
        "fill/not array": lambda: (ops.fill, dict(arr=np.zeros(3, np.int32), value=0), "arr must be a wp.array"),
        "fill/quat scalar": lambda: (ops.fill, dict(arr=wp.empty(2, dtype=wp.quatf, device=DEV), value=1.0), "quatf"),
    }
    return c


_BAD = _bad_cases()


@pytest.mark.parametrize("case", sorted(_BAD))
def test_bad_arguments_raise_before_any_launch(case):
    fn, kw, name = _BAD[case]()
    rec = _RecCtx()
    with pytest.raises(ValueError, match=name.replace(".", r"\.")):
        fn(rec, **kw)
    assert rec.launches == []


def test_device_mismatch_raises_before_any_launch():
    # A context on another device (simulated: this host is CPU-only).
    rec = _RecCtx(device="cuda:0")
    with pytest.raises(ValueError, match="src must be on device 'cuda:0'"):
        ops.arc_resample(rec, **_arc_args())
    with pytest.raises(ValueError, match="src must be on device"):
        ops.self_intersections(rec, ev2(6), 6, ei32(1), count=ei32(1))
    with pytest.raises(ValueError, match="seeds must be on device"):
        ops.sample_grid_points(rec, ei32(1), 1, 0, 2, 2, 0.5, 1.0, ei32(2), ev2(2))
    with pytest.raises(ValueError, match="arr must be on device 'cuda:0'"):
        ops.fill(rec, ei32(3), 0)
    assert rec.launches == []


def test_device_aliases_are_canonicalised(monkeypatch):
    # A duck-typed context may carry a device alias ("cuda" for cuda:0). Simulated on a
    # CPU-only host: "host-alias" resolves to the cpu device through wp.get_device.
    real = wp.get_device
    monkeypatch.setattr(wp, "get_device",
                        lambda d=None: real("cpu") if d == "host-alias" else real(d))
    ops.check_array("x", ei32(4), (4,), wp.int32, "host-alias")
    ops.check_array("x", ei32(4), (4,), wp.int32, real("cpu"))      # a wp.Device
    rec = _RecCtx(device="host-alias")
    E, n = 2, 8
    out = ei32(E)
    ops.self_intersections(rec, v2(np.stack([_circle(n)] * E)), n, out, count=i32([n] * E))
    ops.fill(rec, out, 3)
    ops.any_set(rec, out, ei32(1))
    assert rec.launches and (out.numpy() == 3).all()
    # An identifier Warp does not resolve is compared verbatim (and named in the error).
    with pytest.raises(ValueError, match="x must be on device 'cuda:7', got 'cpu'"):
        ops.check_array("x", ei32(4), (4,), wp.int32, "cuda:7")


@pytest.mark.parametrize("bad,match", [
    ("notarray", "x must be a wp.array"),
    ("shape", r"x must have shape \(4,\), got \(3,\)"),
    ("dtype", "x must have dtype int32, got float32"),
    ("device", "x must be on device 'cuda:0'"),
])
def test_check_array_messages(bad, match):
    arr = {"notarray": [1, 2, 3, 4], "shape": ei32(3), "dtype": ef32(4), "device": ei32(4)}[bad]
    dev = "cuda:0" if bad == "device" else DEV
    with pytest.raises(ValueError, match=match):
        ops.check_array("x", arr, (4,), wp.int32, dev)
    ops.check_array("x", ei32(4), (4,), wp.int32, DEV)


# ---------------------------------------------------------------------------------------
# semantics on hand-built cases
# ---------------------------------------------------------------------------------------

def test_arc_resample_compacts_nan_and_degenerate_rows(ctx):
    M, num = 8, 16
    sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
    rows = np.full((3, M, 2), np.nan, np.float32)
    rows[0, [0, 2, 4, 6]] = sq            # interior NaN between the 4 real corners
    rows[1, 3] = (5.0, 5.0)               # R == 1 -> degenerate
    rows[2, :4] = sq
    rows[2, 4:] = sq                      # 8 real points (square traced twice)
    sc = ops.ArcScratch.alloc(3, M, DEV)
    out, co = ev2(3 * num), ei32(3)
    ops.arc_resample(ctx, v2(rows), M, num, sc, out, co)
    np.testing.assert_array_equal(co.numpy(), [num, 0, num])
    np.testing.assert_array_equal(sc.count_real.numpy(), [4, 1, 8])
    o = out.numpy().reshape(3, num, 2)
    assert np.isnan(o[1]).all()
    np.testing.assert_array_equal(o[0, 0], sq[0])             # starts at the first real point
    # uniform spacing along the closed square (perimeter 4 -> 0.25 per step)
    step = np.linalg.norm(np.diff(np.vstack([o[0], o[0, :1]]), axis=0), axis=1)
    np.testing.assert_allclose(step, 0.25, atol=1e-6)


def test_arc_resample_active_leaves_inactive_rows_bitwise(ctx):
    E, M, num = 4, 12, 7
    rs = np.random.default_rng(0)
    src = v2(_noisy_loops(rs, E, M, nan_rows=False))
    sentinel = 3.25
    out, co = ev2(E * num, sentinel), i32([9] * E)
    sc = ops.ArcScratch.alloc(E, M, DEV)
    ops.arc_resample(ctx, src, M, num, sc, out, co, active=i32([1, 0, 1, 0]))
    o = out.numpy().reshape(E, num, 2)
    assert (o[[1, 3]] == sentinel).all()
    assert np.isfinite(o[[0, 2]]).all() and not (o[[0, 2]] == sentinel).any()
    np.testing.assert_array_equal(co.numpy(), [num, 0, num, 0])
    np.testing.assert_array_equal(sc.count_real.numpy(), [M, 0, M, 0])


def test_resample_uniform_count_semantics(ctx):
    n = 10
    rows = np.stack([_circle(n), _circle(n, 2.0), _circle(n, 3.0)])
    count = i32([n, 4, 0])
    out = ev2(3 * n)
    ops.resample_uniform(ctx, v2(rows), n, ops.ArcScratch.alloc(3, n, DEV, compact=False),
                         out, count=count)
    o = out.numpy().reshape(3, n, 2)
    assert np.isfinite(o[0]).all()
    assert np.isfinite(o[1, :4]).all() and np.isnan(o[1, 4:]).all()
    assert np.isnan(o[2]).all()
    np.testing.assert_array_equal(o[1, 0], rows[1, 0])


def test_resample_constant_spacing_counts_truncation_and_nan(ctx):
    N = 64
    rows = np.stack([_circle(N), _circle(N), _circle(N)])
    rows[2, 5] = np.nan
    perim = float(np.linalg.norm(np.diff(np.vstack([rows[0], rows[0, :1]]), axis=0),
                                 axis=1).sum())
    spacing = 0.5
    want = math.floor(perim / spacing) + 1
    while want > 1 and (want - 1) * spacing >= perim:
        want -= 1
    sc = ops.ArcScratch.alloc(3, N, DEV, compact=False)
    for n_max, cnt0, tr0 in ((20, want, 0), (want - 3, want - 3, 1)):
        out, co, tr = ev2(3 * n_max), ei32(3), ei32(3)
        ops.resample_constant_spacing(ctx, v2(rows), N, spacing, n_max, sc, out, co, tr)
        np.testing.assert_array_equal(co.numpy(), [cnt0, cnt0, 0])
        np.testing.assert_array_equal(tr.numpy(), [tr0, tr0, 0])
        o = out.numpy().reshape(3, n_max, 2)
        assert np.isfinite(o[0, :cnt0]).all() and np.isnan(o[0, cnt0:]).all()
        assert np.isnan(o[2]).all()
        d = np.linalg.norm(np.diff(o[0, :cnt0], axis=0), axis=1)
        np.testing.assert_allclose(d, spacing, rtol=2e-2)   # chord ~ arc on a fine polygon


def test_self_intersections_known_shapes(ctx):
    n = 8
    square = np.zeros((n, 2), np.float32)
    square[:4] = [[0, 0], [1, 0], [1, 1], [0, 1]]
    eight = np.zeros((n, 2), np.float32)
    eight[:4] = [[0, 0], [1, 1], [1, 0], [0, 1]]      # bow-tie: one crossing
    eight_nan = eight.copy()
    eight_nan[0] = np.nan                             # read through nan0 as (0, 0)
    eight_nan[6] = np.nan                             # beyond count: never read
    rows = np.stack([square, eight, eight, eight_nan])
    out = i32([7, 7, 7, 7])
    ops.self_intersections(ctx, v2(rows), n, out, count=i32([4, 4, 3, 4]))
    np.testing.assert_array_equal(out.numpy(), [0, 1, 0, 1])


def test_turning_number_sign(ctx):
    n = 32
    rows = np.stack([_circle(n), _circle(n, ccw=False)])
    out = ef32(2)
    ops.turning_number(ctx, v2(rows), n, out, count=i32([n, n]))
    np.testing.assert_allclose(out.numpy(), [2 * np.pi, -2 * np.pi], atol=1e-4)


def test_thickness_of_a_circle_is_its_radius(ctx):
    n, r = 64, 1.7
    out = ef32(1)
    ops.thickness(ctx, v2(_circle(n, r)), n, i32([n // 2 - 1]), out, count=i32([n]))
    assert out.numpy()[0] == pytest.approx(r, rel=1e-3)


def test_normalize_extent_dense_and_counted(ctx):
    rs = np.random.default_rng(3)
    n, E, target = 9, 4, 1.44
    pts = (rs.normal(size=(E, n, 2)) * [3.0, 0.5] + 7.0).astype(np.float32)
    a = v2(pts)
    ops.normalize_extent(ctx, a, n, target)
    o = a.numpy().reshape(E, n, 2)
    lo, hi = o.min(1), o.max(1)
    np.testing.assert_allclose((hi - lo).max(1), target, rtol=1e-5)
    np.testing.assert_allclose(lo + hi, 0.0, atol=1e-5)
    b = v2(pts)
    ops.normalize_extent(ctx, b, n, target, count=i32([n, 3, 0, n + 5]))
    o = b.numpy().reshape(E, n, 2)
    assert np.isnan(o[1, 3:]).all() and np.isnan(o[2]).all() and np.isfinite(o[3]).all()
    lo, hi = o[1, :3].min(0), o[1, :3].max(0)
    assert (hi - lo).max() == pytest.approx(target, rel=1e-5)


def _np_angle_order(p):
    c = p.astype(np.float64).mean(0).astype(np.float32)
    keys = np.arctan2(p[:, 0] - c[0], p[:, 1] - c[1])
    return p[np.argsort(keys, kind="stable")]


@pytest.mark.parametrize("out_stride", [None, 3, 9])
def test_angle_sort_order_tail_and_clamps(ctx, out_stride):
    rs = np.random.default_rng(5)
    P, E = 6, 5
    os_ = P if out_stride is None else out_stride
    pts = rs.normal(size=(E, P, 2)).astype(np.float32)
    counts = [P, 4, 0, P + 3, -2]
    out = ev2(E * os_)
    ops.angle_sort(ctx, v2(pts), P, ef32(E * os_), out, count=i32(counts),
                   out_stride=out_stride)
    o = out.numpy().reshape(E, os_, 2)
    for e, c in enumerate(counts):
        m = max(0, min(c, P, os_))
        if m:
            np.testing.assert_allclose(o[e, :m], _np_angle_order(pts[e, :m]), atol=1e-6)
        assert np.isnan(o[e, m:]).all()


def test_sample_counts_range_determinism_and_salts(ctx):
    seeds = i32(np.arange(64) * 7919 + 3)
    a, b, c = ei32(64), ei32(64), ei32(64)
    ops.sample_counts(ctx, seeds, 6151, 0, 3, 9, a)
    ops.sample_counts(ctx, seeds, 6151, 0, 3, 9, b)
    ops.sample_counts(ctx, seeds, 5119, 1, 3, 9, c)
    va = a.numpy()
    assert va.min() >= 3 and va.max() <= 9 and len(set(va.tolist())) > 3
    assert same(a, b)
    assert not np.array_equal(va, c.numpy())
    ops.sample_counts(ctx, seeds, 6151, 0, 4, 4, a)
    assert (a.numpy() == 4).all()


def test_sample_grid_points_bounds_and_array_scale(ctx):
    E, P, mpd = 16, 7, 0.05
    nc, cell = ops.grid_cells(mpd)
    assert (nc, cell) == (int(1.0 / (mpd * 2)), mpd * 2.0)
    seeds = i32(_INT32_OVERFLOW_SEEDS + list(range(E - len(_INT32_OVERFLOW_SEEDS))))
    out, used = ev2(E * P), ei32(E * P)
    ops.sample_grid_points(ctx, seeds, 9781, 0, P, nc, cell, 1.0, used, out)
    o = out.numpy()
    assert o.min() >= -0.5 and o.max() < (nc - 1) * cell + 0.5
    u = used.numpy().reshape(E, P)
    assert u.min() >= 0 and u.max() < nc * nc
    out2, used2 = ev2(E * P), ei32(E * P)
    ops.sample_grid_points(ctx, seeds, 9781, 0, P, nc, cell, f32([1.0] * E), used2, out2)
    assert same(out, out2) and same(used, used2)


def test_closed_catmull_rom_interpolates_controls(ctx):
    K, ps, E = 5, 4, 2
    rs = np.random.default_rng(9)
    ctrl = rs.normal(size=(E, K, 2)).astype(np.float32)
    out = ev2(E * K * ps)
    ops.closed_catmull_rom(ctx, v2(ctrl), K, ps, out)
    o = out.numpy().reshape(E, K, ps, 2)
    np.testing.assert_array_equal(o[:, :, 0], ctrl)               # u = 0 is the control
    ops.closed_catmull_rom(ctx, v2(ctrl), K, ps, out, endpoint=True)
    o = out.numpy().reshape(E, K, ps, 2)
    np.testing.assert_allclose(o[:, :, -1], np.roll(ctrl, -1, axis=1), atol=1e-5)
    ops.closed_catmull_rom(ctx, v2(ctrl), K, ps, out, count=i32([3, 1]), endpoint=True)
    o = out.numpy().reshape(E, K, ps, 2)
    assert np.isfinite(o[0, :3]).all() and np.isnan(o[0, 3:]).all()
    assert np.isnan(o[1]).all()
    np.testing.assert_allclose(o[0, 2, -1], ctrl[0, 0], atol=1e-5)  # wraps mod count


def test_closed_polyline_writes_all_or_only_active_rows(ctx):
    K, ps, E = 4, 3, 3
    rs = np.random.default_rng(11)
    ctrl = rs.normal(size=(E, K, 2)).astype(np.float32)
    every = ev2(E * K * ps, -9.5)
    ops.closed_polyline(ctx, v2(ctrl), K, ps, every)
    assert np.isfinite(every.numpy()).all()
    for count in (None, i32([K, 2, 1])):
        out = ev2(E * K * ps, -9.5)
        ops.closed_polyline(ctx, v2(ctrl), K, ps, out, count=count, active=i32([1, 0, 1]))
        o = out.numpy().reshape(E, K, ps, 2)
        assert (o[1] == -9.5).all()
        if count is None:
            assert same(out[0:K * ps], every[0:K * ps])
        np.testing.assert_array_equal(o[0, :, 0], ctrl[0])
        u = np.arange(ps, dtype=np.float32) / ps
        expect = (1 - u)[:, None] * ctrl[0, 1] + u[:, None] * ctrl[0, 2]
        np.testing.assert_allclose(o[0, 1], expect, atol=1e-6)
        if count is not None:
            assert np.isnan(o[2]).all()                   # count 1 < 2 -> all NaN


def test_count_above_stride_is_clamped_to_the_row(ctx):
    # count[e] > stride behaves as count[e] == stride in every count-aware op (the kernels
    # clamp, so an oversized count never reads or writes the next env's row).
    rs = np.random.default_rng(21)
    E, n, K, ps = 3, 12, 6, 4
    loops = v2(_noisy_loops(rs, E, n, nan_rows=False))
    big, full = i32([n + 3, n, n + 1]), i32([n] * E)
    sc = ops.ArcScratch.alloc(E, n, DEV, compact=False)
    a, b = ev2(E * n), ev2(E * n)
    ops.resample_uniform(ctx, loops, n, sc, a, count=big)
    ops.resample_uniform(ctx, loops, n, sc, b, count=full)
    assert same(a, b)
    for op, mk in ((ops.self_intersections, ei32), (ops.turning_number, ef32)):
        a, b = mk(E), mk(E)
        op(ctx, loops, n, a, count=big)
        op(ctx, loops, n, b, count=full)
        assert same(a, b), op.__name__
    band = i32([2] * E)
    a, b = ef32(E), ef32(E)
    ops.thickness(ctx, loops, n, band, a, count=big)
    ops.thickness(ctx, loops, n, band, b, count=full)
    assert same(a, b)
    ctrl = v2(rs.normal(size=(E * K, 2)))
    kb, kf = i32([K + 2, K, K + 1]), i32([K] * E)
    for op, kw in ((ops.closed_catmull_rom, {}), (ops.closed_polyline, {}),
                   (ops.closed_polyline, {"active": i32([1, 0, 1])})):
        a, b = ev2(E * K * ps, 0.5), ev2(E * K * ps, 0.5)
        op(ctx, ctrl, K, ps, a, count=kb, **kw)
        op(ctx, ctrl, K, ps, b, count=kf, **kw)
        assert same(a, b), op.__name__


def test_count_above_stride_stays_in_bounds_in_debug_mode():
    # Warp's debug mode bounds-checks every array access (an out-of-bounds index aborts
    # the process), so this proves the clamps rather than relying on the values above.
    code = "\n".join([
        "import numpy as np, warp as wp",
        "wp.config.mode = 'debug'",
        "wp.config.quiet = True",
        "from track_gen import ops",
        "from track_gen.exec import LaunchContext",
        "D = 'cpu'; ctx = LaunchContext(D, None); E, n, K, ps = 2, 8, 5, 3",
        "v = lambda a: wp.array(np.asarray(a, np.float32).reshape(-1, 2), dtype=wp.vec2f, device=D)",
        "i = lambda a: wp.array(np.asarray(a, np.int32), dtype=wp.int32, device=D)",
        "t = np.linspace(0, 2 * np.pi, n, endpoint=False)",
        "loops = v(np.stack([np.stack([np.cos(t), np.sin(t)], -1)] * E)); big = i([n, n + 3])",
        "sc = ops.ArcScratch.alloc(E, n, D, compact=False)",
        "ops.resample_uniform(ctx, loops, n, sc, wp.empty(E * n, dtype=wp.vec2f, device=D), count=big)",
        "ops.self_intersections(ctx, loops, n, wp.empty(E, dtype=wp.int32, device=D), count=big)",
        "ops.turning_number(ctx, loops, n, wp.empty(E, dtype=wp.float32, device=D), count=big)",
        "ops.thickness(ctx, loops, n, i([1, 1]), wp.empty(E, dtype=wp.float32, device=D), count=big)",
        "ctrl = v(np.random.default_rng(0).normal(size=(E * K, 2))); kb = i([K, K + 3])",
        "o = wp.empty(E * K * ps, dtype=wp.vec2f, device=D)",
        "ops.closed_catmull_rom(ctx, ctrl, K, ps, o, count=kb)",
        "ops.closed_polyline(ctx, ctrl, K, ps, o, count=kb, active=i([1, 1]))",
        "wp.synchronize()",
        "print('in bounds')",
    ])
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       timeout=600)
    assert r.returncode == 0 and "in bounds" in r.stdout, r.stdout + r.stderr


def test_grid_cells_validates_the_distance():
    assert ops.grid_cells(0.05) == (int(1.0 / 0.1), 0.1)
    assert ops.grid_cells(0.5) == (1, 1.0)
    for bad in (0.0, -0.1, 0.51, float("nan"), float("inf"), "x"):
        with pytest.raises(ValueError, match="min_point_distance"):
            ops.grid_cells(bad)


def test_fill_quatf(ctx):
    arr = wp.empty(3, dtype=wp.quatf, device=DEV)
    ops.fill(ctx, arr, (0.0, 0.0, 0.0, 1.0))
    np.testing.assert_array_equal(arr.numpy(), [[0, 0, 0, 1]] * 3)


# ---------------------------------------------------------------------------------------
# bitwise equivalence with the frozen HEAD kernels
# ---------------------------------------------------------------------------------------

def test_arc_resample_bitwise_vs_head(ctx):
    rs = np.random.default_rng(1)
    E, M, num = 7, 40, 24
    src = v2(_noisy_loops(rs, E, M))
    act = i32([1, 0, 1, 1, 0, 1, 1])
    for active in (None, act):
        new = ops.ArcScratch.alloc(E, M, DEV)
        old = ops.ArcScratch.alloc(E, M, DEV)
        o_new, o_old = ev2(E * num, 3.25), ev2(E * num, 3.25)
        c_new, c_old = ei32(E), ei32(E)
        ops.arc_resample(ctx, src, M, num, new, o_new, c_new, active=active)
        if active is None:
            launch(L._pipe_arc_scan_k, E, [src, M, num, old.real, old.seg, old.s,
                                            old.count_real, c_old])
            launch(L._pipe_arc_lookup_k, E * num, [old.real, old.seg, old.s,
                                                   old.count_real, M, num, o_old])
        else:
            launch(L._pipe_arc_scan_selected_k, E, [src, active, M, num, old.real, old.seg,
                                                    old.s, old.count_real, c_old])
            launch(L._pipe_arc_lookup_selected_k, E * num, [old.real, old.seg, old.s,
                                                            old.count_real, active, M, num,
                                                            o_old])
        assert same(o_new, o_old) and same(c_new, c_old)
        assert same(new.count_real, old.count_real)


def test_resample_uniform_bitwise_vs_head(ctx):
    rs = np.random.default_rng(2)
    E, n = 8, 30
    src = v2(_noisy_loops(rs, E, n, nan_rows=False))
    count = i32([30, 30, 12, 29, 5, 30, 0, 1])
    new = ops.ArcScratch.alloc(E, n, DEV, compact=False)
    o_new, o_old = ev2(E * n), ev2(E * n)
    seg, s = ef32(E * n), ef32(E * (n + 1))
    ops.resample_uniform(ctx, src, n, new, o_new, count=count)
    launch(L._pipe_resample_scan_k, E, [src, n, count, seg, s])
    launch(L._pipe_resample_lookup_k, E * n, [src, seg, s, n, count, o_old])
    assert same(o_new, o_old)


def test_resample_constant_spacing_bitwise_vs_head(ctx):
    rs = np.random.default_rng(4)
    E, N = 6, 30
    pts = _noisy_loops(rs, E, N, nan_rows=False)
    pts[5, 7] = np.nan
    src = v2(pts)
    for n_max, spacing in ((64, 0.21), (16, 0.21), (40, 0.05)):
        sc = ops.ArcScratch.alloc(E, N, DEV, compact=False)
        o1, c1, t1 = ev2(E * n_max), ei32(E), ei32(E)
        o2, c2, t2 = ev2(E * n_max), ei32(E), ei32(E)
        seg, s = ef32(E * N), ef32(E * (N + 1))
        ops.resample_constant_spacing(ctx, src, N, spacing, n_max, sc, o1, c1, t1)
        launch(L._pipe_cs_scan_k, E, [src, N, float(spacing), n_max, seg, s, c2, t2])
        launch(L._pipe_cs_lookup_k, E * n_max, [src, seg, s, N, float(spacing), n_max, c2, o2])
        assert same(o1, o2) and same(c1, c2) and same(t1, t2)


def _random_polys(rs, E, n):
    poly = rs.normal(size=(E, n, 2)).astype(np.float32)
    poly[1, 3] = np.nan
    poly[2, 0] = (np.inf, 1.0)
    return poly


def test_self_intersections_bitwise_vs_head(ctx):
    rs = np.random.default_rng(6)
    E, n = 8, 30
    poly = v2(_random_polys(rs, E, n))
    count = i32([30, 0, 3, 4, 17, 30, 12, 25])
    a, b = ei32(E), ei32(E)
    ops.self_intersections(ctx, poly, n, a, count=count)
    launch(L._pipe_fill_i32_k, E, [b, 0])
    launch(L._pipe_self_intersections_by_i_k, E * n, [poly, n, count, b])
    assert same(a, b) and a.numpy().max() > 0


def test_turning_and_thickness_bitwise_vs_head(ctx):
    rs = np.random.default_rng(7)
    E, n = 8, 24
    poly = v2(_noisy_loops(rs, E, n, nan_rows=False))
    count = i32([24, 24, 12, 3, 0, 1, 2, 20])
    band = i32([1, 2, 3, 0, 1, 2, 5, 4])
    a, b = ef32(E), ef32(E)
    ops.turning_number(ctx, poly, n, a, count=count)
    launch(L._pipe_turning_k, E, [poly, n, count, b])
    assert same(a, b)
    ops.thickness(ctx, poly, n, band, a, count=count)
    launch(L._pipe_thickness_k, E, [poly, band, n, count, b])
    assert same(a, b)


@wp.kernel
def _geom_funcs_new_k(poly: wp.array(dtype=wp.vec2f), n: int, count: wp.array(dtype=wp.int32),
                      band: wp.array(dtype=wp.int32), x: wp.array(dtype=wp.int32),
                      tu: wp.array(dtype=wp.float32), th: wp.array(dtype=wp.float32)):
    e = wp.tid()
    x[e] = _ops_selfx_func(poly, e * n, count[e])
    tu[e] = _ops_turning_func(poly, e * n, count[e])
    th[e] = _ops_thickness_func(poly, e * n, count[e], band[e])


@wp.kernel
def _geom_funcs_old_k(poly: wp.array(dtype=wp.vec2f), n: int, count: wp.array(dtype=wp.int32),
                      band: wp.array(dtype=wp.int32), x: wp.array(dtype=wp.int32),
                      tu: wp.array(dtype=wp.float32), th: wp.array(dtype=wp.float32)):
    e = wp.tid()
    x[e] = L._pipe_self_intersections_func(poly, e * n, count[e])
    tu[e] = L._pipe_turning_func(poly, e * n, count[e])
    th[e] = L._pipe_thickness_func(poly, e * n, count[e], band[e])


def test_geometry_device_funcs_bitwise_vs_head():
    # The @wp.func forms _validity_k composes (border crossings, turning, thickness).
    rs = np.random.default_rng(8)
    E, n = 8, 20
    poly = v2(_random_polys(rs, E, n))
    count = i32([20, 0, 3, 4, 17, 20, 12, 19])
    band = i32([1, 2, 3, 0, 1, 2, 5, 4])
    outs = [(ei32(E), ef32(E), ef32(E)) for _ in range(2)]
    launch(_geom_funcs_new_k, E, [poly, n, count, band, *outs[0]])
    launch(_geom_funcs_old_k, E, [poly, n, count, band, *outs[1]])
    for a, b in zip(*outs):
        assert same(a, b)


def test_normalize_extent_bitwise_vs_head(ctx):
    rs = np.random.default_rng(10)
    E, n = 6, 17
    pts = (rs.normal(size=(E, n, 2)) * 3 + 2).astype(np.float32)
    pts[5] = 0.0                                          # zero extent -> 1e-8 floor
    for k in (L._polar_normalize_centerline_k, L._voronoi_normalize_centerline_k,
              L._ckpt_normalize_centerline_k):
        a, b = v2(pts), v2(pts)
        ops.normalize_extent(ctx, a, n, 1.44 * 0.8)
        launch(k, E, [b, n, 1.44 * 0.8])
        assert same(a, b), k.key
    count = i32([n, 0, 3, -1, n + 4, 1])
    a, b = v2(pts), v2(pts)
    ops.normalize_extent(ctx, a, n, 1.44, count=count)
    launch(L._gate_normalize_positions_k, E, [b, n, count, 1.44])
    assert same(a, b)


def test_angle_sort_bitwise_vs_head(ctx):
    rs = np.random.default_rng(12)
    E, P = 9, 11
    pts = v2(rs.normal(size=(E * P, 2)))
    # legacy bezier / hull sort: m floor 1, counts include 0, 1, P and > P
    cnt = i32([0, 1, 2, 3, P, P + 4, 5, 7, 9])
    for legacy in (L._bezier_ccw_sort_k, L._hull_angle_sort_k):
        k1, k2, o1, o2 = ef32(E * P), ef32(E * P), ev2(E * P), ev2(E * P)
        _angle_sort_launch(ctx, pts, P, cnt, P, k1, o1, 1)
        launch(legacy, E, [pts, P, cnt, k2, o2])
        assert same(o1, o2), legacy.key
    # public op (m floor 0) == bezier sort for every count >= 1
    pos = i32([1, 2, 3, P, P + 4, 5, 7, 9, 1])
    k1, k2, o1, o2 = ef32(E * P), ef32(E * P), ev2(E * P), ev2(E * P)
    ops.angle_sort(ctx, pts, P, k1, o1, count=pos)
    launch(L._bezier_ccw_sort_k, E, [pts, P, pos, k2, o2])
    assert same(o1, o2)
    # public op == gate ccw ordering (src stride != out stride, counts 0, -1, > stride)
    S, G = 9, 12
    src = v2(rs.normal(size=(E * S, 2)))
    gcnt = i32([0, 1, 2, 5, 9, 12, 7, -1, 3])
    k1, k2, o1, o2 = ef32(E * G), ef32(E * G), ev2(E * G), ev2(E * G)
    ops.angle_sort(ctx, src, S, k1, o1, count=gcnt, out_stride=G)
    launch(L._gate_order_ccw_k, E, [src, S, gcnt, G, k2, o2])
    assert same(o1, o2)


def test_sample_counts_bitwise_vs_head(ctx):
    seeds = i32(_INT32_OVERFLOW_SEEDS)
    E = len(_INT32_OVERFLOW_SEEDS)
    for attempt in (0, 3):
        a, b = ei32(E), ei32(E)
        ops.sample_counts(ctx, seeds, 6151, attempt, 3, 11, a)
        launch(L._bezier_corner_count_sample_k, E, [seeds, attempt, 3, 11, b])
        assert same(a, b)
    a, b = ei32(E), ei32(E)
    ops.sample_counts(ctx, seeds, 5119, 1, 4, 9, a)
    launch(L._hull_point_count_sample_k, E, [seeds, 4, 9, b])
    assert same(a, b)


def test_sample_grid_points_bitwise_vs_head(ctx):
    seeds = i32(_INT32_OVERFLOW_SEEDS)
    E, P, mpd, scale = len(_INT32_OVERFLOW_SEEDS), 11, 0.05, 0.9
    nc, cell = ops.grid_cells(mpd)
    nc2 = nc * nc
    sc = f32(np.random.default_rng(13).uniform(0.5, 1.5, E))

    def legacy(kernel, args):
        used, out = ei32(E * P), ev2(E * P)
        launch(L._pipe_fill_i32_k, E * P, [used, -1])
        launch(kernel, E, [seeds, *args, P, used, out])
        return used, out

    for attempt in (0, 3):
        for s_new, kern, s_old in ((scale, L._bezier_corner_sample_k, float(scale)),
                                   (sc, L._bezier_corner_sample_style_k, sc)):
            used, out = ei32(E * P), ev2(E * P)
            ops.sample_grid_points(ctx, seeds, 9781, attempt, P, nc, cell, s_new, used, out)
            u2, o2 = legacy(kern, [attempt, nc, nc2, float(cell), s_old])
            assert same(out, o2) and same(used, u2)
    used, out = ei32(E * P), ev2(E * P)
    ops.sample_grid_points(ctx, seeds, 7919, 1, P, nc, cell, scale, used, out)
    u2, o2 = legacy(L._hull_point_sample_k, [nc, nc2, float(cell), float(scale)])
    assert same(out, o2) and same(used, u2)


def test_catmull_rom_bitwise_vs_head(ctx):
    rs = np.random.default_rng(14)
    E, K, ps = 9, 13, 30
    ctrl = v2(rs.normal(size=(E * K, 2)))
    a, b = ev2(E * K * ps), ev2(E * K * ps)
    ops.closed_catmull_rom(ctx, ctrl, K, ps, a)
    launch(L._voronoi_catmull_rom_dense_k, E * K * ps, [ctrl, K, ps, b])
    assert same(a, b)
    launch(L._polar_polar_spline_dense_k, E, [ctrl, K, ps, b])
    assert same(a, b)
    # hull: count-aware, endpoint-inclusive (u = s / (npseg - 1))
    AP = 14
    aug = v2(rs.normal(size=(E * AP, 2)))
    cnt = i32([0, 1, 2, 3, 6, 14, 13, 8, 10])
    a, b = ev2(E * AP * ps), ev2(E * AP * ps)
    ops.closed_catmull_rom(ctx, aug, AP, ps, a, count=cnt, endpoint=True)
    launch(L._hull_catmull_rom_k, E * AP * ps, [aug, cnt, AP, ps, b])
    assert same(a, b)


def test_polygon_fallback_bitwise_vs_head(ctx):
    rs = np.random.default_rng(15)
    E, AP, ps = 9, 14, 7
    aug = v2(rs.normal(size=(E * AP, 2)))
    act = i32([1, 0, 1, 1, 0, 1, 1, 1, 0])
    a, b = ev2(E * AP * ps, 1.5), ev2(E * AP * ps, 1.5)
    ops.closed_polyline(ctx, aug, AP, ps, a, active=act)
    launch(L._voronoi_assemble_polygon_selected_k, E * AP * ps, [aug, AP, ps, act, b])
    assert same(a, b)
    cnt = i32([0, 1, 2, 3, 6, 14, 13, 8, 10])
    a, b = ev2(E * AP * ps, 1.5), ev2(E * AP * ps, 1.5)
    ops.closed_polyline(ctx, aug, AP, ps, a, count=cnt, endpoint=True, active=act)
    launch(L._hull_assemble_polygon_selected_k, E * AP * ps, [aug, cnt, AP, ps, act, b])
    assert same(a, b)


@wp.kernel
def _funcs_new_k(v: wp.array(dtype=wp.vec2f), f: wp.array(dtype=wp.float32),
                 ov: wp.array(dtype=wp.vec2f), of: wp.array(dtype=wp.float32),
                 oi: wp.array(dtype=wp.int32)):
    t = wp.tid()
    n = v.shape[0]
    a = v[t]
    b = v[(t + 1) % n]
    c = v[(t + 2) % n]
    d = v[(t + 3) % n]
    ov[4 * t + 0] = funcs.safe_normalize2(a)
    ov[4 * t + 1] = funcs.catmull_rom(a, b, c, d, f[t])
    of[3 * t + 0] = funcs.cross2(a, b)
    of[3 * t + 1] = funcs.ccw(a[0], a[1], b[0], b[1], c[0], c[1])
    of[3 * t + 2] = funcs.nan0(f[t] / a[0])
    oi[t] = funcs.segments_cross(a, b, c, d)


@wp.kernel
def _funcs_old_k(v: wp.array(dtype=wp.vec2f), f: wp.array(dtype=wp.float32),
                 ov: wp.array(dtype=wp.vec2f), of: wp.array(dtype=wp.float32),
                 oi: wp.array(dtype=wp.int32), which: int):
    t = wp.tid()
    n = v.shape[0]
    a = v[t]
    b = v[(t + 1) % n]
    c = v[(t + 2) % n]
    d = v[(t + 3) % n]
    if which == 0:
        ov[4 * t + 0] = L._bezier_safe_normalize2(a)
        ov[4 * t + 1] = L._polar_catmull_rom(a, b, c, d, f[t])
    elif which == 1:
        ov[4 * t + 0] = L._gate_safe_normalize2(a)
        ov[4 * t + 1] = L._voronoi_catmull_rom(a, b, c, d, f[t])
    else:
        ov[4 * t + 0] = L._rep_safe_dir(a)
        ov[4 * t + 1] = L._voronoi_catmull_rom(a, b, c, d, f[t])
    of[3 * t + 0] = L._gate_cross2(a, b)
    of[3 * t + 1] = L._pipe_ccw(a[0], a[1], b[0], b[1], c[0], c[1])
    of[3 * t + 2] = L._pipe_nan0(f[t] / a[0])
    oi[t] = L._gate_proper_segment_intersection(a, b, c, d)


def test_device_funcs_bitwise_vs_head():
    rs = np.random.default_rng(16)
    n = 64
    pts = rs.normal(size=(n, 2)).astype(np.float32)
    pts[3] = 0.0
    pts[4] = (1e-12, -1e-12)
    pts[5] = (np.nan, 1.0)
    pts[6, 0] = 0.0                      # f/0 -> inf -> nan0
    v, f = v2(pts), f32(rs.uniform(0, 1, n))
    new = (ev2(4 * n, 0.0), ef32(3 * n), ei32(n))
    launch(_funcs_new_k, n, [v, f, *new])
    for which in (0, 1, 2):
        old = (ev2(4 * n, 0.0), ef32(3 * n), ei32(n))
        launch(_funcs_old_k, n, [v, f, *old, which])
        for a, b in zip(new, old):
            assert same(a, b), which


@pytest.mark.parametrize("dtype,new_k,old_k,value", [
    (wp.int32, "_fill_i32_k", L._pipe_fill_i32_k, -1),
    (wp.int32, "_fill_i32_k", L._gate_fill_i32_k, 7),
    (wp.float32, "_fill_f32_k", L._pipe_fill_f32_k, 0.3),
    (wp.float32, "_fill_f32_k", L._gate_fill_f32_k, -2.5),
    (wp.vec2f, "_fill_vec2f_k", L._gate_fill_vec2_k, wp.vec2f(1.0, float("nan"))),
    (wp.vec3f, "_fill_vec3f_k", L._gate_fill_vec3_k, wp.vec3f(1.0, 2.0, 3.0)),
    (wp.quatf, "_fill_quatf_k", L._gate_fill_quat_k, wp.quatf(0.0, 0.0, 0.0, 1.0)),
])
def test_fill_kernels_bitwise_vs_head(dtype, new_k, old_k, value):
    fill_mod = importlib.import_module("track_gen.ops.fill")
    a, b = wp.empty(5, dtype=dtype, device=DEV), wp.empty(5, dtype=dtype, device=DEV)
    launch(getattr(fill_mod, new_k), 5, [a, value])
    launch(old_k, 5, [b, value])
    assert same(a, b)


# ---------------------------------------------------------------------------------------
# no allocation inside an op
# ---------------------------------------------------------------------------------------

_ALLOCATORS = ("empty", "zeros", "ones", "full", "empty_like", "zeros_like", "ones_like",
               "full_like", "clone", "from_numpy")


@contextlib.contextmanager
def _no_alloc():
    """Tripwire every device allocation: ``wp.array(...)`` without ``ptr=`` (views and
    slices pass ``ptr=``) and the ``wp.empty``/``zeros``/... factories."""
    hits = []
    real_init = wp.array.__init__

    def init(self, *args, **kw):
        if kw.get("ptr") is None and len(args) < 6:
            hits.append("wp.array(...)")
            raise AssertionError("device allocation inside an op")
        return real_init(self, *args, **kw)

    def trip(name):
        def call(*a, **k):
            hits.append(f"wp.{name}")
            raise AssertionError(f"wp.{name} inside an op")
        return call

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(wp.array, "__init__", init)
        for name in _ALLOCATORS:
            if hasattr(wp, name):
                mp.setattr(wp, name, trip(name))
        yield hits
    assert hits == []


def _all_op_calls():
    """name -> thunk for every public op and variant (inputs built on first use)."""
    rs = np.random.default_rng(17)
    E, M, num, n, P, K, ps = 4, 12, 8, 10, 6, 5, 3
    ctx = LaunchContext(DEV, None)
    src = v2(_noisy_loops(rs, E, M))
    loops = v2(_noisy_loops(rs, E, n, nan_rows=False))
    sc_m = ops.ArcScratch.alloc(E, M, DEV)
    sc_n = ops.ArcScratch.alloc(E, n, DEV, compact=False)
    act, cnt_n, cnt_p, band = i32([1, 0, 1, 1]), i32([n, 4, 0, 3]), i32([P, 2, 0, P + 1]), i32([1] * E)
    seeds = i32([1, 2, 2147483647, -7])
    ctrl, cnt_k = v2(rs.normal(size=(E * K, 2))), i32([K, 2, 1, 3])
    o_num, co, o_n, o_cs, cs_c, cs_t = ev2(E * num), ei32(E), ev2(E * n), ev2(E * 16), ei32(E), ei32(E)
    xi, tf, pts = ei32(E), ef32(E), v2(rs.normal(size=(E * P, 2)))
    keys, o_p, o_p2 = ef32(E * (P + 2)), ev2(E * P), ev2(E * (P + 2))
    used, o_c, o_cr = ei32(E * P), ei32(E), ev2(E * K * ps)
    flag, scale = ei32(1), f32([1.0, 0.5, 2.0, 1.0])
    return [
        ("arc_resample", lambda: ops.arc_resample(ctx, src, M, num, sc_m, o_num, co)),
        ("arc_resample[active]", lambda: ops.arc_resample(ctx, src, M, num, sc_m, o_num, co, active=act)),
        ("resample_uniform", lambda: ops.resample_uniform(ctx, loops, n, sc_n, o_n, count=cnt_n)),
        ("resample_constant_spacing", lambda: ops.resample_constant_spacing(ctx, loops, n, 0.3, 16, sc_n, o_cs, cs_c, cs_t)),
        ("self_intersections", lambda: ops.self_intersections(ctx, loops, n, xi, count=cnt_n)),
        ("turning_number", lambda: ops.turning_number(ctx, loops, n, tf, count=cnt_n)),
        ("thickness", lambda: ops.thickness(ctx, loops, n, band, tf, count=cnt_n)),
        ("normalize_extent", lambda: ops.normalize_extent(ctx, o_n, n, 1.44)),
        ("normalize_extent[count]", lambda: ops.normalize_extent(ctx, o_n, n, 1.44, count=cnt_n)),
        ("angle_sort", lambda: ops.angle_sort(ctx, pts, P, keys, o_p, count=cnt_p)),
        ("angle_sort[out_stride]", lambda: ops.angle_sort(ctx, pts, P, keys, o_p2, count=cnt_p, out_stride=P + 2)),
        ("sample_counts", lambda: ops.sample_counts(ctx, seeds, 6151, 0, 2, P, o_c)),
        ("sample_grid_points", lambda: ops.sample_grid_points(ctx, seeds, 9781, 0, P, *ops.grid_cells(0.05), 1.0, used, o_p)),
        ("sample_grid_points[array]", lambda: ops.sample_grid_points(ctx, seeds, 9781, 0, P, 10, 0.1, scale, used, o_p)),
        ("closed_catmull_rom", lambda: ops.closed_catmull_rom(ctx, ctrl, K, ps, o_cr)),
        ("closed_catmull_rom[count,endpoint]", lambda: ops.closed_catmull_rom(ctx, ctrl, K, ps, o_cr, count=cnt_k, endpoint=True)),
        ("closed_polyline", lambda: ops.closed_polyline(ctx, ctrl, K, ps, o_cr)),
        ("closed_polyline[active]", lambda: ops.closed_polyline(ctx, ctrl, K, ps, o_cr, active=act)),
        ("closed_polyline[count,endpoint,active]", lambda: ops.closed_polyline(ctx, ctrl, K, ps, o_cr, count=cnt_k, endpoint=True, active=act)),
        ("fill", lambda: ops.fill(ctx, o_cr, (0.0, 1.0))),
        ("any_set", lambda: ops.any_set(ctx, act, flag)),
        ("check_array", lambda: ops.check_array("x", act, (E,), wp.int32, DEV)),
    ]


_OP_NAMES = (
    "arc_resample", "arc_resample[active]", "resample_uniform", "resample_constant_spacing",
    "self_intersections", "turning_number", "thickness", "normalize_extent",
    "normalize_extent[count]", "angle_sort", "angle_sort[out_stride]", "sample_counts",
    "sample_grid_points", "sample_grid_points[array]", "closed_catmull_rom",
    "closed_catmull_rom[count,endpoint]", "closed_polyline", "closed_polyline[active]",
    "closed_polyline[count,endpoint,active]", "fill", "any_set", "check_array",
)


@functools.lru_cache(maxsize=1)
def _op_calls():
    calls = dict(_all_op_calls())
    assert tuple(calls) == _OP_NAMES
    return calls


@pytest.mark.parametrize("name", _OP_NAMES)
def test_op_allocates_nothing(name):
    thunk = _op_calls()[name]
    thunk()                       # compile/load outside the tripwire
    with _no_alloc():
        thunk()


def test_no_alloc_tripwire_positive_control():
    from track_gen._src import warp_pipeline
    E, n = 2, 8
    loops = v2(np.stack([_circle(n)] * E))
    out, cnt = ev2(E * n), i32([n] * E)
    with pytest.raises(AssertionError, match="allocation|wp.empty"):
        with _no_alloc():
            warp_pipeline.resample_uniform(loops, out, n, cnt, device=DEV)
