"""Reproduce the racing geometry guide from real CPU queries.

Run ``python -m viz.render_collision_guide --out-dir /tmp/geometry``. Synthetic
fixtures are deliberately small counterexamples, not generator-yield measurements.
All coordinates use one shared, arbitrary distance unit. Rendering is optional tooling;
Sphinx only copies the selected SVGs and JSON manifest.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle, Circle
import numpy as np
import warp as wp

from track_gen import TrackGenerator
from track_gen.collision import CollisionChecker, DiscChecker
from track_gen.exec import resolve
from track_gen.ops import geometry
from track_gen.pipeline import load_centerlines, presets
from track_gen.results import Layout

BLUE, RED, GREEN, INK = "#2563a6", "#b83e35", "#237344", "#263445"


def _array(a, dtype):
    return wp.array(np.asarray(a), dtype=dtype, device="cpu")


def _closed(p):
    return np.vstack([p, p[0]])


def _diagnostics(p, half_width):
    p = np.asarray(p, np.float32)
    n = len(p)
    spacing = float(np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1).mean())
    band = max(1, int(np.rint(2 * half_width / spacing)))
    count = _array([n], wp.int32)
    src = _array(p, wp.vec2f)
    cross, thick, turn = (wp.zeros(1, dtype=d, device="cpu")
                           for d in (wp.int32, wp.float32, wp.float32))
    ctx = resolve(None, "cpu")
    geometry.self_intersections(ctx, src, n, cross, count=count)
    geometry.thickness(ctx, src, n, _array([band], wp.int32), thick, count=count)
    geometry.turning_number(ctx, src, n, turn, count=count)
    return dict(center_crossings=int(cross.numpy()[0]), thickness=float(thick.numpy()[0]),
                turning=float(turn.numpy()[0]), band=band, mean_spacing=spacing)


def _validity_cases():
    t = np.linspace(0, 2 * np.pi, 192, endpoint=False) + .013
    shapes = [
        ("Valid band", np.c_[np.cos(t), np.sin(t)], .10, 256, .045),
        ("Raw centerline crossing", np.c_[np.sin(t), np.sin(t)*np.cos(t)], .10, 256, .035),
        ("Close nonlocal beads", np.c_[np.cos(t), .08*np.sin(t)], .10, 256, .025),
        ("Tight bend", np.c_[(1+.32*np.cos(5*t))*np.cos(t),
                            (1+.32*np.cos(5*t))*np.sin(t)], .18, 384, .035),
        ("Overlapping road bands", np.c_[np.cos(t), np.sin(t)*(.10+.8*np.cos(t)**2)],
         .16, 256, .03),
        ("Truncated capacity", np.c_[np.cos(t), np.sin(t)], .10, 32, .02),
    ]
    rows = []
    for name, xy, hw, capacity, spacing in shapes:
        spec = presets.track(num_envs=1, device="cpu", generator="external",
                             num_points=len(xy), n_max=capacity, spacing=spacing,
                             half_width=hw, relax=None, on_invalid="keep",
                             checks=(("default", {"validity_border_check": True}),))
        gen = TrackGenerator(spec)
        if name == "Truncated capacity":
            # The cold loader rejects insufficient capacity. Deliberately write the
            # public raw inputs to illustrate the authoritative device truncation bit.
            wp.copy(gen.inputs["centerline.xy"], _array(xy.astype(np.float32), wp.vec2f))
            wp.copy(gen.inputs["centerline.count"], _array([len(xy)], wp.int32))
        else:
            load_centerlines(gen, [xy.astype(np.float32)])
        track = gen.generate()
        n = int(track.count.numpy()[0])
        row = dict(name=name, fixture="synthetic imported centerline", half_width=hw,
                   spacing=spacing, capacity=capacity, count=n, on_invalid="keep",
                   valid=int(track.valid.numpy()[0]), fail_mask=int(track.fail_mask.numpy()[0]),
                   truncated=int(track.truncated.numpy()[0]),
                   input_xy=xy, center=track.center.numpy()[:n, :2], outer=track.outer.numpy()[:n, :2],
                   inner=track.inner.numpy()[:n, :2])
        row.update(_diagnostics(row["center"], hw))
        row["border_crossings"] = sum(_diagnostics(row[k], hw)["center_crossings"]
                                      for k in ("inner", "outer"))
        rows.append(row)
    return rows


def _band(inner, outer):
    inner, outer = np.asarray(inner, np.float32), np.asarray(outer, np.float32)
    assert inner.shape == outer.shape
    def xyz(p):
        return _array(np.c_[p, np.zeros(len(p), np.float32)], wp.vec3f)
    return SimpleNamespace(inner=xyz(inner), outer=xyz(outer), center=xyz((inner+outer)*.5),
                           count=_array([len(inner)], wp.int32), valid=_array([1], wp.int32),
                           layout=Layout(1, len(inner), "cpu"))


def _annulus(inner=1., outer=2., n=192):
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    radial = np.c_[np.cos(t), np.sin(t)]
    return _band(inner*radial, outer*radial)


def _box_inputs(boxes):
    a = np.asarray(boxes, np.float32)  # x, y, yaw, hx, hy
    q = np.c_[np.zeros((len(a), 2)), np.sin(.5*a[:, 2]), np.cos(.5*a[:, 2])]
    return (_array(np.c_[a[:, :2], np.zeros(len(a))], wp.vec3f),
            _array(q, wp.quatf), _array(a[:, 3:], wp.vec2f))


def _contact(c, i=0):
    return {k: (int(getattr(c, k).numpy()[i]) if k in ("oob", "boundary")
                else np.asarray(getattr(c, k).numpy()[i]).tolist())
            for k in ("oob", "distance", "nearest", "normal", "boundary")}


def _query(track, box, method="segments", resolution=128):
    checker = CollisionChecker(track, 1, method=method, sdf_resolution=resolution)
    if method == "sdf":
        checker.bake()
    return checker, _contact(checker.query(*_box_inputs([box])))


def collect_cases():
    """Compute and check the actual values used by every figure; no file output."""
    wp.init()
    validity = _validity_cases()
    polygons = [
        ("Proper crossing", [[-1,-.5], [1,.5], [-1,.5], [1,-.5]]),
        ("Endpoint touch", [[-1,0], [0,0], [0,1], [1,0], [0,0], [0,-1]]),
        ("Collinear overlap", [[-1,0], [1,0], [0,0], [2,0], [2,1], [-1,1]]),
    ]
    predicates = []
    for name, p in polygons:
        diag = _diagnostics(p, .1)
        predicates.append(dict(name=name, points=np.asarray(p), crossings=diag["center_crossings"],
                               thickness=diag["thickness"], band=diag["band"]))
    annulus = _annulus()
    rect = _band([[-1,-.5],[1,-.5],[1,.5],[-1,.5]],
                 [[-2,-1],[2,-1],[2,1],[-2,1]])
    footprints = []
    for name, track, box in [
        ("Clearance", annulus, (1.27,0,np.pi/2,.23,.17)),
        ("Edge-only crossing", annulus, (1.16,0,np.pi/2,.23,.17)),
        ("Corner outside", annulus, (1.95,0,0,.15,.10)),
        ("Exact boundary touch", rect, (1.75,0,0,.25,.125)),
    ]:
        _, contact = _query(track, box)
        footprints.append(dict(name=name, track=track, box=box, contact=contact))
    # An off-road slit, 0.04 units wide, lies between the road's two sides.
    # At R=16 no texel centre lands inside it; bilinear interpolation erases it.
    slit = _band([[-.02,-.45],[.02,-.45],[.02,.45],[-.02,.45]],
                 [[-1,-1],[1,-1],[1,1],[-1,1]])
    box = (0,0,0,.005,.03)
    sdf = []
    for method, res in (("segments", 16), ("sdf", 16), ("sdf", 128)):
        checker, contact = _query(slit, box, method, res)
        row = dict(method=method, resolution=res, contact=contact, track=slit, box=box)
        if method == "sdf":
            lo, hi = checker._sdf_lo.numpy()[0], checker._sdf_hi.numpy()[0]
            row.update(phi=checker._sdf_phi.numpy().reshape(res,res), lo=lo, hi=hi,
                       cell_width=float((hi-lo)[0]/res))
        sdf.append(row)
    discs = []
    checker = DiscChecker(_array([[0,0,0]], wp.vec3f), radius=.125,
                          max_boxes=3, num_envs=1)
    inputs = _box_inputs([(x,0,0,.25,.12) for x in (.5,.375,.3)])
    contact = checker.query(*inputs)
    for i, x in enumerate((.5,.375,.3)):
        discs.append(dict(position=[x,0], radius=.125, half_extents=[.25,.12],
                          hit=int(contact.hit.numpy()[i]), depth=float(contact.depth.numpy()[i])))
    # FrameChecker is the implementation wrapped by public tools.FrameCollision.
    from track_gen._src.collision_frames import FrameChecker
    seq = SimpleNamespace(position=_array([[0,0,0]], wp.vec3f),
                          orientation=_array([[0,0,0,1]], wp.quatf),
                          half_size=_array([1.], wp.float32),
                          count=_array([1], wp.int32), valid=_array([1], wp.int32),
                          layout=Layout(1,1,"cpu"))
    checker = FrameChecker(seq, 1, radius=.1, frame_thickness=.1, frame_depth=.1, window=1)
    checker.bind_window(_array([0], wp.int32))
    frames = []
    for p in ([0,0,0], [0,1.05,0], [0,0,1.05]):
        checker.bind_inputs(_array([p], wp.vec3f))
        contact = checker.query()
        frames.append(dict(position=p, hit=int(contact.hit.numpy()[0]),
                           depth=float(contact.depth.numpy()[0])))
    assert validity[0]["valid"] == 1 and validity[-1]["fail_mask"] & 128
    assert [p["crossings"] for p in predicates] == [1,0,0]
    assert [p["contact"]["oob"] for p in footprints] == [0,1,1,1]
    assert footprints[1]["contact"]["distance"] == 0.
    assert [p["contact"]["oob"] for p in sdf] == [1,0,1]
    return dict(validity=validity, predicates=predicates, footprints=footprints,
                sdf=sdf, discs=discs, frames=frames)


def _axes(ax):
    ax.set_aspect("equal")
    ax.set_xlabel("x [distance units]", fontsize=9)
    ax.set_ylabel("y [distance units]", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)


def _draw_band(ax, inner, outer, center=None):
    for i in range(len(inner)):
        j = (i+1) % len(inner)
        ax.add_patch(Polygon([inner[i], inner[j], outer[j], outer[i]],
                             facecolor="#e2eaf1", edgecolor="none"))
    for p in (inner, outer):
        ax.plot(*_closed(p).T, color=BLUE, lw=1.)
    if center is not None:
        ax.plot(*_closed(center).T, color=INK, lw=.8, ls="--")
        ax.scatter(*center.T, s=3, color=INK)


def _draw_source(ax, track):
    _draw_band(ax, track.inner.numpy()[:, :2], track.outer.numpy()[:, :2])


def _draw_box(ax, box, color):
    x,y,yaw,hx,hy = box
    p = np.array([[hx,hy],[-hx,hy],[-hx,-hy],[hx,-hy]])
    c,s = np.cos(yaw), np.sin(yaw)
    p = p @ np.array([[c,s],[-s,c]]) + [x,y]
    ax.add_patch(Polygon(p, facecolor=color, alpha=.2, edgecolor=color, lw=2))
    ax.scatter(*p.T, s=13, color=color)


def render(cases, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10,
                         "svg.fonttype":"none", "figure.facecolor":"white"})
    fig, axs = plt.subplots(3,2,figsize=(9,11.6), layout="constrained")
    for ax, row in zip(axs.flat, cases["validity"]):
        if row["truncated"]:
            ax.plot(*_closed(row["input_xy"]).T, color="#9daab8", ls=":", lw=1.5)
            ax.text(-1.35,.95,"dotted: supplied loop",color="#667585",fontsize=9)
        _draw_band(ax, row["inner"], row["outer"], row["center"])
        ax.set_title(row["name"], color=GREEN if row["valid"] else RED, fontsize=11)
        ax.text(.02,.02,f"valid={row['valid']}   fail=0x{row['fail_mask']:02x}\n"
                f"thickness={row['thickness']:.3f}   half-width={row['half_width']:.2f}\n"
                f"center crossings={row['center_crossings']}   count={row['count']}",
                transform=ax.transAxes, fontsize=9,
                bbox=dict(facecolor="white",edgecolor="none",alpha=.9))
        ax.set_xlim(-1.55,1.55)
        ax.set_ylim(-1.55,1.55)
        _axes(ax)
    fig.suptitle("Validity checks inspect different parts of the geometry", fontsize=15)
    fig.savefig(out_dir/"geometry-validity.svg")
    fig.savefig(out_dir/"geometry-validity.png", dpi=130)
    plt.close(fig)

    fig, axs = plt.subplots(1,3,figsize=(10,3.4),layout="constrained")
    for ax,row in zip(axs,cases["predicates"]):
        p = row["points"]
        ax.plot(*_closed(p).T,color=BLUE,lw=2)
        ax.scatter(*p.T,color=INK,s=24)
        ax.set_title(f"{row['name']}\nproper crossings = {row['crossings']}\n"
                     f"sampled thickness = {row['thickness']:.3f}",fontsize=11)
        ax.margins(.25)
        _axes(ax)
    fig.savefig(out_dir/"geometry-crossing-predicates.svg")
    fig.savefig(out_dir/"geometry-crossing-predicates.png", dpi=130)
    plt.close(fig)

    fig, axs = plt.subplots(2,2,figsize=(9,7),layout="constrained")
    for ax,row in zip(axs.flat,cases["footprints"]):
        _draw_source(ax,row["track"])
        c=row["contact"]
        color=RED if c["oob"] else GREEN
        _draw_box(ax,row["box"],color)
        ax.scatter(*c["nearest"],color=INK,s=22,zorder=5)
        ax.arrow(*c["nearest"],*(np.asarray(c["normal"])*.14),width=.004,color=INK,zorder=5)
        ax.set_xlim(.7,2.25); ax.set_ylim(-.62,.62)
        ax.set_title(f"{row['name']}\noob={c['oob']}   distance={c['distance']:+.4f}",fontsize=11)
        _axes(ax)
    fig.suptitle("Exact segment queries: read oob separately from distance",fontsize=14)
    fig.savefig(out_dir/"geometry-footprints.svg")
    fig.savefig(out_dir/"geometry-footprints.png", dpi=130)
    plt.close(fig)

    fig, axs = plt.subplots(1,3,figsize=(10,4),layout="constrained")
    for ax,row in zip(axs,cases["sdf"]):
        if row["method"] == "sdf":
            lo,hi=row["lo"],row["hi"]
            ax.imshow(row["phi"],origin="lower",extent=(*[lo[0],hi[0]],lo[1],hi[1]),
                      cmap="RdBu",vmin=-.08,vmax=.08,interpolation="bilinear")
            ax.contour(np.linspace(lo[0],hi[0],row["resolution"],endpoint=False)+(hi[0]-lo[0])/row["resolution"]/2,
                       np.linspace(lo[1],hi[1],row["resolution"],endpoint=False)+(hi[1]-lo[1])/row["resolution"]/2,
                       row["phi"],levels=[0],colors=["white"],linewidths=1)
            tx=lo[0]+(np.arange(row["resolution"])+.5)*(hi[0]-lo[0])/row["resolution"]
            ty=lo[1]+(np.arange(row["resolution"])+.5)*(hi[1]-lo[1])/row["resolution"]
            xx,yy=np.meshgrid(tx[(tx>-.2)&(tx<.2)],ty[(ty>-.52)&(ty<.52)])
            ax.scatter(xx,yy,s=2,color="#8595a8",alpha=.55)
            title=f"SDF {row['resolution']} × {row['resolution']}\ncell={row['cell_width']:.4f}"
        else:
            _draw_source(ax,row["track"])
            title="Exact boundary segments\noff-road slit width=0.0400"
        _draw_box(ax,row["box"],INK)
        inn=row["track"].inner.numpy()[:,:2]
        ax.plot(*_closed(inn).T,color=RED,lw=1.3,ls="--")
        c=row["contact"]
        ax.set_title(f"{title}\noob={c['oob']}   distance={c['distance']:+.4f}",fontsize=10)
        ax.set_xlim(-.2,.2);ax.set_ylim(-.52,.52)
        _axes(ax)
    fig.suptitle("A narrow off-road gap can disappear between texel centres",fontsize=14)
    fig.savefig(out_dir/"geometry-sdf.svg")
    fig.savefig(out_dir/"geometry-sdf.png", dpi=130)
    plt.close(fig)

    fig,axs=plt.subplots(1,2,figsize=(10,4.5),layout="constrained")
    ax=axs[0]
    for i,row in enumerate(cases["discs"]):
        y=.55*i
        ax.add_patch(Circle((0,y),.125,facecolor="#e2eaf1",edgecolor=BLUE))
        x=row["position"][0]
        _draw_box(ax,(x,y,0,.25,.12),RED if row["hit"] else GREEN)
        ax.text(.8,y,f"hit={row['hit']}\ndepth={row['depth']:.3f}",va="center",fontsize=10)
    ax.set_title("Planar discs against box footprints\ngrazing counts as hit",fontsize=11)
    ax.set_xlim(-.2,1.3);ax.set_ylim(-.2,1.4);_axes(ax)
    ax=axs[1]
    for x,y,w,h in [(-1.1,-1.1,.1,2.2),(1.,-1.1,.1,2.2),(-1.1,-1.1,2.2,.1),(-1.1,1.,2.2,.1)]:
        ax.add_patch(Rectangle((x,y),w,h,facecolor=BLUE))
    for row in cases["frames"]:
        y,z=row["position"][1:]
        ax.add_patch(Circle((y,z),.1,facecolor=RED if row["hit"] else GREEN,alpha=.45))
        ax.text(y+.13,z-.12,f"hit={row['hit']}, depth={row['depth']:.2f}",fontsize=8)
    ax.set_title("3D frame against an agent sphere\nfront view (all sphere centres at x=0)",fontsize=11)
    ax.set_xlim(-1.4,1.9);ax.set_ylim(-1.4,1.4);_axes(ax)
    ax.set_xlabel("gate-local y [distance units]");ax.set_ylabel("gate-local z [distance units]")
    fig.savefig(out_dir/"geometry-obstacles.svg")
    fig.savefig(out_dir/"geometry-obstacles.png", dpi=130)
    plt.close(fig)

    fig, axs = plt.subplots(1,2,figsize=(10,4),layout="constrained")
    t=np.linspace(0,2*np.pi,64,endpoint=False)
    p=np.c_[np.cos(t),np.sin(t)]
    diag=_diagnostics(p,.1)
    ax=axs[0]
    ax.plot(*_closed(p).T,color="#c7d0d9",lw=1)
    ax.scatter(*p.T,color="#9daab8",s=12)
    band=diag["band"]
    selected=[i%len(p) for i in range(-band,band+1)]
    ax.scatter(*p[selected].T,color=BLUE,s=38)
    ax.scatter(*p[0],color=RED,s=45,zorder=5)
    for i in selected:
        ax.annotate(str(i),p[i],xytext=(8,2),textcoords="offset points",fontsize=9)
    ax.set_xlim(.72,1.18);ax.set_ylim(-.5,.5)
    ax.set_title(f"Index neighbourhood: band={band}\nmean edge length={diag['mean_spacing']:.4f}",fontsize=11)
    _axes(ax)
    ax=axs[1]
    for x in (-.115,.115):
        ax.add_patch(Circle((x,0),.1,facecolor="#e2eaf1",edgecolor=BLUE))
        ax.add_patch(Circle((x,0),.115,fill=False,edgecolor=RED,linestyle="--"))
        ax.scatter(x,0,color=INK,s=22)
    ax.annotate("",xy=(-.115,.15),xytext=(.115,.15),arrowprops=dict(arrowstyle="<->",color=INK))
    ax.text(0,.17,"solver target = 0.230",ha="center",fontsize=10)
    ax.text(0,-.20,"w=0.100  |  margin=0.15\nvalidity pair floor = 0.196 (tol=0.02)",
            ha="center",fontsize=10)
    ax.set_xlim(-.32,.32);ax.set_ylim(-.27,.25)
    ax.set_title("Margin expands the separation target\nblue: road radius; dashed: solver radius",fontsize=11)
    _axes(ax)
    fig.savefig(out_dir/"geometry-constraints.svg")
    fig.savefig(out_dir/"geometry-constraints.png",dpi=130)
    plt.close(fig)


def _json_value(value):
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,dict):
        return {k:_json_value(v) for k,v in value.items() if k not in ("track", "phi")}
    if isinstance(value,list): return [_json_value(v) for v in value]
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir",type=Path,default=Path("docs/assets/racing"))
    args=parser.parse_args()
    cases=collect_cases()
    render(cases,args.out_dir)
    manifest=dict(source_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
                  command="python -m viz.render_collision_guide --out-dir DIR", device="cpu",
                  warp_version=wp.__version__, numpy_version=np.__version__,
                  units="arbitrary shared distance units", selection="synthetic counterexamples; no yield claim",
                  cases=_json_value(cases))
    (args.out_dir/"geometry-manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(f"Wrote six SVGs, PNG previews and geometry-manifest.json to {args.out_dir}")


if __name__ == "__main__":
    main()
