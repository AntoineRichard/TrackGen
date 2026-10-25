"""Render data-layout/lifecycle diagrams and a real external-geometry example.

Run ``python -m viz.render_data_guide --out-dir /tmp/trackgen-data-guide``.
The first two figures are explicitly schematic; the third uses current public APIs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np

INK = "#23334b"
BLUE = "#2563eb"
TEAL = "#087f8c"
PALE = "#e5efff"
AMBER = "#b76a0a"


def _save(fig, path):
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_layout(path):
    fig, ax = plt.subplots(figsize=(10, 4.3))
    ax.set(xlim=(-1.35, 10.9), ylim=(-.8, 3.9))
    ax.axis("off")
    ax.text(-1.2, 3.48, "One flat array; fixed-capacity rows", fontsize=17, weight="bold", color=INK)
    ax.text(-1.2, 3.08, "Schematic: E = 3, stride = 8. Each cell is one vec3f point.", fontsize=11, color=INK)
    for e, n in enumerate((5, 3, 0)):
        y = 2.1 - e
        ax.text(-.25, y + .34, f"env {e}", ha="right", fontsize=12, weight="bold", color=INK)
        for i in range(8):
            real = i < n
            color = PALE if real else "#f0f1f3"
            ax.add_patch(Rectangle((i, y), .94, .72, facecolor=color, edgecolor="white"))
            ax.text(i+.47, y+.41, f"p{i}" if real else "NaN", ha="center", fontsize=11,
                    color=BLUE if real else "#6b7280")
            ax.text(i+.47, y+.12, str(e*8+i), ha="center", fontsize=8, color="#657086")
        ax.text(8.25, y+.46, f"count={n}   valid={int(n > 0)}", fontsize=11, color=INK)
        ax.text(8.25, y+.12, "fail_mask=0" if n else "fail_mask ≠ 0", fontsize=10,
                color=INK if n else AMBER)
    ax.text(0, -.66, "Read count to stop. Skip invalid rows. Padding is not geometry.", fontsize=12, color=INK)
    _save(fig, path)


def render_lifecycle(path):
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.set(xlim=(0, 12), ylim=(0, 5))
    ax.axis("off")
    ax.text(.15, 4.65, "Stable storage, borrowed results", fontsize=17, weight="bold", color=INK)
    boxes = [(0.2, 2.1, 2.65, "Construct once", "allocate buffers\n+ warm-up generation"),
             (4.0, 2.1, 3.0, "generate()", "write the same result\ninto the same buffers"),
             (8.2, 2.1, 3.3, "Simulator steps", "read geometry; query\nprogress and collision")]
    for x, y, w, title, body in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,1.25,boxstyle="round,pad=.1",facecolor=PALE,edgecolor="none"))
        ax.text(x+w/2,y+.9,title,ha="center",fontsize=13,weight="bold",color=INK)
        ax.text(x+w/2,y+.35,body,ha="center",va="center",fontsize=10.5,color=INK)
    for start,end in [((2.98,2.72),(3.8,2.72)),((7.15,2.72),(8.0,2.72))]:
        ax.add_patch(FancyArrowPatch(start,end,arrowstyle="-|>",mutation_scale=15,color=BLUE,lw=1.8))
    ax.add_patch(FancyArrowPatch((9.9,2.0),(5.6,2.0),connectionstyle="arc3,rad=-.5",
                                arrowstyle="-|>",mutation_scale=15,color=BLUE,lw=1.8))
    ax.text(7.8,.75,"Regenerate selected envs when needed",ha="center",fontsize=11,color=BLUE)
    ax.add_patch(FancyBboxPatch((4.1,3.8),2.8,.55,boxstyle="round,pad=.1",facecolor="#e2f3ef",edgecolor="none"))
    ax.text(5.5,4.08,"clone(): independent snapshot",ha="center",va="center",fontsize=10.5,color=TEAL)
    ax.add_patch(FancyArrowPatch((5.5,3.45),(5.5,3.72),arrowstyle="-|>",mutation_scale=12,color=TEAL,lw=1.5))
    ax.text(.15,.16,"A second Python reference still points to the borrowed result. Copy to retain its previous contents.",
            fontsize=10.5,color=INK)
    _save(fig,path)


def imported_example():
    from track_gen import TrackGenerator, GateGenerator
    from track_gen.pipeline import load_centerlines, presets
    t=np.linspace(0,2*np.pi,64,endpoint=False)
    r=1+.12*np.cos(3*t)
    xy=np.column_stack((2*r*np.cos(t),1.25*r*np.sin(t))).astype(np.float32)
    road=TrackGenerator(presets.track(num_envs=1,device="cpu",generator="external",
                                      num_points=128,n_max=512,spacing=.04,half_width=.05))
    load_centerlines(road,[xy])
    track=road.generate()
    gates_gen=GateGenerator(presets.gates(num_envs=1,device="cpu",generator="external",
                                          source="stations",num_points=128,gates=12,
                                          gate_width=.15))
    load_centerlines(gates_gen,[xy])
    gates=gates_gen.generate()
    assert track.valid.numpy()[0] == 1 and gates.valid.numpy()[0] == 1
    n,g=int(track.count.numpy()[0]),int(gates.count.numpy()[0])
    return xy,track,gates,n,g


def render_import(path):
    xy,track,gates,n,g=imported_example()
    fig,axes=plt.subplots(1,3,figsize=(12,4.5),sharex=True,sharey=True)
    close=lambda a: np.vstack((a,a[0]))
    for ax,title in zip(axes,("1. Supply your loop","2. Resample and add a road","3. Place gates along the route")):
        ax.set_title(title,fontsize=12,weight="bold",color=INK,pad=13)
        ax.set_aspect("equal");ax.set(xlim=(-2.6,2.6),ylim=(-2.0,2.0))
        ax.set_xticks([]);ax.set_yticks([])
        for spine in ax.spines.values():spine.set_visible(False)
    p=close(xy);axes[0].plot(p[:,0],p[:,1],"o-",color=BLUE,ms=2,lw=1.5)
    c=track.center.numpy()[:n,:2];o=track.outer.numpy()[:n,:2];i=track.inner.numpy()[:n,:2]
    for j in range(n):
        k=(j+1)%n;quad=np.array((o[j],o[k],i[k],i[j]))
        axes[1].fill(quad[:,0],quad[:,1],color=PALE,ec="none")
    for boundary in (o,i):
        p=close(boundary);axes[1].plot(p[:,0],p[:,1],color=BLUE,lw=1.2)
    p=close(c);axes[1].plot(p[:,0],p[:,1],color=INK,lw=.7,ls="--")
    p=close(xy);axes[2].plot(p[:,0],p[:,1],color="#b6becb",lw=1)
    pos=gates.position.numpy()[:g,:2];left=gates.left.numpy()[:g,:2];right=gates.right.numpy()[:g,:2]
    for j,(a,b) in enumerate(zip(left,right)):
        axes[2].plot([a[0],b[0]],[a[1],b[1]],color=TEAL,lw=3)
        axes[2].annotate(str(j),(pos[j,0],pos[j,1]),xytext=(5,5),textcoords="offset points",fontsize=8,color=INK)
    fig.text(.5,.04,f"Actual public API output · 64 supplied points · {n} road samples · {g} gates · both valid · coordinate units",
             ha="center",fontsize=10,color=INK)
    fig.tight_layout(rect=(0,.08,1,1))
    _save(fig,path)
    return {"input_count":len(xy),"road_count":n,"gate_count":g,"valid":[1,1],
            "road_config":{"num_points":128,"n_max":512,"spacing":.04,"half_width":.05},
            "gate_config":{"source":"stations","num_points":128,"gates":12,"gate_width":.15},
            "input_formula":"x=2*(1+.12*cos(3t))*cos(t), y=1.25*(1+.12*cos(3t))*sin(t), t=2*pi*i/64"}


def render_gate_course(path):
    """A finished 3D course: square openings from actual positions and quaternions."""
    from track_gen import GateGenerator
    from track_gen.pipeline import presets
    params={"polar_num_knots":10,"polar_radial_jitter":.25,"polar_angular_jitter":.12}
    elevation={"z_min":.2,"z_max":1.2,"z_base":.7,"z_noise_amplitude":.4,
               "z_noise_harmonics":2}
    config=dict(num_envs=1,device="cpu",generator="polar",generator_params=params,
                size=2.,gate_width=.25,gate_radius=.05,gate_align="full_tangent",
                elevation=("noise",elevation))
    gates=GateGenerator(presets.gates(**config),seeds=23).generate()
    assert gates.valid.numpy()[0] == 1
    n=int(gates.count.numpy()[0]);pos=gates.position.numpy()[:n]
    quat=gates.orientation.numpy()[:n];half=gates.half_size.numpy()[:n]
    assert n == 10 and np.isfinite(pos).all() and np.isfinite(quat).all()
    rotate=lambda q,v: v+2*np.cross(q[:3],np.cross(q[:3],v)+q[3]*v)
    frames=[]
    for c,q,h in zip(pos,quat,half):
        left=rotate(q,np.array([0.,1.,0.]));up=rotate(q,np.array([0.,0.,1.]))
        frame=np.array([c+h*(u*left+v*up) for u,v in [(-1,-1),(1,-1),(1,1),(-1,1),(-1,-1)]])
        frames.append(frame)
    fig=plt.figure(figsize=(10,5.8));ax=fig.add_subplot(111,projection="3d")
    ordered=np.vstack((pos,pos[0]));ax.plot(*ordered.T,color="#9aa9bb",lw=1.2,ls="--")
    for j,frame in enumerate(frames):
        ax.plot(*frame.T,color=AMBER if j==0 else TEAL,lw=2.6)
        ax.text(*(pos[j]+np.array([0.,0.,.17])),str(j),fontsize=9,color=INK)
    all_points=np.concatenate(frames);lo=all_points.min(axis=0);hi=all_points.max(axis=0)
    span=hi-lo;pad=.12*max(span)
    ax.set(xlim=(lo[0]-pad,hi[0]+pad),ylim=(lo[1]-pad,hi[1]+pad),
           zlim=(0,hi[2]+pad),xlabel="x [units]",ylabel="y [units]",zlabel="z [units]")
    ax.set_box_aspect((span[0]+2*pad,span[1]+2*pad,hi[2]+pad))
    ax.view_init(elev=24,azim=-62)
    ax.set_title("A finished 3D gate course",fontsize=17,weight="bold",color=INK,pad=8)
    ax.tick_params(labelsize=8)
    fig.text(.5,.035,"Actual GateSequence poses · 10 gates · seed 23 · valid · dashed line shows passing order",
             ha="center",fontsize=10,color=INK)
    fig.tight_layout(rect=(0,.07,1,1));_save(fig,path)
    return {"seed":23,"valid":1,"count":n,"config":config,
            "render":"Square openings from position/orientation/half_size; line shows order, not a spline."}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir",type=Path,required=True)
    out=parser.parse_args().out_dir;out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","text.color":INK})
    render_layout(out/"data-layout.png");render_lifecycle(out/"data-lifecycle.png")
    record=render_import(out/"data-import.png")
    gate_record=render_gate_course(out/"data-gate-course.png")
    import track_gen
    (out/"data-manifest.json").write_text(json.dumps({"source_version":track_gen.__version__,
        "command":"python -m viz.render_data_guide --out-dir OUT","schematics":["data-layout.png","data-lifecycle.png"],
        "import":record,"gate_course":gate_record},indent=2)+"\n")
    print(out)


if __name__ == "__main__":
    main()
