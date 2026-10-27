#!/usr/bin/env python3
"""Render short, reproducible track/gate relaxation clips from current TrackGen output.

Every frame is a fresh fixed-seed generation at a solver-prefix budget. No geometry is
interpolated. Track frames show the pipeline's re-uniformised/inflated view of each
prefix, rather than the solver's internal bead buffers. The total-budget-independent
Chebyshev schedule and separation cadence make each main-solve budget a prefix of the
same solve. Taubin prefixes have spacing polish OFF; only after all Taubin passes do
spacing-polish prefixes run. Gate rounds use in-place Gauss-Seidel pair corrections.

Examples (optional Matplotlib + ffmpeg or imageio-ffmpeg; never a Sphinx dependency)::

    python -m viz.render_relaxation_video --device cpu --out-dir /tmp/track-clips
    python -m viz.render_relaxation_video --gates --device cpu --out-dir /tmp/gate-clips
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from viz.racing_theme import (save_figure, get_palette, geometry_digest,
                              rendered_geometry, render_variants, video_canvas_size, add_video_branding)
import numpy as np
import warp as wp

from track_gen import GateGenConfig, GateGenerator, TrackGenConfig, TrackGenerator

REGIME = dict(half_width=.5, scale=10., spacing=.30, N_max=384)
GATE_REGIME = dict(generator="bezier", gate_radius=.13, gate_width=.16, max_gates=32)


def verify_prefix_schedule(sweeps):
    """Fail if the source's acceleration schedule stops being prefix-invariant."""
    from track_gen._src.warp_relax import _cheby_schedule
    full = _cheby_schedule(sweeps, .98, 8)
    for k in range(sweeps + 1):
        assert _cheby_schedule(k, .98, 8) == full[:k]
    # Cache refresh uses step_i % sep_every; neither cadence depends on total budget.
    for k in range(sweeps + 1):
        assert [i for i in range(k) if i % 20 == 0] == [i for i in range(sweeps) if i % 20 == 0 and i < k]


def _numpy(result, fields, batch):
    count = result.count.numpy().astype(int)
    stride = result.layout.stride
    data = {name:getattr(result,name).numpy().reshape(batch,stride,3)[...,:2]
            for name in fields}
    data.update(count=count, valid=result.valid.numpy().astype(int),
                fail_mask=result.fail_mask.numpy().astype(int))
    for e,n in enumerate(count):
        assert n > 1
        assert all(np.isfinite(data[name][e,:n]).all() for name in fields)
    return data


def _track(seed, batch, device, **overrides):
    cfg = TrackGenConfig(num_envs=batch,device=device,on_invalid="keep",**REGIME,**overrides)
    # An integer expands to seed + arange(batch), with an owned seed buffer.
    result = TrackGenerator(cfg,seeds=seed).generate()
    data = _numpy(result,("center","outer","inner"),batch)
    data["config"] = dataclasses.asdict(cfg)
    return data


def capture_track_states(*, batch=24, sweeps=50, smooth_passes=5, polish=10,
                         seed=0, device="cpu"):
    verify_prefix_schedule(sweeps)
    raw = _track(seed,batch,device,relax_enable=False)
    repeat = _track(seed,batch,device,relax_enable=False)
    assert np.array_equal(raw["center"],repeat["center"],equal_nan=True)
    raw.update(phase="raw", iteration=0)
    states = [raw]
    for k in range(1,sweeps+1):
        data = _track(seed,batch,device,relax_iters=k,
                      relax_smooth_passes=0,relax_smooth_spacing_iters=0)
        data.update(phase="sweep",iteration=k)
        states.append(data)
    for k in range(1,smooth_passes+1):
        data = _track(seed,batch,device,relax_iters=sweeps,
                      relax_smooth_passes=k,relax_smooth_spacing_iters=0)
        data.update(phase="Taubin",iteration=k)
        states.append(data)
    for k in range(1,polish+1):
        data = _track(seed,batch,device,relax_iters=sweeps,
                      relax_smooth_passes=smooth_passes,relax_smooth_spacing_iters=k)
        data.update(phase="polish",iteration=k)
        states.append(data)
    return states, dict(seed_start=seed,batch=batch,on_invalid="keep",device=device,
                        sweeps=sweeps,smooth_passes=smooth_passes,polish=polish,
                        prefix_verified=True,config=states[-1]["config"],
                        display="re-uniformised and inflated view of each solver prefix")


def _gate(k,seed,batch,device):
    cfg=GateGenConfig(num_envs=batch,device=device,on_invalid="keep",
                      gate_solve_iters=k,**GATE_REGIME)
    result=GateGenerator(cfg,seeds=seed).generate()
    data=_numpy(result,("position","left","right"),batch)
    data.update(phase="round",iteration=k,config=dataclasses.asdict(cfg))
    return data


def capture_gate_states(*,batch=24,rounds=16,seed=0,device="cpu"):
    states=[_gate(k,seed,batch,device) for k in range(rounds+1)]
    repeat=_gate(0,seed,batch,device)
    assert np.array_equal(states[0]["position"],repeat["position"],equal_nan=True)
    # A longer budget has the same initial input; the round-index tie break and
    # early exit read no total-budget-dependent quantities (warp_gate.py).
    long=_gate(rounds+16,seed,batch,device)
    return states,dict(seed_start=seed,batch=batch,on_invalid="keep",device=device,
                       rounds=rounds,config=states[-1]["config"],prefix_verified=True,
                       long_state=long)


def _min_distance(data,e,gates=False):
    key="position" if gates else "center"
    p=data[key][e,:data["count"][e]]
    n=len(p)
    d=np.linalg.norm(p[:,None,:]-p[None,:,:],axis=-1)
    ii=np.arange(n)
    diff=np.abs(ii[:,None]-ii[None,:])
    band=0 if gates else max(1,int(np.rint(2*REGIME["half_width"] / np.linalg.norm(np.roll(p,-1,axis=0)-p,axis=1).mean())))
    mask=np.minimum(diff,n-diff)>band
    return float(d[mask].min()) if mask.any() else float("inf")


def _choose(states,gates,panels=3):
    raw,final=states[0],states[-1]
    batch=len(raw["count"])
    scores=sorted(range(batch),key=lambda e:_min_distance(raw,e,gates))
    successes=[e for e in scores if final["valid"][e]]
    failures=[e for e in scores if not final["valid"][e]]
    chosen=successes[:panels-1]
    if failures: chosen+=failures[:1]
    else: chosen+=successes[panels-1:panels]
    assert len(chosen)==panels, "increase --batch to find enough finite examples"
    assert any(np.max(np.abs(raw["position" if gates else "center"][e,:raw["count"][e]]-
                            final["position" if gates else "center"][e,:final["count"][e]]))>1.e-4 for e in chosen)
    return chosen


def _limits(states,envs,gates):
    key="position" if gates else "center"
    limits=[]
    for e in envs:
        p=np.vstack([s[key][e,:s["count"][e]] for s in states])
        lo,hi=p.min(0),p.max(0)
        center=(lo+hi)/2
        radius=max(hi-lo)*.58 + (.15 if gates else REGIME["half_width"])
        limits.append((center-radius,center+radius))
    return limits


def _frame(path,data,envs,limits,metadata,gates):
    fig,axs=plt.subplots(1,len(envs),figsize=video_canvas_size(12,100),dpi=100,facecolor="white")
    axs=np.atleast_1d(axs)
    for ax,e,(lo,hi) in zip(axs,envs,limits):
        n=data["count"][e]
        p=data["position" if gates else "center"][e,:n]
        closed=np.vstack([p,p[0]])
        if gates:
            ax.plot(*closed.T,color="#5d6874",ls="--",lw=.8)
            for center,left,right in zip(p,data["left"][e,:n],data["right"][e,:n]):
                ax.add_patch(plt.Circle(center,GATE_REGIME["gate_radius"],fill=False,
                                       color="#738499",lw=.8))
                ax.plot([left[0],right[0]],[left[1],right[1]],color="#2563a6",lw=1.5)
            ax.scatter(*p.T,color="#263445",s=12)
        else:
            for key in ("inner","outer"):
                b=data[key][e,:n]
                ax.plot(*np.vstack([b,b[0]]).T,color="#2563a6",lw=1.)
            ax.plot(*closed.T,color="#263445",ls="--",lw=.9)
        valid=data["valid"][e];fail=data["fail_mask"][e]
        distance = _min_distance(data,e,gates)
        distance_label = f"{distance:.7f}" if gates else f"{distance:.4f}"
        panel_title = (f"seed {metadata['seed_start']+e}  |  env {e}\n"
                       f"valid={valid}  fail=0x{fail:02x}  count={n}")
        if not gates:
            panel_title += f"\nmin nonlocal bead distance {distance_label}"
        ax.set_title(panel_title,fontsize=10,
                     color="#237344" if valid else "#b83e35")
        if gates:
            ax.text(.03,.03,f"min pair distance\n{distance_label}",
                    transform=ax.transAxes,fontsize=9,
                    bbox=dict(facecolor="white",edgecolor="none",alpha=.85))
        ax.set_xlim(lo[0],hi[0]);ax.set_ylim(lo[1],hi[1]);ax.set_aspect("equal")
        ax.set_xlabel("x [distance units]",fontsize=9);ax.set_ylabel("y [distance units]",fontsize=9)
        ax.tick_params(labelsize=8);ax.spines[["top","right"]].set_visible(False)
    phase=data["phase"];k=data["iteration"]
    title=f"Gate sphere separation — round {k}" if gates else f"Track XPBD — {phase} {k}"
    fig.suptitle(title,fontsize=16,fontweight="bold",y=.98)
    footer=("Actual rounds; radius=0.13, target=0.2600000. Selected examples; no yield claim."
            if gates else "Each prefix shown after uniform resampling + inflation. Selected examples; no yield claim.")
    fig.text(.04,.025,footer,ha="left",fontsize=9,color="#344253")
    add_video_branding(fig)
    fig.subplots_adjust(left=.05,right=.99,top=.79,bottom=.14,wspace=.25)
    save_figure(fig, path,facecolor="white")
    frame_size=list(fig.canvas.get_width_height())
    plt.close(fig)
    return frame_size


def _ffmpeg(explicit=None):
    if explicit: return explicit
    found=shutil.which("ffmpeg")
    if found: return found
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        return get_ffmpeg_exe()
    except ImportError as exc:
        raise RuntimeError("Install optional ffmpeg or imageio-ffmpeg, or pass --ffmpeg") from exc


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gates",action="store_true")
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--out-dir",type=Path,default=Path("docs/_static"))
    parser.add_argument("--ffmpeg")
    parser.add_argument("--batch",type=int,default=24)
    parser.add_argument("--seed",type=int,default=0)
    parser.add_argument("--sweeps",type=int,default=50)
    parser.add_argument("--rounds",type=int,default=16)
    parser.add_argument("--smooth-passes",type=int,default=5)
    parser.add_argument("--polish",type=int,default=10)
    parser.add_argument("--theme",choices=("light","dark","both"),default="light")
    args=parser.parse_args(argv)
    encoder=_ffmpeg(args.ffmpeg)
    wp.init()
    if args.gates:
        states,meta=capture_gate_states(batch=args.batch,rounds=args.rounds,seed=args.seed,device=args.device)
    else:
        states,meta=capture_track_states(batch=args.batch,sweeps=args.sweeps,
                                         smooth_passes=args.smooth_passes,polish=args.polish,
                                         seed=args.seed,device=args.device)
    envs=_choose(states,args.gates)
    limits=_limits(states,envs,args.gates)
    stem="relaxation-gates-iterations" if args.gates else "relaxation-iterations"
    args.out_dir.mkdir(parents=True,exist_ok=True)
    fps=3 if args.gates else 8
    long=meta.pop("long_state",None)
    meta.update(source_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
                warp_version=wp.__version__, numpy_version=np.__version__,
                command=("python -m viz.render_relaxation_video " +
                         ("--gates " if args.gates else "") +
                         f"--device {args.device} --batch {args.batch} --seed {args.seed} " +
                         f"--sweeps {args.sweeps} --rounds {args.rounds} " +
                         f"--smooth-passes {args.smooth_passes} --polish {args.polish} --out-dir DIR"),
                selection="smallest raw pair distances among final-valid examples, plus final-invalid example when available",
                selected_envs=envs,fps=fps,
                selected_seeds=[args.seed+e for e in envs],
                frames=[dict(phase=s["phase"],iteration=s["iteration"],
                             envs=[dict(env=e,count=int(s["count"][e]),valid=int(s["valid"][e]),
                                        fail_mask=int(s["fail_mask"][e]),
                                        min_distance=_min_distance(s,e,args.gates)) for e in envs]) for s in states])
    if long is not None:
        meta["max_displacement_to_longer_budget"]={str(e):float(np.linalg.norm(
            states[-1]["position"][e,:states[-1]["count"][e]]-
            long["position"][e,:long["count"][e]],axis=1).max()) for e in envs}
    capture_hash=geometry_digest(states)
    def variant(out):
        variant_meta=dict(meta)
        with tempfile.TemporaryDirectory(prefix="trackgen-video-") as tmp:
            tmp=Path(tmp)
            idx=0
            frame_size=None
            for i,s in enumerate(states):
                size=_frame(tmp/f"frame_{idx:04d}.png",s,envs,limits,meta,args.gates)
                if frame_size is not None and size!=frame_size:
                    raise AssertionError("iteration frames changed their native canvas size")
                frame_size=size
                idx+=1
                if i==0:
                    for _ in range(fps):
                        shutil.copyfile(tmp/"frame_0000.png",tmp/f"frame_{idx:04d}.png");idx+=1
            final=tmp/f"frame_{idx-1:04d}.png"
            shutil.copyfile(final,out/f"{stem}-poster.png")
            for _ in range(2*fps):
                shutil.copyfile(final,tmp/f"frame_{idx:04d}.png");idx+=1
            target=tmp/f"{stem}.mp4"
            subprocess.run([encoder,"-y","-loglevel","error","-framerate",str(fps),
                            "-i",str(tmp/"frame_%04d.png"),"-c:v","libx264","-crf","24",
                            "-pix_fmt","yuv420p","-movflags","+faststart",str(target)],check=True)
            shutil.copyfile(target,out/target.name)
        variant_meta.update(encoded_frames=idx,duration_seconds=idx/fps,
                            frame_size=frame_size,poster_frame_size=frame_size,aspect_ratio="16:9",
                            theme=get_palette().name,geometry_sha256=capture_hash,
                            render_geometry=rendered_geometry())
        (out/f"{stem}-manifest.json").write_text(json.dumps(variant_meta,indent=2)+"\n")
        return variant_meta
    results=render_variants(args.out_dir,args.theme,variant)
    for theme,record in results.items():
        print(json.dumps({"stem":stem,"theme":theme,"envs":envs,"seeds":meta["selected_seeds"],
                          "frames":record["encoded_frames"],"seconds":record["duration_seconds"],
                          "out_dir":str(args.out_dir)}))


if __name__=="__main__":
    main()
