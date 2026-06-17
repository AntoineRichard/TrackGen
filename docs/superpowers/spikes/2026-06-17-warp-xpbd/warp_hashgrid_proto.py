import time, torch, warp as wp
wp.init()
from track_gen import relaxation, geometry
from track_gen.types import TrackGenConfig
from track_gen.warp_relax import separation_disp as dense_sep
from benchmarks.benchmark_relaxation import _gen_simple_tracks

dev="cuda"; N,hw=256,0.03; CHUNK=2048; margin=0.15; D=2*hw; target=D*(1.0+margin)
base=_gen_simple_tracks(1024,N,1.0,"cpu",20)
center=base.repeat((CHUNK+1023)//1024,1,1)[:CHUNK].contiguous().to(dev)
E=center.shape[0]
cfg=TrackGenConfig(device=dev,num_envs=E,num_points=N,half_width=hw,relax_margin=margin)
band=relaxation._band(center,cfg)
ZS=1.0   # z-slab spacing per env (>> target so envs never share a query neighborhood)

@wp.kernel
def sep_hg(grid: wp.uint64, pts: wp.array(dtype=wp.vec3f), band: wp.array(dtype=wp.int32),
          N: int, target: wp.float32, out: wp.array(dtype=wp.vec2f)):
    tid=wp.tid()
    i=wp.hash_grid_point_id(grid, tid)        # original point index
    e=i//N; li=i%N; xi=pts[i]
    disp=wp.vec2f(0.0,0.0); cnt=int(0)
    q=wp.hash_grid_query(grid, xi, target)
    nbr=int(0)
    while wp.hash_grid_query_next(q, nbr):
        ej=nbr//N; lj=nbr%N
        if ej==e:
            d=wp.abs(li-lj); circ=wp.min(d, N-d)
            if circ>band[e]:
                xj=pts[nbr]
                diff=wp.vec2f(xi[0]-xj[0], xi[1]-xj[1])
                dist=wp.max(wp.length(diff), 1.0e-9); pen=target-dist
                if pen>0.0:
                    disp=disp+(0.5*pen/dist)*diff; cnt+=1
    if cnt>0: out[i]=disp/wp.float32(cnt)
    else: out[i]=wp.vec2f(0.0,0.0)

grid=wp.HashGrid(128,128,128,device=dev)
def hg_sep(c):
    flat=c.reshape(E*N,2)
    z=(torch.arange(E,device=dev,dtype=torch.float32).repeat_interleave(N)*ZS).unsqueeze(1)
    pts3=torch.cat([flat,z],dim=1).contiguous()
    pw=wp.from_torch(pts3,dtype=wp.vec3f)
    grid.build(points=pw, radius=target)
    bw=wp.from_torch(band.to(torch.int32),dtype=wp.int32)
    out_t=torch.empty(E*N,2,device=dev,dtype=torch.float32); ow=wp.from_torch(out_t,dtype=wp.vec2f)
    wp.launch(sep_hg, dim=E*N, inputs=[grid.id,pw,bw,N,target,ow], device=dev)
    torch.cuda.synchronize()
    return out_t.view(E,N,2)

ref=dense_sep(center,band,target)
got=hg_sep(center)
print(f"E={E} N={N}")
print(f"hashgrid vs dense max abs err: {(ref-got).abs().max().item():.2e}  allclose={torch.allclose(ref,got,atol=1e-5)}")
def timeit(fn,n=30):
    for _ in range(3): fn()
    torch.cuda.synchronize(); t0=time.time()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.time()-t0)/n*1000
print(f"dense    sep: {timeit(lambda: dense_sep(center,band,target)):7.3f} ms")
print(f"hashgrid sep: {timeit(lambda: hg_sep(center)):7.3f} ms")
