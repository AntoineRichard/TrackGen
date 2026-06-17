import time, torch, warp as wp
wp.init()
from track_gen import relaxation, geometry
from track_gen.types import TrackGenConfig
from benchmarks.benchmark_relaxation import _gen_simple_tracks

dev="cuda"; N,hw=256,0.03; E_gen,E=1024,8192; ITERS=150
margin=0.15; D=2*hw; target=D*(1.0+margin); R_min=hw*(1.0+margin)
base=_gen_simple_tracks(E_gen,N,1.0,"cpu",20)
center0=base.repeat((E+E_gen-1)//E_gen,1,1)[:E].contiguous().to(dev)
cfg=TrackGenConfig(device=dev,num_envs=E,num_points=N,half_width=hw,relax_margin=margin)
band_t=relaxation._band(center0,cfg).to(torch.int32).contiguous()
L0_t=(geometry.perimeter(center0)/N).contiguous().float()

@wp.kernel
def disp_k(center: wp.array(dtype=wp.vec2f), band: wp.array(dtype=wp.int32), L0: wp.array(dtype=wp.float32),
          N:int, target:wp.float32, R_min:wp.float32, sr:wp.float32, pr:wp.float32, br:wp.float32,
          out: wp.array(dtype=wp.vec2f)):
    t=wp.tid(); e=t//N; i=t%N; b=e*N; xi=center[t]
    sep=wp.vec2f(0.0,0.0); cnt=int(0)
    for j in range(N):
        dd=wp.abs(i-j); circ=wp.min(dd,N-dd)
        if circ>band[e]:
            diff=xi-center[b+j]; dist=wp.max(wp.length(diff),1.0e-9); pen=target-dist
            if pen>0.0: sep=sep+(0.5*pen/dist)*diff; cnt+=1
    if cnt>0: sep=sep/wp.float32(cnt)
    xn=center[b+((i+1)%N)]; xp=center[b+((i+N-1)%N)]
    dn=xn-xi; ln=wp.max(wp.length(dn),1.0e-9); dp=xi-xp; lp=wp.max(wp.length(dp),1.0e-9)
    spc=0.25*(((ln-L0[e])/ln)*dn - ((lp-L0[e])/lp)*dp)
    a=xi-xp; bb=xn-xi
    la=wp.length(a); lb=wp.length(bb); lc=wp.length(xn-xp)
    denom=wp.max(la*lb*lc,1.0e-12); cross=a[0]*bb[1]-a[1]*bb[0]; area=0.5*wp.abs(cross)
    kappa=4.0*area/denom; radius=1.0/wp.max(kappa,1.0e-12)
    mid=0.5*(xp+xn); toward=mid-xi; deficit=wp.max((R_min-radius)/R_min,0.0)
    bscale=wp.min(br*deficit,1.0)
    out[t]=sr*sep + pr*spc + bscale*toward

@wp.kernel
def apply_k(center: wp.array(dtype=wp.vec2f), disp: wp.array(dtype=wp.vec2f)):
    t=wp.tid(); center[t]=center[t]+disp[t]

center_buf=torch.empty(E*N,2,device=dev,dtype=torch.float32)
disp_buf=torch.empty(E*N,2,device=dev,dtype=torch.float32)
cw=wp.from_torch(center_buf,dtype=wp.vec2f); dw=wp.from_torch(disp_buf,dtype=wp.vec2f)
bw=wp.from_torch(band_t,dtype=wp.int32); lw=wp.from_torch(L0_t,dtype=wp.float32)

center_buf.copy_(center0.reshape(E*N,2)); torch.cuda.synchronize()
wp.capture_begin(device=dev)
for _ in range(ITERS):
    wp.launch(disp_k,dim=E*N,inputs=[cw,bw,lw,N,target,R_min,1.0,1.0,1.5,dw],device=dev)
    wp.launch(apply_k,dim=E*N,inputs=[cw,dw],device=dev)
graph=wp.capture_end(device=dev)

def solve():
    center_buf.copy_(center0.reshape(E*N,2)); torch.cuda.synchronize()
    wp.capture_launch(graph); wp.synchronize()
    return relaxation._resample_uniform(center_buf.view(E,N,2), N)

solve()  # warmup
t0=time.time(); out=solve(); torch.cuda.synchronize(); sec=time.time()-t0
sub=out[:2048].contiguous(); band2=relaxation._band(sub,cfg); th=geometry.thickness(sub,band2)
_,Nrm=geometry.tangents_normals(sub); bx=geometry.self_intersections(sub+hw*Nrm)+geometry.self_intersections(sub-hw*Nrm)
valid=((th>=0.98*hw)&(bx==0)).float().mean().item()
print(f"GRAPH-captured full-warp xpbd @ E={E} iters={ITERS}: {sec*1000:.1f} ms   valid(2048)={valid:.3f}")
print(f"(dense-warp integrated was ~550ms; torch baseline ~295000ms)")
