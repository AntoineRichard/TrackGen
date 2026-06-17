import time, torch, warp as wp
wp.init()
margin=0.15; hw=0.03; D=2*hw; target=D*(1.0+margin); CHUNK=1024; ZS=1.0

@wp.kernel
def dense(center: wp.array(dtype=wp.vec2f), band: wp.array(dtype=wp.int32), N:int, target:wp.float32, out: wp.array(dtype=wp.vec2f)):
    t=wp.tid(); e=t//N; i=t%N; xi=center[t]; disp=wp.vec2f(0.0,0.0); cnt=int(0); b=e*N
    for j in range(N):
        d=wp.abs(i-j); c=wp.min(d,N-d)
        if c>band[e]:
            diff=xi-center[b+j]; dist=wp.max(wp.length(diff),1.0e-9); pen=target-dist
            if pen>0.0: disp=disp+(0.5*pen/dist)*diff; cnt+=1
    out[t]= disp/wp.float32(cnt) if cnt>0 else wp.vec2f(0.0,0.0)

@wp.kernel
def hg(grid: wp.uint64, pts: wp.array(dtype=wp.vec3f), band: wp.array(dtype=wp.int32), N:int, target:wp.float32, out: wp.array(dtype=wp.vec2f)):
    tid=wp.tid(); i=wp.hash_grid_point_id(grid,tid); e=i//N; li=i%N; xi=pts[i]
    disp=wp.vec2f(0.0,0.0); cnt=int(0); q=wp.hash_grid_query(grid,xi,target); nbr=int(0)
    while wp.hash_grid_query_next(q,nbr):
        ej=nbr//N; lj=nbr%N
        if ej==e:
            d=wp.abs(li-lj); c=wp.min(d,N-d)
            if c>band[e]:
                xj=pts[nbr]; diff=wp.vec2f(xi[0]-xj[0],xi[1]-xj[1]); dist=wp.max(wp.length(diff),1.0e-9); pen=target-dist
                if pen>0.0: disp=disp+(0.5*pen/dist)*diff; cnt+=1
    out[i]= disp/wp.float32(cnt) if cnt>0 else wp.vec2f(0.0,0.0)

dev="cuda"; grid=wp.HashGrid(128,128,128,device=dev)
def bench(N):
    E=CHUNK
    torch.manual_seed(0)
    c=(torch.randn(E,N,2)*0.5).to(dev)
    band=torch.full((E,),max(1,round(target/(2.0/N))),dtype=torch.int32,device=dev)
    cf=wp.from_torch(c.reshape(E*N,2).contiguous(),dtype=wp.vec2f)
    bw=wp.from_torch(band,dtype=wp.int32); out=torch.empty(E*N,2,device=dev); ow=wp.from_torch(out,dtype=wp.vec2f)
    z=(torch.arange(E,device=dev,dtype=torch.float32).repeat_interleave(N)*ZS).unsqueeze(1)
    def d_(): wp.launch(dense,dim=E*N,inputs=[cf,bw,N,target,ow],device=dev)
    def h_():
        pts3=torch.cat([c.reshape(E*N,2),z],1).contiguous(); pw=wp.from_torch(pts3,dtype=wp.vec3f)
        grid.build(points=pw,radius=target); wp.launch(hg,dim=E*N,inputs=[grid.id,pw,bw,N,target,ow],device=dev)
    def tm(fn,n=20):
        for _ in range(3): fn()
        torch.cuda.synchronize(); t0=time.time()
        for _ in range(n): fn()
        torch.cuda.synchronize(); return (time.time()-t0)/n*1000
    print(f"N={N:5d} chunk={E}: dense={tm(d_):7.3f}ms  hashgrid={tm(h_):7.3f}ms")
for N in (256,512,1024,2048): bench(N)
