"""Independent RC2 math audit; runtime variants exist only in this Python process.

Run this tool from the records checkout and point --repo at an immutable RC2
checkout/export. --help and smoke need only NumPy; --out-dir is explicit.
No runtime source is edited. This is diagnostic CPU code, not a production fix.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pathlib
import platform
import sys

import numpy as np

RUNTIME_REVISION = "a42759a3f999a8668a200cddc3f04e0508478dd1"
DOCS_REVISION = "4aec44a8f30d7ef998bb37ce6cb86e6ad2d12bd2"
EXPECTED_SOURCE = {
    "track_gen/_src/warp_generate_repulsive.py": "f87c497dbd86aedfdd4f8987f2feda546f78fb02ca165d274fadb866d0b41570",
    "tests/test_generate_repulsive.py": "baaf1e644be09466bece4172bd3f98de56c64b915fe1d710f1d0e69ef6edcb98",
}
ARGS = OUT = BASE = rep = wp = None


def source_hashes(repo):
    return {name: hashlib.sha256((repo / name).read_bytes()).hexdigest()
            for name in EXPECTED_SOURCE}


def load_runtime(repo):
    global rep, wp
    sys.path.insert(0, str(repo))
    import warp as warp_module
    from track_gen._src import warp_generate_repulsive as runtime_module
    if not pathlib.Path(runtime_module.__file__).resolve().is_relative_to(repo):
        raise RuntimeError("runtime import did not resolve inside --repo")
    wp, rep = warp_module, runtime_module


def _analytic_grad_warp(x, obs_pts, obs_mw, Lt, Li, dev):
    """Launch the original analytic kernels (same harness as the reference test).

    Inline harness avoids importing pytest/torch merely to run the NumPy/Warp audit.
    """
    n, M = x.shape[0], obs_pts.shape[0]
    center = wp.array(x.astype(np.float32), dtype=wp.vec2f, device=dev)
    frozen = wp.zeros(1, dtype=wp.int32, device=dev)
    reached = wp.zeros(1, dtype=wp.int32, device=dev)
    op = wp.array(obs_pts.astype(np.float32), dtype=wp.vec2f, device=dev)
    ow = wp.array(obs_mw.astype(np.float32), dtype=wp.float32, device=dev)
    lt = wp.array([Lt], dtype=wp.float32, device=dev)
    li = wp.array([Li], dtype=wp.float32, device=dev)
    tang, btan, grad = [wp.zeros(n, dtype=wp.vec2f, device=dev) for _ in range(3)]
    wdual, wcoef = [wp.zeros(n, dtype=wp.float32, device=dev) for _ in range(2)]
    coefficient, perimeter = [wp.zeros(1, dtype=wp.float32, device=dev) for _ in range(2)]
    wp.launch(rep._tangent_weight_k, dim=(1, n),
              inputs=[center, n, frozen, tang, wdual], device=dev)
    wp.launch(rep._tp_prepass_k, dim=(1, n),
              inputs=[center, n, 3., 6., 1e-4, frozen, tang, wdual, wcoef, btan], device=dev)
    wp.launch(rep._len_coef_k, dim=1,
              inputs=[center, n, lt, li, 30., frozen, coefficient, perimeter], device=dev)
    wp.launch(rep._grad_gather_k, dim=(1, n),
              inputs=[center, n, 3., 6., 1e-4, tang, wdual, wcoef, btan,
                      op, ow, M, -1.5, reached, 96, 0, coefficient, frozen, grad], device=dev)
    return grad.numpy().reshape(n, 2)


def run_case(name, seed=7, overrides=None, instrument=False):
    """Use the portable capture observer, but import runtime only from --repo.

    Overrides below mutate helper globals in memory. Neither checkout is edited.
    """
    helper_path = pathlib.Path(__file__).with_name("reproduce.py")
    spec = importlib.util.spec_from_file_location("repulsive_audit_capture", helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    helper.ROOT = ARGS.repo
    helper.CASES[name] = (seed, overrides or {})

    def audit_source_record():
        return {"reference_runtime_revision": RUNTIME_REVISION,
                "original_docs_revision": DOCS_REVISION,
                "file_sha256": source_hashes(ARGS.repo),
                "platform": platform.system() + "-" + platform.machine(),
                "python": platform.python_version(), "numpy": np.__version__,
                "audit_variant": name}
    helper.source_record = audit_source_record
    helper.run_case(name, BASE)
    record = json.loads((BASE / (name + ".json")).read_text())
    with np.load(BASE / record["array_archive"], allow_pickle=False) as archive:
        frames = [dict(frame, xy=archive[frame["xy_key"]].copy()) for frame in record["frames"]]
    return record, frames, None, record["substeps"]


def ordinary_sample():
    from track_gen import TrackGenConfig, TrackGenerator
    config = TrackGenConfig(generator="repulsive", device="cpu", num_envs=1, on_invalid="keep")
    generator = TrackGenerator(config, seeds=wp.array([7], dtype=wp.int32, device="cpu"))
    generator.generate()


def peri(x): return np.linalg.norm(np.roll(x,-1,axis=0)-x,axis=1).sum()
def turns(x):
    d=np.roll(x,-1,axis=0)-x;p=np.roll(d,1,axis=0)
    return np.degrees(np.abs(np.arctan2(p[:,0]*d[:,1]-p[:,1]*d[:,0],(p*d).sum(1))))
def energy(x,op,ow,Lt,Li):
    n=len(x);t=np.roll(x,-1,axis=0)-np.roll(x,1,axis=0);t/=np.maximum(np.linalg.norm(t,axis=1)[:,None],1e-8)
    el=np.linalg.norm(np.roll(x,-1,axis=0)-x,axis=1);w=.5*(el+np.roll(el,1))
    dd=np.abs(np.arange(n)[:,None]-np.arange(n)[None,:]);mask=np.minimum(dd,n-dd)>2
    d=x[None,:,:]-x[:,None,:];d2=(d*d).sum(2);wedge=d[:,:,0]*t[:,None,1]-d[:,:,1]*t[:,None,0]
    tp=(((abs(wedge)+1e-4)**3/(d2+1e-8)**3)*w[:,None]*w[None,:]*mask).sum()
    d=x[:,None,:]-op[None,:,:];ob=(ow[None,:]*((d*d).sum(2)+1e-8)**-1.5).sum()
    le=30*((peri(x)-Lt)/Li)**2
    return np.array([tp,ob,le])
def projected(x,grad,s=5/6):
    n=len(x);k=np.arange(n//2+1);lam=2-2*np.cos(2*np.pi*k/n);f=1/(lam**s+1e-3)
    conv=lambda q:np.fft.irfft(np.fft.rfft(q,axis=0)*f[:,None],n=n,axis=0)
    ed=np.roll(x,-1,axis=0)-x;u=ed/np.maximum(np.linalg.norm(ed,axis=1)[:,None],1e-8);lg=np.roll(u,1,axis=0)-u
    g=conv(grad);al=conv(lg);g-=((g*lg).sum()/(lg*al).sum())*al;g-=g.mean(0)
    return g,lg

def state_tests():
    z=np.load(BASE/'baseline.npz');meta=json.loads((BASE/'baseline.json').read_text());op=z['obs'];finite=np.isfinite(op).all(1);op=op[finite].astype(float)
    # Mass exactly as sampled; zero unused slots already removed.
    rdom=meta['config']['repulsive_domain_frac']*4.9029
    ow=np.r_[np.full(96,2*np.pi*rdom/96),np.repeat(2*np.pi*z['radii']/12*.25,12)]
    Li=2*np.pi*rdom/4;records=[];rng=np.random.default_rng(31022)
    for it in [1,47,128,176,191]:
        x=z[f'xy_{it}'].astype(float);Lt=meta['frames'][it]['target'];ow_active=ow.copy()
        if meta['frames'][it]['reached']:ow_active[96:]=0
        ga=_analytic_grad_warp(x,op,ow_active,Lt,Li,'cpu').astype(float)
        gtp=_analytic_grad_warp(x,op,ow_active*0,peri(x),Li,'cpu').astype(float)
        directional=[]
        for term,gref,weights in [('full',ga,ow_active),('tp',gtp,ow_active*0)]:
            for j in range(4):
                u=rng.normal(size=x.shape);u/=np.linalg.norm(u);h=1e-6
                fd=(energy(x+h*u,op,weights,Lt if term=='full' else peri(x),Li).sum()-energy(x-h*u,op,weights,Lt if term=='full' else peri(x),Li).sum())/(2*h)
                an=(gref*u).sum();directional.append(dict(term=term,analytic=float(an),fd=float(fd),rel=float(abs(an-fd)/max(abs(fd),1e-12))))
        candidates=[]
        for sexp in [5/6,5/3]:
            g,lg=projected(x,ga,sexp);d=-g/np.linalg.norm(g,axis=1).max()*peri(x)/len(x)
            for tau in [.4,.2,.1,.05,.025,.0125]:
                y=x+tau*d;y=y.mean(0)+(y-y.mean(0))*(Lt/peri(y))
                E=energy(y,op,ow_active,Lt,Li);E0=energy(x,op,ow_active,Lt,Li)
                candidates.append(dict(s=sexp,tau=tau,energy=E.tolist(),delta_E=float((E-E0).sum()),turnmax=float(turns(y).max()),dot_descent=float((ga*d).sum()),length_tangent=float((lg*d).sum())))
        records.append(dict(iteration=it,n=len(x),energy=energy(x,op,ow_active,Lt,Li).tolist(),turnmax=float(turns(x).max()),gradient_norm=float(np.linalg.norm(ga)),fd=directional,candidates=candidates))
        print('state',it,'FD max rel',max(r['rel'] for r in directional),'E',records[-1]['energy'],flush=True)
    (OUT/'state_tests.json').write_text(json.dumps(records,indent=2))

def mode_tests():
    rec=[]
    for n in [32,64,128,256]:
        a=np.arange(n)*2*np.pi/n;x=np.c_[np.cos(a),np.sin(a)];L=peri(x);zero=np.empty((0,2));ow=np.empty(0)
        E0=energy(x,zero,ow,L,L)[0]
        for k in sorted(set([2,4,8,n//4,n//2])):
            if k>n//2:continue
            v=x*np.cos(k*a)[:,None];amp=1e-5;ys=[]
            for sign in [-1,1]:
                y=x+sign*amp*v;y=y*(L/peri(y));ys.append(energy(y,zero,ow,L,L)[0])
            H=(ys[0]+ys[1]-2*E0)/(amp*amp*np.sum(v*v))
            lam=2-2*np.cos(2*np.pi*k/n)
            normalized=lam/(2-2*np.cos(2*np.pi/n))
            rec.append(dict(n=n,k=k,constrained_eigenvalue=float(H),base_preconditioned=float(H/(lam**(5/6)+1e-3)),order_corrected_preconditioned=float(H/(lam**(5/3)+1e-3)),normalized_base_preconditioned=float(H/(normalized**(5/6)+1e-3)),normalized_order_preconditioned=float(H/(normalized**(5/3)+1e-3))))
    (OUT/'mode_tests.json').write_text(json.dumps(rec,indent=2));print(json.dumps(rec,indent=2))

def filtered_run():
    orig=rep._sobolev_circulant_row
    mode=ARGS.variant or 'order'
    seed=ARGS.seed
    keep=ARGS.keep
    if mode=='order':rep._sobolev_circulant_row=lambda n,s,eps:orig(n,2*s,eps)
    elif mode in ('normalized','normalized_dc0'):
        def normalized(n,s,eps):
            k=np.arange(n//2+1);lam=(2-2*np.cos(2*np.pi*k/n))/(2-2*np.cos(2*np.pi/n))
            f=1/(lam**(2*s)+eps)
            if mode=='normalized_dc0':f[0]=0
            return np.fft.irfft(f,n=n).astype(np.float32)
        rep._sobolev_circulant_row=normalized
    run_case(f'audit_filter_{mode}_seed{seed}'+('_keep' if keep else ''),seed,{'repulsive_deactivate_obstacles':False} if keep else {},False)

def backtrack_run():
    original=rep._generate_repulsive
    mode=ARGS.variant or 'base'
    seed=ARGS.seed
    keep=ARGS.keep
    h_original=rep._sobolev_circulant_row
    if mode=='order':rep._sobolev_circulant_row=lambda n,s,eps:h_original(n,2*s,eps)
    if mode=='normalized_dc0':
        def normalized(n,s,eps):
            k=np.arange(n//2+1);lam=(2-2*np.cos(2*np.pi*k/n))/(2-2*np.cos(2*np.pi/n));f=1/(lam**(2*s)+eps);f[0]=0
            return np.fft.irfft(f,n=n).astype(np.float32)
        rep._sobolev_circulant_row=normalized
    events=[]
    def wrapped(seeds,config,out,valid,scratch,ctx,size):
        s=scratch;op=s.obs_pts.numpy();ow=s.obs_mw.numpy();Li=float(s.L_init.numpy()[0])
        class Proxy:
            def __getattr__(self,k):return getattr(ctx,k)
            def bounded_loop(self,n_iters,body,**kw):
                def with_ctx(c,it):
                    self.it=it;body(self,it)
                return ctx.bounded_loop(n_iters,with_ctx,**kw)
            def launch(self,kernel,dim,inputs=None,**kw):
                if kernel.key=='_step_k':
                    self.n=inputs[6];self.x=s.center.numpy()[:self.n].astype(float).copy()
                    self.d=-(s.g.numpy()[:self.n].astype(float)-s.gmean.numpy()[0])*float(s.msl.numpy()[0])/float(s.gmax.numpy()[0])
                ans=ctx.launch(kernel,dim,inputs,**kw)
                if kernel.key=='_rescale_k' and not s.frozen.numpy()[0]:
                    n=self.n;x=self.x;d=self.d;Lt=float(s.L_target.numpy()[0]);Li=float(s.L_init.numpy()[0]);op=s.obs_pts.numpy();ow=s.obs_mw.numpy().copy();active=ow!=0
                    if config.repulsive_deactivate_obstacles and s.reached.numpy()[0]:active[96:]=False
                    op=op[active].astype(float);ow=ow[active].astype(float)
                    def retr(a):
                        y=x+a*d;return y.mean(0)+(y-y.mean(0))*(Lt/peri(y))
                    E0=float(energy(retr(0),op,ow,Lt,Li).sum());h=1e-5
                    derivative=float((energy(retr(h),op,ow,Lt,Li).sum()-energy(retr(-h),op,ow,Lt,Li).sum())/(2*h))
                    chosen=0.;backtracks=8
                    for j in range(8):
                        a=config.repulsive_tau/(2**j);y=retr(a);E=float(energy(y,op,ow,Lt,Li).sum())
                        if derivative<0 and E<=E0+1e-4*a*derivative:
                            chosen=a;backtracks=j;break
                    y=retr(chosen).astype(np.float32);flat=s.center.numpy();flat[:n]=y
                    wp.copy(s.center,wp.array(flat,dtype=wp.vec2f,device='cpu'))
                    events.append(dict(it=self.it+1,step=chosen,backtracks=backtracks,derivative=derivative,energy_before=E0,energy_after=float(energy(y,op,ow,Lt,Li).sum())))
                return ans
        return original(seeds,config,out,valid,s,Proxy(),size)
    rep._generate_repulsive=wrapped
    name=f'audit_backtrack_{mode}_seed{seed}'+('_keep' if keep else '')
    rec,frames,_,_=run_case(name,seed,{'repulsive_deactivate_obstacles':False} if keep else {},False)
    (OUT/(name+'-events.json')).write_text(json.dumps(events,indent=2))
    print('backtrack counts',np.unique([e['backtracks'] for e in events],return_counts=True),flush=True)

def cancellation():
    z=np.load(BASE/'baseline.npz');meta=json.loads((BASE/'baseline.json').read_text());op=z['obs'];active=np.isfinite(op).all(1);op=op[active].astype(float)
    rdom=meta['config']['repulsive_domain_frac']*4.9029;ow=np.r_[np.full(96,2*np.pi*rdom/96),np.repeat(2*np.pi*z['radii']/12*.25,12)];Li=2*np.pi*rdom/4
    records=[]
    for it in [1,47,128,176,191]:
        x=z[f'xy_{it}'].astype(float);L=peri(x);Lt=L*1.012;wa=ow.copy()
        if meta['frames'][it]['reached']:wa[96:]=0
        ga=_analytic_grad_warp(x,op,wa,Lt,Li,'cpu').astype(float);g0=_analytic_grad_warp(x,op,wa,L,Li,'cpu').astype(float)
        pa,lg=projected(x,ga);p0,_=projected(x,g0);expected=60*(L-Lt)/Li**2*lg
        rec=dict(iteration=it,gradient_length_residual=float(np.linalg.norm(ga-g0-expected)/np.linalg.norm(expected)),projected_difference_relative=float(np.linalg.norm(pa-p0)/np.linalg.norm(p0)),projected_length_gradient_norm=float(np.linalg.norm(projected(x,expected)[0])))
        records.append(rec)
    (OUT/'length_cancellation.json').write_text(json.dumps(records,indent=2));print(json.dumps(records,indent=2))

def weighted_obstacles():
    rng=np.random.default_rng(991128);records=[]
    saved=np.load(BASE/'baseline.npz')
    for n,it in [(32,None),(64,None),(256,191)]:
        if it is None:
            a=np.arange(n)*2*np.pi/n;x=np.c_[np.cos(a),np.sin(a)]*(1+.1*rng.normal(size=n))[:,None]
        else:x=saved[f'xy_{it}'].astype(float)
        op=rng.normal(size=(15,2))*2;ow=rng.uniform(.1,1,size=15);d=x[:,None,:]-op[None,:,:];d2=(d*d).sum(2)+1e-8
        O=(ow[None,:]*d2**-1.5).sum(1);dO=(-3*ow[None,:,None]*d*d2[:,:,None]**-2.5).sum(1)
        ed=np.roll(x,-1,axis=0)-x;l=np.linalg.norm(ed,axis=1);u=ed/l[:,None];w=.5*(l+np.roll(l,1))
        g=w[:,None]*dO+.5*np.roll(u,1,axis=0)*(np.roll(O,1)+O)[:,None]-.5*u*(O+np.roll(O,-1))[:,None]
        def E(y,points=op,mass=ow):
            ell=np.linalg.norm(np.roll(y,-1,axis=0)-y,axis=1);wd=.5*(ell+np.roll(ell,1));dd=y[:,None,:]-points[None,:,:]
            return float((wd[:,None]*mass[None,:]*((dd*dd).sum(2)+1e-8)**-1.5).sum())
        fd=[]
        for j in range(8):
            v=rng.normal(size=x.shape);v/=np.linalg.norm(v);h=1e-6
            exact=(g*v).sum();approx=(E(x+h*v)-E(x-h*v))/(2*h)
            fd.append(float(abs(exact-approx)/max(abs(approx),1e-12)))
        # Epsilon stays fixed; large distances make its scale-dependent departure small.
        scale=3.;ratio=E(x*scale,op*scale,ow*scale)/E(x)
        records.append(dict(n=n,it=it,max_directional_relative_error=max(fd),scale3_energy_ratio=ratio,expected_ratio=1/3))
    (OUT/'weighted_obstacles.json').write_text(json.dumps(records,indent=2));print(json.dumps(records,indent=2))

def paper_discretization():
    n=256;a=np.arange(n)*2*np.pi/n;x=np.c_[np.cos(a),np.sin(a)];L=peri(x);rec=[]
    def Epaper(y):
        ed=np.roll(y,-1,axis=0)-y;l=np.linalg.norm(ed,axis=1);T=ed/l[:,None];dd=np.abs(np.arange(n)[:,None]-np.arange(n)[None,:]);mask=np.minimum(dd,n-dd)>1;total=0.
        for di in [0,1]:
            for dj in [0,1]:
                p=np.roll(y,-di,axis=0);q=np.roll(y,-dj,axis=0);d=q[None,:,:]-p[:,None,:];d2=(d*d).sum(2);wedge=d[:,:,0]*T[:,None,1]-d[:,:,1]*T[:,None,0]
                # Use same epsilon only to isolate quadrature/tangent choices.
                total+=np.sum((abs(wedge)+1e-4)**3/(d2+1e-8)**3*l[:,None]*l[None,:]*mask)/4
        return total
    zero=np.empty((0,2));ow=np.empty(0)
    for name,E in [('vertex',lambda y:energy(y,zero,ow,L,L)[0]),('paper_edge',Epaper)]:
        E0=E(x)
        for k in [2,4,8,32,64,128]:
            v=x*np.cos(k*a)[:,None];amp=1e-5;ys=[]
            for sign in [-1,1]:
                y=x+sign*amp*v;y*=L/peri(y);ys.append(E(y))
            H=(ys[0]+ys[1]-2*E0)/(amp*amp*np.sum(v*v));rec.append(dict(discretization=name,k=k,constrained_eigenvalue=float(H)))
    (OUT/'paper_discretization.json').write_text(json.dumps(rec,indent=2));print(json.dumps(rec,indent=2))

def spectra():
    records=[]
    for n in [64,128,256]:
        for p in [5/6,5/3]:
            for normalized,dc in [(False,1000),(True,1000),(True,0)]:
                k=np.arange(n//2+1);lam=2-2*np.cos(2*np.pi*k/n)
                if normalized:lam/=lam[1]
                f=1/(lam**p+1e-3);f[0]=dc;h=np.fft.irfft(f,n=n).astype(np.float32);actual=np.fft.rfft(h).real
                records.append(dict(n=n,exponent=p,normalized=normalized,dc=dc,min_realized_eigenvalue=float(actual[1:].min()),min_intended_eigenvalue=float(f[1:].min()),max_relative_error=float(max(abs(actual[1:]-f[1:])/f[1:])),negative_modes=int((actual[1:]<0).sum())))
    (OUT/'filter_spectra.json').write_text(json.dumps(records,indent=2));print(json.dumps(records,indent=2))

def stage_jumps():
    original=rep._generate_repulsive;events=[]
    def wrapped(seeds,config,out,valid,s,ctx,size):
        class Proxy:
            n=config.repulsive_stages[0]
            def __getattr__(self,k):return getattr(ctx,k)
            def bounded_loop(self,n_iters,body,**kw):
                def with_ctx(c,it):self.it=it;body(self,it)
                return ctx.bounded_loop(n_iters,with_ctx,**kw)
            def measure(self,x):
                op=s.obs_pts.numpy();ow=s.obs_mw.numpy().copy();active=ow!=0
                if config.repulsive_deactivate_obstacles and s.reached.numpy()[0]:active[96:]=False
                return energy(x.astype(float),op[active].astype(float),ow[active].astype(float),float(s.L_target.numpy()[0]),float(s.L_init.numpy()[0])).tolist()
            def copy(self,dst,src,**kw):
                if dst.ptr==s.center.ptr and kw.get('count',0)>self.n:
                    before=self.measure(s.center.numpy()[:self.n]);nold=self.n
                    ans=ctx.copy(dst,src,**kw);self.n=kw['count']
                    events.append(dict(kind='stage',iteration=self.it+1,n_old=nold,n_new=self.n,energy_before=before,energy_after=self.measure(s.center.numpy()[:self.n])))
                    return ans
                return ctx.copy(dst,src,**kw)
            def launch(self,kernel,dim,inputs=None,**kw):
                if kernel.key=='_copy_unfrozen_k':before=self.measure(s.center.numpy()[:self.n])
                ans=ctx.launch(kernel,dim,inputs,**kw)
                if kernel.key=='_copy_unfrozen_k':events.append(dict(kind='resample',iteration=self.it+1,n=self.n,energy_before=before,energy_after=self.measure(s.center.numpy()[:self.n])))
                return ans
        return original(seeds,config,out,valid,s,Proxy(),size)
    rep._generate_repulsive=wrapped;ordinary_sample()
    (OUT/'stage_jumps.json').write_text(json.dumps(events,indent=2));print(json.dumps(events,indent=2))


def smoke():
    """Small deterministic NumPy checks; no Warp import, trajectories or variants."""
    n = 16
    angle = np.arange(n) * 2 * np.pi / n
    x = np.c_[np.cos(angle), np.sin(angle)]
    gradient = np.random.default_rng(31).normal(size=x.shape)
    direction, length_gradient = projected(x, gradient)
    assert abs((direction * length_gradient).sum()) < 1e-9
    assert np.linalg.norm(direction.mean(0)) < 1e-10
    assert np.linalg.norm(projected(x, length_gradient)[0]) < 1e-9
    lam = 2 - 2 * np.cos(2 * np.pi * np.arange(n // 2 + 1) / n)
    intended = 1 / (lam ** (5 / 6) + 1e-3)
    row = np.fft.irfft(intended, n=n).astype(np.float32)
    assert np.fft.rfft(row).real[1:].min() > 0
    zero = np.empty((0, 2))
    assert np.isfinite(energy(x, zero, np.empty(0), peri(x), peri(x))).all()
    result = {"synthetic_smoke": True, "n": n,
              "length_projection_residual": float(abs((direction * length_gradient).sum())),
              "length_only_projected_norm": float(np.linalg.norm(projected(x, length_gradient)[0])),
              "nonzero_filter_modes_positive": True}
    (OUT / "smoke.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


def main():
    global ARGS, OUT, BASE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("smoke", "capture", "states", "modes", "filter",
                                           "backtrack", "cancellation", "weighted_obstacles",
                                           "paper_discretization", "spectra", "stage_jumps"))
    parser.add_argument("--repo", type=pathlib.Path, required=True,
                        help="immutable RC2 checkout/export to import; never inferred from cwd")
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    parser.add_argument("--states-dir", type=pathlib.Path,
                        help="baseline.json/npz input; defaults to OUT/captures")
    parser.add_argument("--variant", choices=("base", "order", "normalized", "normalized_dc0"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--keep", action="store_true", help="retain inner-disc forces")
    parser.add_argument("--allow-source-difference", action="store_true",
                        help="deliberately audit another source revision; loses RC2 baseline identity")
    ARGS = parser.parse_args()
    if ARGS.command == "backtrack" and ARGS.variant == "normalized":
        parser.error("backtrack supports base, order, or normalized_dc0 variants")
    ARGS.repo = ARGS.repo.resolve()
    OUT = ARGS.out_dir.resolve()
    BASE = ARGS.states_dir.resolve() if ARGS.states_dir else OUT / "captures"
    before = source_hashes(ARGS.repo)
    if before != EXPECTED_SOURCE and not ARGS.allow_source_difference:
        parser.error("--repo differs from audited RC2 source hashes; use an immutable RC2 export")
    OUT.mkdir(parents=True, exist_ok=True)
    BASE.mkdir(parents=True, exist_ok=True)
    runtime_commands = {"capture", "states", "filter", "backtrack", "cancellation", "stage_jumps"}
    if ARGS.command in runtime_commands:
        load_runtime(ARGS.repo)
    environment = {"reference_runtime_revision": RUNTIME_REVISION, "reference_docs_revision": DOCS_REVISION,
                   "source_sha256": before, "source_matches_rc2": before == EXPECTED_SOURCE,
                   "python": platform.python_version(), "platform": platform.system() + "-" + platform.machine(),
                   "numpy": np.__version__, "warp": wp.__version__ if wp is not None else None}
    (OUT / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    functions = {"smoke": smoke, "capture": lambda: run_case("baseline"),
                 "states": state_tests, "modes": mode_tests, "filter": filtered_run,
                 "backtrack": backtrack_run, "cancellation": cancellation,
                 "weighted_obstacles": weighted_obstacles,
                 "paper_discretization": paper_discretization, "spectra": spectra,
                 "stage_jumps": stage_jumps}
    try:
        functions[ARGS.command]()
    finally:
        if source_hashes(ARGS.repo) != before:
            raise AssertionError("audited source hashes changed while this process ran")


if __name__ == "__main__":
    main()
