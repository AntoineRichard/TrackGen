"""Per-kernel-group cost attribution at E=8192, N=256 (final-stage steady state).

Times each O(N^2)-ish kernel group in isolation to see how much of the per-iter cost the
near/far TP split can actually address (vs the untouched Sobolev conv + obstacle floor).
"""
from __future__ import annotations
import sys, time
import numpy as np
import warp as wp

from track_gen._src.types import TrackGenConfig
from track_gen._src import warp_generate_repulsive as rep
import nearfar as nf

E, N = 8192, 256
dev = "cuda"
cfg = TrackGenConfig(generator="repulsive", num_envs=E, device=dev)
s = rep.repulsive_alloc_scratch(cfg)
alpha, beta, eps = 3.0, 6.0, 1e-4
p_exp = -(beta - alpha) / 2.0
M, n_wall = s.M, s.n_wall
maxnbr = 64
nfb = nf._nf_buffers(s, maxnbr, dev)
h = s.h_by_n[N]

# init a plausible mid-growth state (circle + noise) so pow/branches are exercised
import numpy.random as npr
rng = npr.default_rng(1)
ang = np.linspace(0, 2*np.pi, N, endpoint=False)
base = np.stack([np.cos(ang), np.sin(ang)], -1)
c = (np.tile(base, (E, 1)) + 0.02*rng.standard_normal((E*N, 2))).astype(np.float32)
s.center.assign(c)
s.frozen.zero_(); s.reached.zero_(); s.peri_e.fill_(6.28)
s.L_target.fill_(6.3); s.L_init.fill_(6.28); s.len_coef.fill_(0.1)
nfb.overflow.zero_()

wp.launch(rep._tangent_weight_k, dim=(E, N), inputs=[s.center, N, s.frozen, s.tang, s.wdual], device=dev)
wp.launch(nf._cand_build_k, dim=(E, N),
          inputs=[s.center, N, maxnbr, 12.4, s.peri_e, s.frozen, nfb.nbr, nfb.ncount, nfb.overflow], device=dev)
wp.launch(rep._tp_prepass_k, dim=(E, N), inputs=[s.center, N, alpha, beta, eps, s.frozen, s.tang, s.wdual, s.wcoef, s.btan], device=dev)
wp.launch(nf._tp_prepass_near_k, dim=(E, N), inputs=[s.center, N, maxnbr, alpha, beta, eps, s.frozen, s.tang, s.wdual, nfb.nbr, nfb.wcoef_n, nfb.btan_n], device=dev)
wp.synchronize()


def timeit(fn, reps=100):
    fn(); wp.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    wp.synchronize()
    return (time.perf_counter() - t0) / reps * 1e3  # ms/call


groups = {
    "tangent_weight": lambda: wp.launch(rep._tangent_weight_k, dim=(E, N), inputs=[s.center, N, s.frozen, s.tang, s.wdual], device=dev),
    "cand_build": lambda: wp.launch(nf._cand_build_k, dim=(E, N), inputs=[s.center, N, maxnbr, 12.4, s.peri_e, s.frozen, nfb.nbr, nfb.ncount, nfb.overflow], device=dev),
    "tp_prepass_FULL": lambda: wp.launch(rep._tp_prepass_k, dim=(E, N), inputs=[s.center, N, alpha, beta, eps, s.frozen, s.tang, s.wdual, s.wcoef, s.btan], device=dev),
    "tp_gather_FULL(tp-only)": lambda: wp.launch(nf._tp_gather_full_k, dim=(E, N), inputs=[s.center, N, alpha, beta, eps, s.tang, s.wdual, s.wcoef, s.btan, s.frozen, nfb.g_full_tp], device=dev),
    "tp_prepass_NEAR": lambda: wp.launch(nf._tp_prepass_near_k, dim=(E, N), inputs=[s.center, N, maxnbr, alpha, beta, eps, s.frozen, s.tang, s.wdual, nfb.nbr, nfb.wcoef_n, nfb.btan_n], device=dev),
    "tp_gather_NEAR": lambda: wp.launch(nf._tp_gather_near_k, dim=(E, N), inputs=[s.center, N, maxnbr, alpha, beta, eps, s.tang, s.wdual, nfb.wcoef_n, nfb.btan_n, nfb.nbr, s.frozen, nfb.g_near_tp], device=dev),
    "obs_len_gather (O(N*M) M=%d)" % M: lambda: wp.launch(nf._obs_len_gather_k, dim=(E, N), inputs=[s.center, N, s.obs_pts, s.obs_mw, M, p_exp, s.reached, n_wall, 1, s.len_coef, s.frozen, nfb.g_ol], device=dev),
    "grad_gather_PROD (tp+obs+len)": lambda: wp.launch(rep._grad_gather_k, dim=(E, N), inputs=[s.center, N, alpha, beta, eps, s.tang, s.wdual, s.wcoef, s.btan, s.obs_pts, s.obs_mw, M, p_exp, s.reached, n_wall, 1, s.len_coef, s.frozen, s.grad], device=dev),
    "conv_k (Sobolev O(N^2)) x1": lambda: wp.launch(rep._conv_k, dim=(E, N), inputs=[s.grad, h, N, s.frozen, s.g], device=dev),
}

print(f"E={E} N={N} maxnbr={maxnbr}  (ms/call, min-of-100)")
for name, fn in groups.items():
    print(f"  {name:<34} {timeit(fn):8.4f}")

# per-iter model
prepass_full = timeit(groups["tp_prepass_FULL"])
gather_full = timeit(groups["tp_gather_FULL(tp-only)"])
prepass_near = timeit(groups["tp_prepass_NEAR"])
gather_near = timeit(groups["tp_gather_NEAR"])
obs = timeit(groups["obs_len_gather (O(N*M) M=%d)" % M])
conv = timeit(groups["conv_k (Sobolev O(N^2)) x1"])
cand = timeit(groups["cand_build"])
prod_gather = timeit(groups["grad_gather_PROD (tp+obs+len)"])

tp_full = prepass_full + gather_full
tp_near = prepass_near + gather_near
floor = 2*conv + obs  # the O(N^2)-ish work the split cannot touch
print(f"\nTP full (prepass+gather)   = {tp_full:.4f} ms")
print(f"TP near (prepass+gather)   = {tp_near:.4f} ms   ({tp_near/tp_full:.2f}x of full)")
print(f"obstacle+len (every iter)  = {obs:.4f} ms")
print(f"2x Sobolev conv (every it) = {2*conv:.4f} ms")
print(f"cand_build (refresh only)  = {cand:.4f} ms")
print(f"prod combined gather       = {prod_gather:.4f} ms")
for K in (4, 8, 16):
    base_iter = tp_full + obs + 2*conv
    # refresh: full + near + obs + 2conv + cand ; non-refresh: near + obs + 2conv
    split_iter = (tp_full + tp_near + obs + 2*conv + cand + (K-1)*(tp_near + obs + 2*conv)) / K
    print(f"K={K:2d}: modeled per-iter speedup (TP+obs+conv only) = {base_iter/split_iter:.2f}x")
