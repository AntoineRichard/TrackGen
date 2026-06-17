import time, torch, warp as wp
wp.init()
from track_gen import relaxation, geometry
from track_gen.types import TrackGenConfig
from benchmarks.benchmark_relaxation import _gen_simple_tracks

dev = "cuda"; N, hw = 256, 0.03; CHUNK = 2048
base = _gen_simple_tracks(1024, N, 1.0, "cpu", 20)
center = base.repeat((CHUNK + 1023)//1024, 1, 1)[:CHUNK].contiguous().to(dev)  # [E,N,2]
E = center.shape[0]
cfg = TrackGenConfig(device=dev, num_envs=E, num_points=N, half_width=hw, relax_margin=0.15)
band_t = relaxation._band(center, cfg).to(torch.int32)            # [E]
D = 2*hw; margin = 0.15; target = D*(1.0+margin)
circ = geometry.circ_index_dist(N, dev); mask_keep = circ[None] > band_t.long().view(E,1,1)

# ---- torch reference ----
def torch_sep():
    return relaxation._separation_disp(center, mask_keep, D, margin)

# ---- fused warp kernel ----
@wp.kernel
def sep_kernel(center: wp.array(dtype=wp.vec2f), band: wp.array(dtype=wp.int32),
              N: int, target: wp.float32, out: wp.array(dtype=wp.vec2f)):
    t = wp.tid()
    e = t // N
    i = t % N
    xi = center[t]
    disp = wp.vec2f(0.0, 0.0)
    cnt = int(0)
    base = e * N
    for j in range(N):
        d = wp.abs(i - j)
        c = wp.min(d, N - d)
        if c > band[e]:
            diff = xi - center[base + j]
            dist = wp.max(wp.length(diff), 1.0e-9)
            pen = target - dist
            if pen > 0.0:
                disp = disp + (0.5 * pen / dist) * diff
                cnt += 1
    if cnt > 0:
        out[t] = disp / wp.float32(cnt)
    else:
        out[t] = wp.vec2f(0.0, 0.0)

center_flat = wp.from_torch(center.reshape(E*N, 2), dtype=wp.vec2f)
band_w = wp.from_torch(band_t, dtype=wp.int32)
out_t = torch.empty(E*N, 2, device=dev, dtype=torch.float32)
out_w = wp.from_torch(out_t, dtype=wp.vec2f)
def warp_sep():
    wp.launch(sep_kernel, dim=E*N, inputs=[center_flat, band_w, N, target, out_w], device=dev)
    return out_t.view(E, N, 2)

def timeit(fn, n=30):
    for _ in range(3): fn()
    torch.cuda.synchronize(); wp.synchronize(); t0=time.time()
    for _ in range(n): fn()
    torch.cuda.synchronize(); wp.synchronize()
    return (time.time()-t0)/n*1000

ref = torch_sep(); got = warp_sep()
maxerr = (ref - got).abs().max().item()
print(f"E={E} N={N}")
print(f"equivalence max abs err: {maxerr:.2e}  (allclose={torch.allclose(ref, got, atol=1e-5)})")
print(f"torch separation: {timeit(torch_sep):8.3f} ms/call")
print(f"warp  separation: {timeit(warp_sep):8.3f} ms/call")
