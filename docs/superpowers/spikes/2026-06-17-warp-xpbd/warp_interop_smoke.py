import torch, warp as wp
wp.init()

@wp.kernel
def add_one(x: wp.array(dtype=wp.vec2f)):
    i = wp.tid()
    x[i] = x[i] + wp.vec2f(1.0, 1.0)

# torch cuda tensor -> warp array (zero-copy view) -> kernel writes back
t = torch.zeros(8, 2, device="cuda", dtype=torch.float32)
a = wp.from_torch(t, dtype=wp.vec2f)
wp.launch(add_one, dim=t.shape[0], inputs=[a], device="cuda")
wp.synchronize()
print("warp 1.14:", wp.config.version)
print("all ones:", bool((t == 1.0).all().item()), "| sample:", t[0].tolist())

# also test reading a torch float matrix [N,N] in a kernel (the separation pattern)
@wp.kernel
def rowsum(m: wp.array2d(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    s = wp.float32(0.0)
    for j in range(m.shape[1]):
        s += m[i, j]
    out[i] = s

M = torch.arange(12, device="cuda", dtype=torch.float32).reshape(3, 4)
o = torch.zeros(3, device="cuda", dtype=torch.float32)
wp.launch(rowsum, dim=3, inputs=[wp.from_torch(M), wp.from_torch(o)], device="cuda")
wp.synchronize()
print("rowsum ok:", o.tolist(), "(expect [6,22,38])")
