"""Repulsive self-graph: top-level CUDA graph node count + replay time (pre/post PR12).

    python bench_repulsive.py [E ...]
"""
import ctypes
import sys
import time

import numpy as np
import warp as wp

wp.config.quiet = True
from track_gen import PerEnvSeededRNG, TrackGenConfig, TrackGenerator  # noqa: E402


def node_count(graph):
    try:
        cuda = ctypes.CDLL("libcuda.so.1")
        n = ctypes.c_size_t(0)
        h = graph.graph if not isinstance(graph.graph, int) else ctypes.c_void_p(graph.graph)
        rc = cuda.cuGraphGetNodes(h, None, ctypes.byref(n))
        return int(n.value) if rc == 0 else f"rc={rc}"
    except Exception as exc:  # noqa: BLE001
        return f"n/a ({type(exc).__name__}: {exc})"


def main(Es):
    wp.init()
    dev = "cuda:0"
    for E in Es:
        cfg = TrackGenConfig(generator="repulsive", num_envs=E, device=dev)
        gen = TrackGenerator(cfg, PerEnvSeededRNG(seeds=1234, num_envs=E, device=dev))
        t0 = time.perf_counter()
        gen.generate()
        wp.synchronize()
        first = time.perf_counter() - t0
        g = gen.graph
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            gen.generate()
            wp.synchronize()
            times.append(time.perf_counter() - t0)
        valid = gen.generate().valid.numpy().mean()
        print(f"E={E:5d} first(capture+run)={first*1e3:8.1f} ms  replay median={np.median(times)*1e3:8.2f} ms "
              f"min={min(times)*1e3:8.2f} ms  top-level nodes={node_count(g)}  valid={valid:.3f}")


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or [8, 256, 1024])
