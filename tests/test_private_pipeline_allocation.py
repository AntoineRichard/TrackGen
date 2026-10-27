"""Private numeric composition keeps all elevation storage stable after construction."""
import numpy as np
import pytest
import warp as wp

from track_gen import TrackGenConfig, TrackGenerator
from tests._reference_generators import warp_pipeline
from track_gen.exec import LaunchContext


@pytest.mark.parametrize("profile", ["noise", "uniform", "random_walk"])
@pytest.mark.parametrize("source", ["allocator", "program_view"])
def test_private_composition_reuses_elevation_arrays(profile, source, monkeypatch):
    cfg = TrackGenConfig(device="cpu", num_envs=2, num_points=32, N_max=128,
                         relax_enable=False, z_profile=profile, z_min=-0.2, z_max=0.3)
    seed = wp.array(np.array([3, 9], dtype=np.int32), dtype=wp.int32, device="cpu")
    ctx = LaunchContext.for_call("cpu")
    if source == "allocator":
        out, scratch = warp_pipeline._inflate_warp_alloc(cfg, seed=seed)
    else:
        gen = TrackGenerator(cfg)
        out = gen.generate()
        scratch = gen._scratch
        effective = gen.params["elevation"]
        assert scratch.inflate.z_dyn is not None
        for name in effective:
            assert scratch.inflate.z_dyn[name] is effective.in_effect(name)
    warp_pipeline._run_pipeline(cfg, seed, out=out, scratch=scratch, ctx=ctx)
    pointers = {name: array.ptr for name, array in scratch.inflate.z_dyn.items()}
    center = out.center.numpy().tobytes()

    def forbidden(*args, **kwargs):
        raise AssertionError("a warmed private composition allocated device storage")

    with monkeypatch.context() as guard:
        for name in ("empty", "zeros", "full", "empty_like", "zeros_like", "clone"):
            guard.setattr(wp, name, forbidden)
        warp_pipeline._run_pipeline(cfg, seed, out=out, scratch=scratch, ctx=ctx)
    assert {name: array.ptr for name, array in scratch.inflate.z_dyn.items()} == pointers
    assert out.center.numpy().tobytes() == center
