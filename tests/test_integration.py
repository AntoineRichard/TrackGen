import importlib

import torch


def test_package_imports_without_circular_import():
    # A clean import of the package must not raise (no circular import between
    # track_generator <-> inflation <-> types).
    import track_gen

    importlib.reload(track_gen)
    assert hasattr(track_gen, "TrackGenerator")
    assert hasattr(track_gen, "Centerline")
    assert hasattr(track_gen, "Track")
    assert hasattr(track_gen, "TrackGenConfig")


def test_inflate_runs_on_a_synthetic_centerline_without_warp():
    # End-to-end inflation on a hand-built circle, importing only warp-free leaves.
    import math

    from track_gen.generators import Centerline
    from track_gen.types import TrackGenConfig
    from track_gen import inflation

    theta = torch.linspace(0, 2 * math.pi, 201)[:-1]
    pts = torch.stack([2.0 * torch.cos(theta), 2.0 * torch.sin(theta)], dim=-1).unsqueeze(0)
    cl = Centerline(points=pts, valid=torch.ones(1, dtype=torch.bool))
    cfg = TrackGenConfig(num_envs=1, num_points=128, output_mode="fixed", clamp_self_distance=False)

    track = inflation.inflate(cl, cfg)
    assert track.center.shape == (1, 128, 2)
    assert bool(track.valid[0])
