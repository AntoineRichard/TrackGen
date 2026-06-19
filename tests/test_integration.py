import importlib

import torch


def test_package_imports_without_circular_import():
    # A clean import of the package must not raise (no circular import between
    # track_generator <-> inflation <-> types).
    import track_gen

    importlib.reload(track_gen)
    assert hasattr(track_gen, "TrackGenerator")
    assert hasattr(track_gen, "Track")
    assert hasattr(track_gen, "TrackGenConfig")


def test_inflate_runs_on_a_synthetic_centerline_without_warp():
    # End-to-end inflation on a hand-built circle, importing only warp-free leaves.
    # constant_spacing mode: output is [E, N_max, 2] NaN-padded with a per-env real
    # point count in track.count; assertions are count-aware (mask to [:count]).
    import math

    from tests._oracle.generators import Centerline
    from track_gen._src.types import TrackGenConfig
    from tests._oracle import inflation

    theta = torch.linspace(0, 2 * math.pi, 201)[:-1]
    pts = torch.stack([2.0 * torch.cos(theta), 2.0 * torch.sin(theta)], dim=-1).unsqueeze(0)
    cl = Centerline(points=pts, valid=torch.ones(1, dtype=torch.bool))
    # constant_spacing resample is deterministic: a radius-2 circle (200-gon, total
    # length ~12.56) at spacing=0.1 yields exactly 126 real points. We size N_max to
    # that count so the buffer is fully real -- the torch validity oracle in inflate()
    # does NOT count-mask its turning/thickness checks, so any NaN padding would poison
    # them and (per that oracle's documented limitation) flag the track invalid. With
    # no padding, "a clean circle inflates to a valid track" still holds end-to-end.
    cfg = TrackGenConfig(num_envs=1, half_width=0.1, spacing=0.1, N_max=126)

    track = inflation.inflate(cl, cfg)

    # Output is the constant_spacing layout [E, N_max, 2], not a fixed point count.
    assert track.center.shape == (1, cfg.N_max, 2)
    assert track.outer.shape == (1, cfg.N_max, 2)
    assert track.inner.shape == (1, cfg.N_max, 2)

    # No padding for this circle: the whole buffer is real and finite.
    count = int(track.count[0])
    assert count == cfg.N_max
    finite = torch.isfinite(track.center[0]).all(dim=-1)  # [N_max] bool
    assert bool(finite[:count].all())

    assert bool(track.valid[0])
