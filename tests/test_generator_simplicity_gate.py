import math
import torch
import pytest
from tests._oracle import geometry
from tests._oracle.generators import BezierCenterlineGenerator


def test_simplicity_gate_helper_flags_self_crossing():
    # +0.123 phase so the figure-eight crossing falls between samples (a genuine transversal
    # crossing) rather than on the coincident vertices t=0/pi (a degenerate vertex-touch the
    # collinear-robust detector correctly reports as 0 -- f32 and f64 agree).
    t = torch.linspace(0, 2 * math.pi, 256 + 1)[:-1] + 0.123
    fig8 = torch.stack([torch.sin(t), torch.sin(t) * torch.cos(t)], dim=-1).unsqueeze(0)
    circle = torch.stack([torch.cos(t), torch.sin(t)], dim=-1).unsqueeze(0)
    assert int(geometry.self_intersections(fig8)[0]) >= 1
    assert int(geometry.self_intersections(circle)[0]) == 0


def test_simple_gate_applied_in_generate():
    pytest.importorskip("warp")
    import warp as wp; wp.init()
    from track_gen._src.rng_utils import PerEnvSeededRNG
    from track_gen._src.types import TrackGenConfig
    E = 16
    seeds = torch.arange(E, dtype=torch.int32) + 7
    rng = PerEnvSeededRNG(seeds=wp.from_torch(seeds, dtype=wp.int32), num_envs=E, device="cpu")
    rng.set_seeds_warp(wp.from_torch(seeds, dtype=wp.int32),
                       ids=wp.array(list(range(E)), dtype=wp.int32, device="cpu"))
    cfg = TrackGenConfig(device="cpu", num_envs=E, scale=1.0, max_regen_iters=20)
    cl = BezierCenterlineGenerator(cfg, rng).generate(torch.arange(E))
    # Every VALID centerline must be a simple (non-self-intersecting) loop AT THE
    # RESOLUTION THE PIPELINE USES (256-point arc-length resample) — the same notion
    # the gate enforces and the relaxation consumes. (Sub-resolution corner cusps in
    # the raw dense curve are irrelevant: the pipeline resamples to 256 and the
    # relaxation rounds them out.)
    assert cl.valid.any()
    for e in torch.where(cl.valid)[0].tolist():
        pts = cl.points[e].unsqueeze(0)  # [1, M_max, 2] NaN-padded
        res, _ = geometry.arc_length_resample(pts, num=256)
        assert int(geometry.self_intersections(res)[0]) == 0
