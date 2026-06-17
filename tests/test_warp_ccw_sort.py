import math, pytest, torch
pytest.importorskip("warp")
from track_gen import warp_pipeline as wpl
from track_gen import geometry

DEVS = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _scrambled_env(P, cx, cy, r, scramble, dev):
    """P points on a jittered circle (well-separated angles), then scrambled.

    Angles are evenly spaced (2*pi/P apart) plus a small fixed jitter, so the
    centroid-relative atan2 keys are well separated -> the ccw_sort permutation
    is deterministic across torch-vs-Warp atan2 ULP differences.
    """
    # Fixed per-point radius/angle jitter (deterministic, small).
    base = torch.arange(P, dtype=torch.float32)
    ang = base * (2.0 * math.pi / P) + 0.07 * torch.sin(base * 1.3)
    rad = r * (1.0 + 0.13 * torch.cos(base * 0.9))
    x = cx + rad * torch.cos(ang)
    y = cy + rad * torch.sin(ang)
    pts = torch.stack([x, y], dim=-1)            # [P, 2] in CCW-ish order
    perm = torch.tensor(scramble, dtype=torch.long)
    return pts[perm].to(dev)                      # [P, 2] scrambled


def _make_batch(dev):
    P = 11
    # Three fixed scramble permutations of 0..10 with distinct env shapes/centroids.
    s0 = [3, 7, 0, 10, 4, 1, 9, 2, 6, 8, 5]
    s1 = [10, 0, 5, 2, 8, 6, 1, 9, 3, 7, 4]
    s2 = [1, 6, 9, 4, 0, 7, 2, 10, 5, 3, 8]
    envs = [
        _scrambled_env(P, cx=0.0, cy=0.0, r=2.0, scramble=s0, dev=dev),
        _scrambled_env(P, cx=5.0, cy=-3.0, r=1.0, scramble=s1, dev=dev),
        _scrambled_env(P, cx=-4.0, cy=8.0, r=3.5, scramble=s2, dev=dev),
    ]
    return torch.stack(envs, dim=0)               # [3, 11, 2]


@pytest.mark.parametrize("dev", DEVS)
def test_ccw_sort_matches_oracle(dev):
    pts = _make_batch(dev)
    got = wpl.ccw_sort(pts)
    ref = geometry.ccw_sort(pts)
    # Output is a PERMUTATION of the input (no value arithmetic) -> byte-exact.
    assert torch.equal(got.cpu(), ref.cpu())


@pytest.mark.parametrize("dev", DEVS)
def test_ccw_sort_keys_monotone(dev):
    pts = _make_batch(dev)
    got = wpl.ccw_sort(pts)
    # Recompute the per-point key atan2(dx, dy) (X FIRST) from the output and
    # assert it is non-decreasing along P within each env.
    mean = got.mean(dim=1, keepdim=True)
    d = got - mean
    keys = torch.arctan2(d[:, :, 0], d[:, :, 1])  # [E, P]
    diffs = keys[:, 1:] - keys[:, :-1]
    assert (diffs >= -1e-5).all(), keys
