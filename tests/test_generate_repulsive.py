"""End-to-end tests for the repulsive-growth first-stage generator.

Repulsive's own properties: a seed-driven obstacle layout, the analytic gradient, a
stride-consistent output at the minimal growth budget, the default-config yield and
compactness (CUDA), a fixed host schedule (no host sync, readback or debug sync in the
growth loop) and an absorbing freeze (so per-env results do not depend on the rest of
the batch). The generator contract (registration, output shape, closed loop, NaN tail,
determinism, diversity, batch independence, graph capture + replay on CUDA) is checked
for every generator by the conformance kit, in tests/test_conformance.py, on the same
small config (``_SMALL``).
"""
from __future__ import annotations
from tests._reference_generators import generate_raw

import numpy as np
import pytest
import torch

pytest.importorskip("warp")

import warp as wp  # noqa: E402

from track_gen._src.types import TrackGenConfig  # noqa: E402
from track_gen._src import warp_generate_repulsive as rep  # noqa: E402
from track_gen._src import runtime  # noqa: E402
from track_gen._src.track_generator import TrackGenerator  # noqa: E402
from track_gen import PerEnvSeededRNG  # noqa: E402
from track_gen.testing.contract import no_host_sync  # noqa: E402
from tests._warp_compare import to_t  # noqa: E402

from tests.conftest import DEVS  # noqa: E402

# A deliberately small config keeps the O(N^2) host-driven optimizer fast enough to run
# the structural contract on cpu; the default-config yield/compactness bar runs on cuda.
_SMALL = dict(num_points=64, repulsive_stages=(16, 32, 64), N_max=384, half_width=0.1)


def _compactness(pts: np.ndarray) -> np.ndarray:
    nxt = np.roll(pts, -1, axis=1)
    perimeter = np.linalg.norm(nxt - pts, axis=2).sum(axis=1)
    area = 0.5 * np.abs(
        (pts[:, :, 0] * nxt[:, :, 1] - nxt[:, :, 0] * pts[:, :, 1]).sum(axis=1)
    )
    return 4.0 * np.pi * area / np.maximum(perimeter * perimeter, 1.0e-12)


def _sample(seeds: np.ndarray, config, dev: str):
    """Run the obstacle kernel; return (obs_pts [E,M,2], obs_mw [E,M]) numpy arrays."""
    s = rep._RepulsiveScalars(config)
    E, M = s.E, s.M
    seeds_wp = wp.array(seeds.astype(np.int32), dtype=wp.int32, device=dev)
    obs_pts = wp.empty(E * M, dtype=wp.vec2f, device=dev)
    obs_mw = wp.empty(E * M, dtype=wp.float32, device=dev)
    rep._sample_obstacles_inplace(seeds_wp, config, obs_pts, obs_mw, dev)
    pts = obs_pts.numpy().reshape(E, M, 2)
    mw = obs_mw.numpy().reshape(E, M)
    return pts, mw, s


@pytest.mark.parametrize("dev", DEVS)
def test_obstacle_layout_deterministic(dev):
    E = 32
    cfg = TrackGenConfig(generator="repulsive", num_envs=E, device=dev)
    seeds_a = np.arange(E) + 7
    seeds_b = np.arange(E) + 1000  # distinct seeds

    pts1, mw1, s = _sample(seeds_a, cfg, dev)
    pts2, mw2, _ = _sample(seeds_a, cfg, dev)
    pts3, mw3, _ = _sample(seeds_b, cfg, dev)

    # Determinism: identical seeds -> byte-identical layout (NaN pattern must match too).
    assert np.array_equal(np.nan_to_num(pts1), np.nan_to_num(pts2))
    assert np.array_equal(mw1, mw2)
    assert np.array_equal(np.isnan(pts1), np.isnan(pts2))
    # Diversity: distinct seeds -> different layout.
    assert not np.array_equal(mw1, mw3)

    n_wall, n_disc, M = s.n_wall, s.n_disc, s.M
    wall_mw = 2.0 * np.pi * s.r_dom / n_wall  # weight 1.0 baked into the wall arc mass

    # Wall columns: finite points on the r_dom ring, obs_mw == wall arc mass (weight 1.0).
    wall_pts = pts1[:, :n_wall]
    wall_mw_arr = mw1[:, :n_wall]
    assert np.isfinite(wall_pts).all()
    assert np.allclose(wall_mw_arr, wall_mw, rtol=1e-4)
    assert np.allclose(np.linalg.norm(wall_pts, axis=2), s.r_dom, rtol=1e-4)

    disc_mw = mw1[:, n_wall:].reshape(E, -1, n_disc)  # [E, K_max, n_disc]
    disc_pts = pts1[:, n_wall:].reshape(E, -1, n_disc, 2)
    # Each disc column block is uniform (all n_disc points share one obs_mw).
    for e in range(E):
        for j in range(disc_mw.shape[1]):
            block = disc_mw[e, j]
            assert np.allclose(block, block[0])
            if block[0] == 0.0:
                # Unused / skipped column: NaN-padded points.
                assert np.isnan(disc_pts[e, j]).all()
            else:
                # Active disc column: recover the ring radius from the chord and confirm
                # the weight factor is 0.25 (obs_mw == 0.25 * 2*pi*r_ring / n_disc).
                ring = disc_pts[e, j]
                chord = np.linalg.norm(ring[1] - ring[0])
                r_ring = chord / (2.0 * np.sin(np.pi / n_disc))
                weight = block[0] * n_disc / (2.0 * np.pi * r_ring)
                assert np.isclose(weight, 0.25, rtol=1e-3)

    # Every finite obstacle point lies within the domain radius.
    finite = np.isfinite(pts1).all(axis=2)
    norms = np.linalg.norm(np.nan_to_num(pts1), axis=2)
    assert (norms[finite] <= s.r_dom + 1e-4).all()
    # At default config every env places at least one inner disc (k_min=8 discs fit).
    assert (mw1[:, n_wall:] > 0.0).any(axis=1).all()


# ===========================================================================
# Analytic gradient validation (hand-written adjoints vs a float64 reference + FD)
# ===========================================================================

_ALPHA, _BETA, _EPS = 3.0, 6.0, 1e-4
_P_EXP = -(_BETA - _ALPHA) / 2.0
_BARRIER_DOMAIN = 3.0
_BARRIER_MARGIN = .05
_BARRIER_CENTERS = np.array([[2.0, 0.1], [-1.9, 1.3]], dtype=np.float64)
_BARRIER_RADII = np.array([.08, .12], dtype=np.float64)


def _safe_dir_np(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-8)


def _energy_ref(x, obs_pts, obs_mw, Lt, Li):
    """float64 total energy E_TP + E_obs + E_len, matching the Warp forward kernels."""
    n = x.shape[0]
    edge = np.roll(x, -1, 0) - x
    tangent = _safe_dir_np(edge)
    el = np.linalg.norm(edge, axis=-1)
    w = 0.5 * (el + np.roll(el, 1))
    idx = np.arange(n)
    dd = np.abs(idx[:, None] - idx[None, :])
    mask = np.minimum(dd, n - dd) > 1
    e_tp = 0.0
    for owner_end in (x, np.roll(x, -1, 0)):
        for partner_end in (x, np.roll(x, -1, 0)):
            diff = partner_end[None, :, :] - owner_end[:, None, :]
            d2 = (diff * diff).sum(-1)
            wedge = diff[..., 0] * tangent[:, None, 1] - diff[..., 1] * tangent[:, None, 0]
            e_tp += (.25 * el[:, None] * el[None, :] * np.abs(wedge) ** _ALPHA /
                     (d2 + _EPS * _EPS) ** (_BETA * .5) * mask).sum()
    dob = x[:, None, :] - obs_pts[None, :, :]
    e_obs = (w[:, None] * obs_mw[None, :] * ((dob * dob).sum(-1) + 1e-8) ** _P_EXP).sum()
    peri = np.linalg.norm(np.roll(x, -1, 0) - x, axis=-1).sum()
    wall_gap = _BARRIER_DOMAIN - _BARRIER_MARGIN - np.linalg.norm(x, axis=-1)
    e_barrier = (w * _BARRIER_DOMAIN ** (1. - (_BETA - _ALPHA)) *
                 -np.log(wall_gap / _BARRIER_DOMAIN)).sum()
    edges = np.roll(x, -1, axis=0) - x
    for center, radius in zip(_BARRIER_CENTERS, _BARRIER_RADII):
        fraction = np.clip(((center - x) * edges).sum(-1) / (edges * edges).sum(-1), 0., 1.)
        gap = np.linalg.norm(x + fraction[:, None] * edges - center, axis=-1) - radius - _BARRIER_MARGIN
        e_barrier += (el * _BARRIER_DOMAIN ** (1. - (_BETA - _ALPHA)) *
                      -np.log(gap / _BARRIER_DOMAIN)).sum()
    return e_tp + e_obs + e_barrier


def _analytic_grad_warp(x, obs_pts, obs_mw, Lt, Li, dev):
    """Launch the three analytic-adjoint kernels for a single env; return grad [N,2]."""
    n, M = x.shape[0], obs_pts.shape[0]
    center = wp.array(x.reshape(n, 2).astype(np.float32), dtype=wp.vec2f, device=dev)
    frozen = wp.zeros(1, dtype=wp.int32, device=dev)
    op = wp.array(obs_pts.astype(np.float32), dtype=wp.vec2f, device=dev)
    ow = wp.array(obs_mw.astype(np.float32), dtype=wp.float32, device=dev)
    wdual = wp.zeros(n, dtype=wp.float32, device=dev)
    obstacle_energy = wp.zeros(n, dtype=wp.float32, device=dev)
    grad = wp.zeros(n, dtype=wp.vec2f, device=dev)
    bg = wp.zeros(n, dtype=wp.vec2f, device=dev)
    edge_bg = wp.zeros(n, dtype=wp.vec2f, device=dev)
    dc = wp.array(_BARRIER_CENTERS.astype(np.float32), dtype=wp.vec2f, device=dev)
    dr = wp.array(_BARRIER_RADII.astype(np.float32), dtype=wp.float32, device=dev)
    margin = wp.full(1, _BARRIER_MARGIN, dtype=wp.float32, device=dev)
    wp.launch(rep._dual_weight_k, dim=(1, n),
              inputs=[center, n, frozen, wdual], device=dev)
    wp.launch(rep._obstacle_energy_k, dim=(1, n),
              inputs=[center, n, op, ow, M, _P_EXP, dc, dr, len(_BARRIER_RADII), _BARRIER_DOMAIN,
                      margin, frozen, obstacle_energy, bg, edge_bg],
              device=dev)
    wp.launch(rep._grad_gather_k, dim=(1, n),
              inputs=[center, n, _ALPHA, _BETA, _EPS, wdual, op, ow, M, _P_EXP,
                      obstacle_energy, bg, edge_bg, frozen, grad], device=dev)
    wp.synchronize()
    return grad.numpy().reshape(n, 2)


@pytest.mark.parametrize("dev", DEVS)
def test_analytic_gradient_matches_reference(dev):
    """The hand-written analytic adjoints reproduce a float64 gradient (and central finite
    differences) on randomized mid-growth states -- the correctness backbone of the tape
    replacement. Validated to fp32 precision (~1e-4 rel) on both CPU and CUDA."""
    rng = np.random.default_rng(20260705)
    worst_rel = 0.0
    for _ in range(5):
        n = int(rng.choice([32, 64]))
        ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
        r = 1.0 + 0.3 * np.sin(3 * ang + rng.uniform(0, 6)) + 0.05 * rng.standard_normal(n)
        x = (np.stack([r * np.cos(ang), r * np.sin(ang)], -1)
             + 0.01 * rng.standard_normal((n, 2))).astype(np.float64)
        M = 20
        obs_pts = 2.0 * rng.standard_normal((M, 2))
        obs_mw = np.abs(rng.standard_normal(M)) + 0.05
        peri = np.linalg.norm(np.roll(x, -1, 0) - x, axis=-1).sum()
        Li = peri
        Lt = peri * (1.0 + rng.uniform(-0.05, 0.1))

        ga = _analytic_grad_warp(x, obs_pts, obs_mw, Lt, Li, dev)

        # (a) float64 reference via central differences on the exact energy
        gref = np.zeros_like(x)
        h = 1e-6
        for i in range(n):
            for d in range(2):
                xp = x.copy(); xp[i, d] += h
                xm = x.copy(); xm[i, d] -= h
                gref[i, d] = (_energy_ref(xp, obs_pts, obs_mw, Lt, Li)
                              - _energy_ref(xm, obs_pts, obs_mw, Lt, Li)) / (2 * h)
        rel = np.linalg.norm(ga - gref) / max(np.linalg.norm(gref), 1e-30)
        worst_rel = max(worst_rel, rel)
        # fp32 analytic vs float64 FD over the big O(N^2) pair sums: ~1e-4 is the fp32 floor.
        assert rel < 3e-3, f"analytic gradient rel err {rel:.2e} on n={n} ({dev})"
    assert worst_rel < 3e-3


# ===========================================================================
# Task 4: repulsive-specific checks through the tail (the generator contract itself is
# checked for every generator by tests/test_conformance.py)
# ===========================================================================

@pytest.mark.parametrize("dev", DEVS)
def test_repulsive_stride_consistent_minimal_budget(dev):
    """Regression for the n_iters<=0 / incomplete-stage-schedule mis-stride bug.

    A valid config with the smallest legal growth budget (grow_mult just above 1, the minimum
    settle_iters=1) must still emit a stride-consistent ``[E*num_points]`` centerline: every env
    is a SINGLE closed loop of num_points, not a pack of several coarse-stage circles packed into
    the first quarter (env-0 reading four envs' data) with the later envs left as stale zeros.
    """
    E = 8
    cfg = TrackGenConfig(generator="repulsive", num_envs=E, device=dev,
                         num_points=64, repulsive_stages=(16, 32, 64), N_max=384,
                         half_width=0.1, repulsive_grow_mult_min=1.1,
                         repulsive_grow_mult_max=1.3, repulsive_settle_iters=1)
    N = int(cfg.num_points)
    gen = TrackGenerator(cfg, PerEnvSeededRNG(seeds=1, num_envs=E, device=dev))
    gen.generate(E)
    c = to_t(gen._scratch.gen_centerline).view(E, N, 2)
    assert c.shape == (E, N, 2)
    assert torch.isfinite(c).all()
    steps = (c[:, 1:] - c[:, :-1]).norm(dim=-1)
    med = steps.median(dim=1).values
    # Single closed loop: last->first gap is ~one step, and no giant jump between packed
    # sub-circles (a mis-strided env would blow past both thresholds).
    gap = (c[:, 0] - c[:, -1]).norm(dim=-1)
    assert torch.all(gap <= 3.0 * med)
    assert torch.all(steps.max(dim=1).values <= 6.0 * med)
    # No env collapsed onto the stale-zero origin: every env has real radial extent.
    assert torch.all(c.norm(dim=-1).amax(dim=1) > 1e-3)


@pytest.mark.cuda
def test_repulsive_default_yield_and_shape_cuda():
    """CUDA default-yield and noncircular-shape regression.

    Every accepted finished road must also satisfy the permanent barrier checker.
    The minimum yield remains a regression gate; platform measurements accompany
    intentional algorithm/default changes rather than a frozen numeric oracle.
    """
    E = 64
    cfg = TrackGenConfig(generator="repulsive", num_envs=E, device="cuda")
    gen = TrackGenerator(cfg, PerEnvSeededRNG(seeds=11, num_envs=E, device="cuda"))
    track = gen.generate(E)
    assert gen._graph is not None  # captured (fixed-schedule loop, no host readback)

    valid = to_t(track.valid).bool()
    yield_frac = float(valid.float().mean())
    assert yield_frac > 0.5, f"post-tail yield {yield_frac:.3f} <= 0.5 (design contract)"
    assert int(valid.sum()) >= 58, f"yield {int(valid.sum())}/64 below the default-yield regression gate"

    N = int(cfg.num_points)
    gen_center = to_t(gen._scratch.gen_centerline).cpu().numpy().reshape(E, N, 2)
    compactness = _compactness(gen_center)
    # Serpentine/foldy band: well below the round-blob gate (0.85).
    assert float(np.median(compactness)) < 0.85, (
        f"median compactness {float(np.median(compactness)):.3f} not in the foldy band")


# ===========================================================================
# Capture readiness: fixed schedule, absorbing freeze
# ===========================================================================

def test_repulsive_generate_has_no_host_sync_or_readback(monkeypatch):
    """The growth loop follows a fixed host schedule: no device->host readback, no host
    sync. Both are illegal inside a CUDA graph capture; on cpu we prove their absence by
    running the generator body under the conformance kit's ``no_host_sync`` guard. On top
    of the contract, repulsive's loop never even calls the debug sync that every
    ``ctx.sync()`` funnels into (``runtime._sync``)."""
    E = 4
    cfg = TrackGenConfig(generator="repulsive", num_envs=E, device="cpu", **_SMALL)
    scratch = rep._alloc_repulsive_scratch(cfg)
    seeds = wp.array(np.arange(E, dtype=np.int32) + 3, dtype=wp.int32, device="cpu")
    out = wp.empty(E * int(cfg.num_points), dtype=wp.vec2f, device="cpu")
    valid = wp.empty(E, dtype=wp.int32, device="cpu")

    def forbidden(*args, **kwargs):
        raise AssertionError("debug sync inside generate_repulsive_warp")

    # The generator's own module no longer holds a warp_pipeline reference (it is built
    # from track_gen.ops); the debug sync every ctx.sync() funnels into is runtime._sync.
    monkeypatch.setattr(runtime, "_sync", forbidden)
    size = wp.full(E, 1.0, dtype=wp.float32, device="cpu")
    from track_gen.exec import LaunchContext
    with no_host_sync():
        rep._generate_repulsive(seeds, cfg, out, valid, scratch,
                                LaunchContext.for_call("cpu"), size=size)
    monkeypatch.undo()
    assert np.isfinite(out.numpy()).all()
    assert (valid.numpy() == 1).all()


def _grow(seeds: list[int], dev: str = "cpu"):
    """Run the generator through the tail for explicit per-env seeds; return
    (gen_centerline [E,N,2], frozen [E], valid [E]) as numpy arrays."""
    E = len(seeds)
    cfg = TrackGenConfig(generator="repulsive", num_envs=E, device=dev, **_SMALL)
    seeds_wp = wp.array(np.asarray(seeds, dtype=np.int32), dtype=wp.int32, device=dev)
    gen = TrackGenerator(cfg, PerEnvSeededRNG(seeds=seeds_wp, num_envs=E, device=dev))
    track = gen.generate(E)
    N = int(cfg.num_points)
    return (gen._scratch.gen_centerline.numpy().reshape(E, N, 2),
            gen._scratch.gen.scratch["scratch:generator/frozen"].numpy(),
            track.valid.numpy())


@pytest.mark.parametrize("dev", DEVS)
def test_repulsive_raw_and_freeze_state_are_batch_independent(dev):
    # Default bounded runs may reach their target without optimizer equilibrium.
    # The deterministic controlled-loop tests prove absorbing early-stop behavior;
    # this public-API check covers real seed trajectories without a skip/search.
    big_seeds = [0, 1, 3, 7, 11, 23]
    big_c, big_frozen, big_valid = _grow(big_seeds, dev)
    rows = np.array([0, 3, 5])
    small_c, small_frozen, small_valid = _grow([big_seeds[i] for i in rows], dev)
    assert np.array_equal(small_c, big_c[rows])
    assert np.array_equal(small_frozen, big_frozen[rows])
    assert np.array_equal(small_valid, big_valid[rows])
