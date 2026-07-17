"""End-to-end tests for the 3D gate pipeline: Z profile wiring, gate_align
frame modes, and grade validity (Task 4)."""
import numpy as np
import pytest
import warp as wp

from track_gen._src.gate_generator import GateGenerator
from track_gen._src.rng_utils import PerEnvSeededRNG
from track_gen._src.types import GateGenConfig

E = 8


def _gen(**kw):
    cfg = GateGenConfig(device="cpu", num_envs=E, gate_width=0.05, **kw)
    rng = PerEnvSeededRNG(seeds=42, num_envs=E, device="cpu")
    g = GateGenerator(cfg, rng)
    return g, g.generate(), cfg


def _valid_gates(seq, cfg, e):
    n = int(seq.count.numpy()[e])
    G = int(cfg.max_gates)
    sl = slice(e * G, e * G + n)
    return (seq.position.numpy()[sl], seq.tangent.numpy()[sl],
            seq.orientation.numpy()[sl], seq.left.numpy()[sl],
            seq.right.numpy()[sl])


def test_z_profile_reaches_positions():
    _, seq, cfg = _gen(z_profile="uniform", z_min=1.0, z_max=2.0)
    valid = seq.valid.numpy()
    assert valid.any()
    for e in np.flatnonzero(valid):
        p, _, _, _, _ = _valid_gates(seq, cfg, e)
        assert (p[:, 2] >= 1.0 - 1e-5).all() and (p[:, 2] <= 2.0 + 1e-5).all()
        assert p[:, 2].std() > 0.0


def test_yaw_only_gates_stay_upright():
    _, seq, cfg = _gen(z_profile="uniform", z_min=0.5, z_max=3.0,
                       gate_align="yaw_only")
    for e in np.flatnonzero(seq.valid.numpy()):
        p, _, _, l, r = _valid_gates(seq, cfg, e)
        # posts horizontal: left/right at the same altitude as the center
        np.testing.assert_allclose(l[:, 2], p[:, 2], atol=1e-5)
        np.testing.assert_allclose(r[:, 2], p[:, 2], atol=1e-5)


def test_full_tangent_follows_slope():
    _, seq, cfg = _gen(z_profile="random_walk", z_base=1.5, z_min=0.5,
                       z_max=3.0, z_max_step=0.5, gate_align="full_tangent")
    saw_tilt = False
    for e in np.flatnonzero(seq.valid.numpy()):
        p, t, q, _, _ = _valid_gates(seq, cfg, e)
        # orientation x-axis == unit tangent (full alignment)
        for i in range(len(p)):
            x, y, z, w = q[i]
            fwd = np.array([
                1 - 2 * (y * y + z * z),
                2 * (x * y + w * z),
                2 * (x * z - w * y)])
            np.testing.assert_allclose(fwd, t[i], atol=1e-4)
            if abs(t[i][2]) > 1e-3:
                saw_tilt = True
    assert saw_tilt


def test_grade_validity_flags_steep_uniform():
    _, seq_off, _ = _gen(z_profile="uniform", z_min=0.0, z_max=50.0)
    _, seq_on, _ = _gen(z_profile="uniform", z_min=0.0, z_max=50.0,
                        z_valid_grade=0.5)
    # absurd z range: with the check on, strictly fewer (realistically zero)
    # envs stay valid
    assert seq_on.valid.numpy().sum() < max(1, seq_off.valid.numpy().sum())


def _run_validity(positions, tangents, cnt, grade, gate_width=0.0,
                  center_distance=0.05, min_gates=4):
    """Launch _finalize_validity_k on a hand-built single-env course and
    return the resulting valid flag. Fields other than position/tangent are
    filled with valid (finite, non-crossing) placeholders so only the grade
    check can flip validity."""
    from track_gen._src import warp_gate

    warp_gate._init()
    G = positions.shape[0]
    pos = wp.array(positions.astype(np.float32), dtype=wp.vec3f, device="cpu")
    tan = wp.array(tangents.astype(np.float32), dtype=wp.vec3f, device="cpu")
    ident = np.tile(np.array([0.0, 0.0, 0.0, 1.0], np.float32), (G, 1))
    orient = wp.array(ident, dtype=wp.quatf, device="cpu")
    half = wp.zeros(G, dtype=wp.float32, device="cpu")
    # posts coincide with centers (gate_width 0) => no crossings
    left = wp.array(positions.astype(np.float32), dtype=wp.vec3f, device="cpu")
    right = wp.array(positions.astype(np.float32), dtype=wp.vec3f, device="cpu")
    count = wp.array(np.array([cnt], np.int32), dtype=wp.int32, device="cpu")
    valid = wp.zeros(1, dtype=wp.int32, device="cpu")
    wp.launch(
        warp_gate._finalize_validity_k,
        dim=1,
        inputs=[pos, tan, orient, half, left, right, count, G,
                int(min_gates), float(center_distance), float(gate_width),
                float(grade), valid],
        device="cpu",
    )
    return int(valid.numpy()[0])


def test_grade_validity_flags_closing_chord():
    # Footgun 3: the grade check wraps around (j = (i+1) % cnt). Build a course
    # that is FLAT on every open chord but steep across the closing chord
    # (gate n-1 -> gate 0). With the grade check off it is valid; turning the
    # check on must flag it invalid, and the ONLY steep chord is the closing
    # one, so the wraparound term is the sole thing that can flip it.
    xy = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    zs = np.array([0.0, 0.0, 0.0, 5.0])  # closing chord p3->p0: dz=5 over ds=1
    positions = np.column_stack([xy, zs])
    # non-degenerate 3D tangents (full norm > 0) so the tangent-length check
    # passes regardless of the grade check
    tangents = np.tile(np.array([1.0, 0.0, 0.0]), (4, 1))
    assert _run_validity(positions, tangents, cnt=4, grade=0.0) == 1
    assert _run_validity(positions, tangents, cnt=4, grade=1.0) == 0


def test_grade_validity_open_chord_still_flagged():
    # Control: a steep OPEN chord (p1->p2) is flagged too, confirming the loop
    # covers interior chords, not only the closing one.
    xy = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    zs = np.array([0.0, 0.0, 5.0, 5.0])  # open chord p1->p2: dz=5 over ds=1
    positions = np.column_stack([xy, zs])
    tangents = np.tile(np.array([1.0, 0.0, 0.0]), (4, 1))
    assert _run_validity(positions, tangents, cnt=4, grade=0.0) == 1
    assert _run_validity(positions, tangents, cnt=4, grade=1.0) == 0


def test_near_vertical_tangent_stays_valid():
    # Footgun 1: full_tangent on a steep segment yields a near-vertical 3D
    # tangent whose XY norm is tiny. The tangent-length validity check must use
    # the full 3D norm, or such a gate is wrongly flagged invalid.
    xy = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    zs = np.array([0.0, 1.0, 1.0, 0.0])
    positions = np.column_stack([xy, zs])
    # gate 0 has an almost purely vertical tangent (tiny XY component): its 3D
    # norm is ~1 but its XY norm is 1e-7, so an XY-only length check would
    # flag it. Other gates have plain horizontal tangents.
    tangents = np.array([
        [1e-7, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ])
    # grade off: geometry is otherwise valid; the near-vertical tangent must
    # not by itself invalidate the env.
    assert _run_validity(positions, tangents, cnt=4, grade=0.0) == 1


def test_flat_default_matches_2d_goldens():
    # z_profile default is "flat" with z_base 0: Task 1 goldens still hold
    # (also covered by test_golden_migration, asserted here for locality).
    _, seq, cfg = _gen()
    for e in np.flatnonzero(seq.valid.numpy()):
        p, _, _, _, _ = _valid_gates(seq, cfg, e)
        np.testing.assert_allclose(p[:, 2], 0.0, atol=0.0)
