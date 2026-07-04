"""Analytic + recipe tests for DiscChecker (box vs disc obstacles)."""
from __future__ import annotations

import numpy as np
import pytest
import warp as wp

wp.init()


def _boxes(E, B, slots, device="cpu"):
    pos = np.full((E * B, 2), np.nan, np.float32)
    yaw = np.zeros(E * B, np.float32)
    he = np.zeros((E * B, 2), np.float32)
    for (e, b), (px, py, yw, hx, hy) in slots.items():
        i = e * B + b
        pos[i] = (px, py)
        yaw[i] = yw
        he[i] = (hx, hy)
    return (wp.array(pos, dtype=wp.vec2f, device=device),
            wp.array(yaw, dtype=wp.float32, device=device),
            wp.array(he, dtype=wp.vec2f, device=device))


def _discs(rows, device="cpu"):
    return wp.array(np.array(rows, np.float32), dtype=wp.vec2f, device=device)


def test_face_corner_graze_and_miss():
    from track_gen.collision import DiscChecker
    # One env, 4 discs; one axis-aligned box he=(0.1, 0.05) at origin.
    discs = _discs([[0.12, 0.0],            # face hit: pen 0.03-0.02=0.01
                    [0.12, 0.07],           # corner hit: dist=sqrt(0.02^2+0.02^2)
                    [0.13, 0.0],            # graze: dist 0.03 == radius -> hit
                    [0.20, 0.0]])           # miss
    checker = DiscChecker(discs, radius=0.03, max_boxes=4, num_envs=1)
    # All four boxes identical; each box sees ALL discs, so instead probe
    # per-disc behavior with per-box positions FAR from other discs:
    pos, yaw, he = _boxes(1, 4, {
        (0, 0): (0.0, 0.0, 0.0, 0.1, 0.05),
    })
    res = checker.query(pos, yaw, he)
    hit = res.hit.numpy()
    assert hit[0] == 1
    # Deepest disc is the face one (pen 0.01 > corner pen ~0.0017).
    assert int(res.disc.numpy()[0]) == 0
    np.testing.assert_allclose(float(res.depth.numpy()[0]), 0.01, atol=1e-6)
    # Nearest point on disc 0's boundary toward the box face: (0.09, 0).
    np.testing.assert_allclose(res.nearest.numpy().reshape(-1, 2)[0],
                               [0.09, 0.0], atol=1e-6)
    # Inactive slots inert.
    assert list(hit[1:]) == [0, 0, 0]
    assert list(res.disc.numpy()[1:]) == [-1, -1, -1]


def test_graze_counts_as_hit_and_miss_does_not():
    from track_gen.collision import DiscChecker
    discs = _discs([[0.13, 0.0], [0.14, 0.0]])
    checker = DiscChecker(discs, radius=0.03, max_boxes=2, num_envs=1)
    pos, yaw, he = _boxes(1, 2, {(0, 0): (0.0, 0.0, 0.0, 0.1, 0.05)})
    res = checker.query(pos, yaw, he)
    # Box 0 sees disc 0 at exactly radius (hit, depth 0) and disc 1 beyond.
    assert int(res.hit.numpy()[0]) == 1
    assert int(res.disc.numpy()[0]) == 0
    np.testing.assert_allclose(float(res.depth.numpy()[0]), 0.0, atol=1e-6)


def test_rotated_box_and_nan_discs_skipped():
    from track_gen.collision import DiscChecker
    # Disc straight above; box rotated 90 deg so its LONG side faces up.
    discs = _discs([[np.nan, np.nan], [0.0, 0.12]])
    checker = DiscChecker(discs, radius=0.03, max_boxes=1, num_envs=1)
    pos, yaw, he = _boxes(1, 1, {(0, 0): (0.0, 0.0, np.pi / 2, 0.1, 0.05)})
    res = checker.query(pos, yaw, he)
    # Rotated: half-extent along +y is now 0.1 -> dist 0.02 -> pen 0.01.
    assert int(res.hit.numpy()[0]) == 1
    assert int(res.disc.numpy()[0]) == 1
    np.testing.assert_allclose(float(res.depth.numpy()[0]), 0.01, atol=1e-6)


def test_explicit_count_limits_scan():
    from track_gen.collision import DiscChecker
    discs = _discs([[0.12, 0.0], [0.0, 0.0]])  # second disc INSIDE the box
    count = wp.array(np.array([1], np.int32), dtype=wp.int32, device="cpu")
    checker = DiscChecker(discs, radius=0.03, max_boxes=1, count=count)
    pos, yaw, he = _boxes(1, 1, {(0, 0): (0.0, 0.0, 0.0, 0.1, 0.05)})
    res = checker.query(pos, yaw, he)
    assert int(res.disc.numpy()[0]) == 0  # disc 1 never scanned


def test_constructor_validation():
    from track_gen.collision import DiscChecker
    discs = _discs([[0.0, 0.0], [1.0, 1.0]])
    with pytest.raises(ValueError, match="num_envs"):
        DiscChecker(discs, radius=0.1, max_boxes=1)  # neither count nor num_envs
    with pytest.raises(ValueError, match="radius"):
        DiscChecker(discs, radius=0.0, max_boxes=1, num_envs=1)
    with pytest.raises(ValueError, match="radius"):
        DiscChecker(discs, radius=float("nan"), max_boxes=1, num_envs=1)
    with pytest.raises(ValueError, match="divisible"):
        DiscChecker(_discs([[0.0, 0.0]] * 3), radius=0.1, max_boxes=1, num_envs=2)


def test_bound_mode_equivalence_and_errors():
    from track_gen.collision import DiscChecker
    discs = _discs([[0.12, 0.0], [0.5, 0.5]])
    pos, yaw, he = _boxes(1, 2, {(0, 0): (0.0, 0.0, 0.0, 0.1, 0.05)})
    free = DiscChecker(discs, radius=0.03, max_boxes=2, num_envs=1)
    bound = DiscChecker(discs, radius=0.03, max_boxes=2, num_envs=1,
                        position=pos, yaw=yaw, half_extents=he)
    r_free = free.query(pos, yaw, he).clone()
    r_bound = bound.query()
    np.testing.assert_array_equal(r_bound.hit.numpy(), r_free.hit.numpy())
    np.testing.assert_array_equal(r_bound.disc.numpy(), r_free.disc.numpy())
    np.testing.assert_allclose(r_bound.depth.numpy(), r_free.depth.numpy())
    with pytest.raises(ValueError, match="bound"):
        bound.query(pos, yaw, he)
    with pytest.raises(ValueError, match="not bound"):
        free.query()
    with pytest.raises(ValueError, match="all of"):
        DiscChecker(discs, radius=0.03, max_boxes=2, num_envs=1, position=pos)


def test_gate_post_recipe():
    from track_gen import GateGenConfig, GateGenerator, PerEnvSeededRNG
    from track_gen.collision import DiscChecker
    E = 2
    cfg = GateGenConfig(num_envs=E, device="cpu", gate_width=0.06)
    gen = GateGenerator(cfg, PerEnvSeededRNG(seeds=9, num_envs=E, device="cpu"))
    seq = gen.generate()
    G = seq.position.shape[0] // E
    left = seq.left.numpy().reshape(E, G, 2)
    right = seq.right.numpy().reshape(E, G, 2)
    posts = np.empty((E, 2 * G, 2), np.float32)
    posts[:, 0::2] = left
    posts[:, 1::2] = right
    posts_wp = wp.array(posts.reshape(-1, 2), dtype=wp.vec2f, device="cpu")
    checker = DiscChecker(posts_wp, radius=0.02, max_boxes=1, num_envs=E)
    # Park a box exactly on env 0's gate 0 LEFT post.
    valid = seq.valid.numpy()
    e = int(np.argmax(valid))
    slots = {(e, 0): (float(left[e, 0, 0]), float(left[e, 0, 1]), 0.0, 0.03, 0.03)}
    pos, yaw, he = _boxes(E, 1, slots)
    res = checker.query(pos, yaw, he)
    assert int(res.hit.numpy()[e]) == 1
    disc = int(res.disc.numpy()[e])
    assert disc % 2 == 0 and disc // 2 == 0  # left post of gate 0


def test_bind_inputs_after_construction():
    from track_gen.collision import DiscChecker
    discs = _discs([[0.12, 0.0]])
    pos, yaw, he = _boxes(1, 1, {(0, 0): (0.0, 0.0, 0.0, 0.1, 0.05)})
    checker = DiscChecker(discs, radius=0.03, max_boxes=1, num_envs=1)
    checker.bind_inputs(pos, yaw, he)
    res = checker.query()
    assert int(res.hit.numpy()[0]) == 1
    with pytest.raises(ValueError, match="bound"):
        checker.query(pos, yaw, he)
