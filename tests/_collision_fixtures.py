"""Synthetic annulus Track + box-input builders for collision tests."""
from __future__ import annotations

import numpy as np
import warp as wp

from track_gen._src.types import Track


def make_annulus_track(E=1, n=512, N_max=None, r_center=1.0, half_width=0.3,
                       counts=None, device="cpu"):
    """Concentric-circle Track batch: inner radius r-hw, outer r+hw, CCW."""
    wp.init()
    if N_max is None:
        N_max = n + 8  # NaN tail exercises count-aware kernels
    counts = [n] * E if counts is None else list(counts)
    assert len(counts) == E and max(counts) <= N_max
    ri, ro = r_center - half_width, r_center + half_width
    names = ("outer", "center", "inner", "tangent", "normal")
    # vec3f fields, z = 0 on real rows (planar pipeline); padding rows all-NaN.
    fields = {k: np.full((E, N_max, 3), np.nan, np.float32) for k in names}
    arclen = np.full((E, N_max), np.nan, np.float32)
    length = np.zeros(E, np.float32)
    for e, m in enumerate(counts):
        th = np.linspace(0.0, 2.0 * np.pi, m, endpoint=False)
        radial = np.stack([np.cos(th), np.sin(th)], axis=1)
        fields["center"][e, :m, :2] = r_center * radial
        fields["outer"][e, :m, :2] = ro * radial
        fields["inner"][e, :m, :2] = ri * radial
        fields["tangent"][e, :m, :2] = np.stack([-np.sin(th), np.cos(th)], axis=1)
        fields["normal"][e, :m, :2] = radial
        for k in names:
            fields[k][e, :m, 2] = 0.0
        step = 2.0 * r_center * np.sin(np.pi / m)  # chord length
        arclen[e, :m] = step * np.arange(m)
        length[e] = step * m

    def v3(a):
        return wp.array(a.reshape(-1, 3), dtype=wp.vec3f, device=device)

    return Track(
        outer=v3(fields["outer"]), center=v3(fields["center"]),
        inner=v3(fields["inner"]), tangent=v3(fields["tangent"]),
        normal=v3(fields["normal"]),
        arclen=wp.array(arclen.reshape(-1), dtype=wp.float32, device=device),
        length=wp.array(length, dtype=wp.float32, device=device),
        valid=wp.array(np.ones(E, np.int32), dtype=wp.int32, device=device),
        count=wp.array(np.array(counts, np.int32), dtype=wp.int32, device=device),
        # theta increases 0 -> 2*pi, so every annulus loop winds CCW (+1.0).
        winding=wp.array(np.ones(E, np.float32), dtype=wp.float32, device=device),
    )


def annulus_polylines(track, e, N_max):
    """Real (non-NaN) inner/outer xy polylines of env e as numpy [m, 2] arrays."""
    m = int(track.count.numpy()[e])
    inner = track.inner.numpy().reshape(-1, 3)[e * N_max:e * N_max + m, :2]
    outer = track.outer.numpy().reshape(-1, 3)[e * N_max:e * N_max + m, :2]
    return inner.astype(np.float64), outer.astype(np.float64)


def yaw_quats(yaw):
    """Yaw angles [n] -> quatf components (x, y, z, w) about +z, float32 [n, 4]."""
    yaw = np.asarray(yaw, np.float64)
    q = np.zeros((yaw.shape[0], 4), np.float32)
    q[:, 2] = np.sin(0.5 * yaw)
    q[:, 3] = np.cos(0.5 * yaw)
    return q


def make_boxes(E, B, slots, device="cpu"):
    """Box input arrays [E*B]; slots {(e,b): (px, py, yaw, hx, hy)}; rest inactive.

    Returns (position vec3f (z = 0), orientation quatf (yaw about +z),
    half_extents vec2f)."""
    pos = np.full((E * B, 3), np.nan, np.float32)
    yaw = np.zeros(E * B, np.float32)
    he = np.zeros((E * B, 2), np.float32)
    for (e, b), (px, py, yw, hx, hy) in slots.items():
        i = e * B + b
        pos[i] = (px, py, 0.0)
        yaw[i] = yw
        he[i] = (hx, hy)
    return (wp.array(pos, dtype=wp.vec3f, device=device),
            wp.array(yaw_quats(yaw), dtype=wp.quatf, device=device),
            wp.array(he, dtype=wp.vec2f, device=device))
