import pytest
import torch

pytest.importorskip("warp")

from track_gen._src.types import GateGenConfig
from track_gen._src import warp_gate
from tests._warp_compare import to_t


def _manual_sequence(E=1, G=4):
    return warp_gate.alloc_gate_sequence(GateGenConfig(num_envs=E, max_gates=G))


def test_finalize_computes_normals_and_endpoints():
    gates = _manual_sequence()
    pos = to_t(gates.position).view(1, 4, 2)
    tan = to_t(gates.tangent).view(1, 4, 2)
    count = to_t(gates.count)
    pos[0, 0] = torch.tensor([0.0, 0.0])
    pos[0, 1] = torch.tensor([2.0, 0.0])
    tan[0, 0] = torch.tensor([1.0, 0.0])
    tan[0, 1] = torch.tensor([1.0, 0.0])
    count[0] = 2

    cfg = GateGenConfig(max_gates=4, gate_width=2.0, gate_radius=0.0, min_gates=2)
    warp_gate.finalize_gate_sequence(gates, cfg)

    normal = to_t(gates.normal).view(1, 4, 2)
    left = to_t(gates.left).view(1, 4, 2)
    right = to_t(gates.right).view(1, 4, 2)
    valid = to_t(gates.valid).bool()
    assert torch.allclose(normal[0, 0], torch.tensor([0.0, 1.0]), atol=1e-6)
    assert torch.allclose(left[0, 0], torch.tensor([0.0, 1.0]), atol=1e-6)
    assert torch.allclose(right[0, 0], torch.tensor([0.0, -1.0]), atol=1e-6)
    assert valid.tolist() == [True]
    assert torch.isnan(left[0, 2:]).all()


def test_finalize_invalidates_nonfinite_endpoints():
    gates = _manual_sequence()
    pos = to_t(gates.position).view(1, 4, 2)
    tan = to_t(gates.tangent).view(1, 4, 2)
    count = to_t(gates.count)
    pos[0, 0] = torch.tensor([0.0, 0.0])
    pos[0, 1] = torch.tensor([2.0, 0.0])
    tan[0, 0] = torch.tensor([1.0, 0.0])
    tan[0, 1] = torch.tensor([1.0, 0.0])
    count[0] = 2

    cfg = GateGenConfig(
        max_gates=4,
        gate_width=float("nan"),
        gate_radius=0.0,
        min_gates=2,
    )
    warp_gate.finalize_gate_sequence(gates, cfg)

    assert to_t(gates.valid).bool().tolist() == [False]


def test_finalize_invalidates_too_close_gate_centres():
    gates = _manual_sequence()
    pos = to_t(gates.position).view(1, 4, 2)
    tan = to_t(gates.tangent).view(1, 4, 2)
    count = to_t(gates.count)
    pos[0, 0] = torch.tensor([0.0, 0.0])
    pos[0, 1] = torch.tensor([0.01, 0.0])
    tan[0, :2] = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    count[0] = 2

    cfg = GateGenConfig(max_gates=4, min_gates=2, gate_radius=0.025)
    warp_gate.finalize_gate_sequence(gates, cfg)

    assert to_t(gates.valid).bool().tolist() == [False]


def test_finalize_uses_gate_radius_as_sphere_distance():
    gates = _manual_sequence()
    pos = to_t(gates.position).view(1, 4, 2)
    tan = to_t(gates.tangent).view(1, 4, 2)
    count = to_t(gates.count)
    pos[0, 0] = torch.tensor([0.0, 0.0])
    pos[0, 1] = torch.tensor([0.15, 0.0])
    tan[0, :2] = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    count[0] = 2

    cfg = GateGenConfig(
        max_gates=4,
        min_gates=2,
        gate_radius=0.1,
    )
    warp_gate.finalize_gate_sequence(gates, cfg)

    assert to_t(gates.valid).bool().tolist() == [False]


def test_finalize_invalidates_crossing_gate_segments():
    gates = _manual_sequence()
    pos = to_t(gates.position).view(1, 4, 2)
    tan = to_t(gates.tangent).view(1, 4, 2)
    count = to_t(gates.count)
    pos[0, 0] = torch.tensor([0.0, 0.0])
    pos[0, 1] = torch.tensor([0.0, 0.5])
    tan[0, 0] = torch.tensor([1.0, 0.0])
    tan[0, 1] = torch.tensor([0.0, 1.0])
    count[0] = 2

    cfg = GateGenConfig(max_gates=4, min_gates=2, gate_width=2.0, gate_radius=0.0)
    warp_gate.finalize_gate_sequence(gates, cfg)

    assert to_t(gates.valid).bool().tolist() == [False]


def test_order_points_ccw_sorts_by_centroid_angle():
    import warp as wp

    count = wp.array([4], dtype=wp.int32, device="cpu")
    seeds = wp.array([123], dtype=wp.int32, device="cpu")
    src = wp.array(
        [wp.vec2f(1.0, 0.0), wp.vec2f(0.0, 0.0), wp.vec2f(1.0, 1.0), wp.vec2f(0.0, 1.0)],
        dtype=wp.vec2f,
        device="cpu",
    )
    keys = wp.empty(4, dtype=wp.float32, device="cpu")
    out = wp.empty(4, dtype=wp.vec2f, device="cpu")

    warp_gate.order_points(seeds, src, 4, count, 4, "ccw", keys, out)

    expected = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]])
    assert torch.equal(to_t(out), expected)


def test_order_points_random_pairs_preserves_pair_adjacency_and_singleton():
    import warp as wp

    count = wp.array([5], dtype=wp.int32, device="cpu")
    seeds = wp.array([123], dtype=wp.int32, device="cpu")
    src = wp.array(
        [
            wp.vec2f(10.0, 0.0),
            wp.vec2f(10.0, 1.0),
            wp.vec2f(20.0, 0.0),
            wp.vec2f(20.0, 1.0),
            wp.vec2f(30.0, 0.0),
            wp.vec2f(99.0, 99.0),
        ],
        dtype=wp.vec2f,
        device="cpu",
    )
    keys = wp.empty(6, dtype=wp.float32, device="cpu")
    out_a = wp.empty(6, dtype=wp.vec2f, device="cpu")
    out_b = wp.empty(6, dtype=wp.vec2f, device="cpu")

    warp_gate.order_points(seeds, src, 6, count, 6, "random_pairs", keys, out_a)
    warp_gate.order_points(seeds, src, 6, count, 6, "random_pairs", keys, out_b)

    ordered = to_t(out_a)
    ordered_b = to_t(out_b)
    assert torch.equal(ordered[:5], ordered_b[:5])
    assert torch.isnan(ordered[5]).all()
    assert torch.isnan(ordered_b[5]).all()
    for pair_id in (10.0, 20.0):
        idx = torch.where(ordered[:, 0] == pair_id)[0]
        assert idx.numel() == 2
        assert abs(int(idx[0]) - int(idx[1])) == 1
    assert torch.where(ordered[:, 0] == 30.0)[0].numel() == 1


def test_normalize_positions_centers_scales_and_pads():
    gates = _manual_sequence()
    pos = to_t(gates.position).view(1, 4, 2)
    count = to_t(gates.count)
    pos[0, 0] = torch.tensor([0.0, 0.0])
    pos[0, 1] = torch.tensor([2.0, 0.0])
    pos[0, 2] = torch.tensor([1.0, 1.0])
    count[0] = 3

    warp_gate.normalize_positions(gates.position, 4, gates.count, 4.0)

    out = to_t(gates.position).view(1, 4, 2)
    expected = torch.tensor([[-2.0, -1.0], [2.0, -1.0], [0.0, 1.0]])
    assert torch.allclose(out[0, :3], expected, atol=1e-6)
    assert torch.isnan(out[0, 3]).all()


def test_relax_gate_spheres_separates_overlapping_centres():
    gates = _manual_sequence()
    pos = to_t(gates.position).view(1, 4, 2)
    count = to_t(gates.count)
    pos[0, 0] = torch.tensor([0.0, 0.0])
    pos[0, 1] = torch.tensor([0.01, 0.0])
    count[0] = 2

    warp_gate.relax_gate_spheres(gates.position, 4, gates.count, 0.25, 1)

    out = to_t(gates.position).view(1, 4, 2)
    dist = torch.linalg.norm(out[0, 1] - out[0, 0])
    assert dist >= 0.25 - 1e-6
    assert torch.allclose(out[0, :2].mean(dim=0), torch.tensor([0.005, 0.0]), atol=1e-6)


def test_relax_gate_spheres_handles_coincident_centres_deterministically():
    gates = _manual_sequence()
    pos = to_t(gates.position).view(1, 4, 2)
    count = to_t(gates.count)
    pos[0, 0] = torch.tensor([0.0, 0.0])
    pos[0, 1] = torch.tensor([0.0, 0.0])
    count[0] = 2

    warp_gate.relax_gate_spheres(gates.position, 4, gates.count, 0.2, 1)

    out = to_t(gates.position).view(1, 4, 2)
    assert torch.isfinite(out[0, :2]).all()
    dist = torch.linalg.norm(out[0, 1] - out[0, 0])
    assert dist >= 0.2 - 1e-6


def test_relax_gate_spheres_needs_multiple_passes_for_three_gate_cluster():
    target = 0.2

    def run(iterations):
        gates = _manual_sequence()
        pos = to_t(gates.position).view(1, 4, 2)
        count = to_t(gates.count)
        pos[0, 0] = torch.tensor([0.0, 0.0])
        pos[0, 1] = torch.tensor([0.01, 0.0])
        pos[0, 2] = torch.tensor([0.0, 0.01])
        count[0] = 3
        warp_gate.relax_gate_spheres(gates.position, 4, gates.count, target, iterations)
        out = to_t(gates.position).view(1, 4, 2)[0, :3]
        dist = torch.cdist(out, out)
        return dist[dist > 0].min()

    assert run(1) < target - 1e-3
    assert run(8) >= target - 1e-6


def test_tangents_from_positions_uses_wrapped_central_difference():
    gates = _manual_sequence()
    pos = to_t(gates.position).view(1, 4, 2)
    count = to_t(gates.count)
    pos[0] = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    count[0] = 4

    warp_gate.tangents_from_positions(gates.position, gates.tangent, 4, gates.count)

    tangent = to_t(gates.tangent).view(1, 4, 2)
    expected = torch.tensor([[1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0]])
    assert torch.allclose(tangent[0], expected, atol=1e-6)


def test_two_gate_tangents_are_nonzero_and_finalize_valid():
    gates = _manual_sequence()
    pos = to_t(gates.position).view(1, 4, 2)
    count = to_t(gates.count)
    pos[0, 0] = torch.tensor([0.0, 0.0])
    pos[0, 1] = torch.tensor([2.0, 0.0])
    count[0] = 2

    warp_gate.tangents_from_positions(gates.position, gates.tangent, 4, gates.count)
    cfg = GateGenConfig(max_gates=4, gate_width=2.0, gate_radius=0.0, min_gates=2)
    warp_gate.finalize_gate_sequence(gates, cfg)

    tangent = to_t(gates.tangent).view(1, 4, 2)
    left = to_t(gates.left).view(1, 4, 2)
    right = to_t(gates.right).view(1, 4, 2)
    assert to_t(gates.valid).bool().tolist() == [True]
    assert torch.allclose(tangent[0, 0], torch.tensor([1.0, 0.0]), atol=1e-6)
    assert torch.allclose(tangent[0, 1], torch.tensor([-1.0, 0.0]), atol=1e-6)
    assert torch.isfinite(left[0, :2]).all()
    assert torch.isfinite(right[0, :2]).all()
    endpoint_width = torch.linalg.norm(left[0, :2] - right[0, :2], dim=-1)
    assert torch.allclose(endpoint_width, torch.full((2,), 2.0), atol=1e-6)


def test_order_points_raw_ccw_and_random_pairs_are_deterministic():
    import warp as wp

    cfg = GateGenConfig(num_envs=1, max_gates=4)
    count = wp.array([4], dtype=wp.int32, device="cpu")
    seeds = wp.array([123], dtype=wp.int32, device="cpu")
    src = wp.array(
        [wp.vec2f(0.0, 0.0), wp.vec2f(0.0, 1.0), wp.vec2f(1.0, 0.0), wp.vec2f(1.0, 1.0)],
        dtype=wp.vec2f,
        device="cpu",
    )
    keys = wp.empty(4, dtype=wp.float32, device="cpu")
    out_raw = wp.empty(4, dtype=wp.vec2f, device="cpu")
    out_rand_a = wp.empty(4, dtype=wp.vec2f, device="cpu")
    out_rand_b = wp.empty(4, dtype=wp.vec2f, device="cpu")

    warp_gate.order_points(seeds, src, 4, count, 4, "raw", keys, out_raw)
    warp_gate.order_points(seeds, src, 4, count, 4, "random_pairs", keys, out_rand_a)
    warp_gate.order_points(seeds, src, 4, count, 4, "random_pairs", keys, out_rand_b)

    assert torch.equal(to_t(out_raw), to_t(src))
    assert torch.equal(to_t(out_rand_a), to_t(out_rand_b))
