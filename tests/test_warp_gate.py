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

    cfg = GateGenConfig(max_gates=4, gate_width=2.0, min_gate_distance=0.0, min_gates=2)
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


def test_finalize_invalidates_too_close_gate_centres():
    gates = _manual_sequence()
    pos = to_t(gates.position).view(1, 4, 2)
    tan = to_t(gates.tangent).view(1, 4, 2)
    count = to_t(gates.count)
    pos[0, 0] = torch.tensor([0.0, 0.0])
    pos[0, 1] = torch.tensor([0.01, 0.0])
    tan[0, :2] = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    count[0] = 2

    cfg = GateGenConfig(max_gates=4, min_gates=2, min_gate_distance=0.05)
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

    cfg = GateGenConfig(max_gates=4, min_gates=2, gate_width=2.0, min_gate_distance=0.0)
    warp_gate.finalize_gate_sequence(gates, cfg)

    assert to_t(gates.valid).bool().tolist() == [False]


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
