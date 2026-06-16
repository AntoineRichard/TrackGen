import torch

from track_gen.geometry import segment_directions


def test_unit_square_edges_are_axis_aligned_unit_dirs():
    sq = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]])
    dirs = segment_directions(sq, closed=True)
    assert dirs.shape == sq.shape
    expected = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]]
    )
    assert torch.allclose(dirs, expected, atol=1e-6)
    norms = torch.linalg.norm(dirs, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)


def test_open_chain_last_dir_is_zero():
    pts = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]])
    dirs = segment_directions(pts, closed=False)
    assert torch.allclose(dirs[0, 0], torch.tensor([1.0, 0.0]), atol=1e-6)
    assert torch.allclose(dirs[0, 1], torch.tensor([1.0, 0.0]), atol=1e-6)
    assert torch.allclose(dirs[0, 2], torch.tensor([0.0, 0.0]), atol=1e-6)
