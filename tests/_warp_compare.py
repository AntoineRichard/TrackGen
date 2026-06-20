import warp as wp


def to_t(a):
    """wp.array -> torch.Tensor (zero-copy, same device) for oracle comparisons."""
    import torch  # tests are dev-side; torch is available
    return a if a.__class__.__module__.startswith("torch") else wp.to_torch(a)
