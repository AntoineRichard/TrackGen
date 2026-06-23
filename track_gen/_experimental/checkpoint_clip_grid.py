"""Render a 5x5 checkpoint-clip grid for visual inspection (DEV ONLY).

Run directly to write ``viz/out/checkpoint_k2clip_grid.png``. Importing this module is
side-effect free so it can safely live under the packaged ``track_gen._experimental``
namespace.
"""
from __future__ import annotations

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_OUTPUT = _ROOT / "viz" / "out" / "checkpoint_k2clip_grid.png"


def render_grid(output: str | Path | None = None) -> Path:
    """Render the checkpoint clip comparison grid and return the output path."""
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402

    from benchmarks.track_metrics import (  # noqa: E402
        compactness,
        self_intersects,
        straight_fraction,
    )
    from track_gen._experimental import checkpoint_proto as cp  # noqa: E402

    cfg = cp.DEFAULTS.copy()
    out = Path(output) if output is not None else _DEFAULT_OUTPUT
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(5, 5, figsize=(15, 15))
    for i, ax in enumerate(axes.flat):
        p, _ = cp.generate_centerline_clip(i, cfg, K=2)
        cpoly = np.vstack([p, p[0]])
        crossing = self_intersects(p)
        ax.plot(cpoly[:, 0], cpoly[:, 1], "r-" if crossing else "b-", lw=0.9)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(
            f"s={i} c={compactness(p):.2f} st={straight_fraction(p):.2f}",
            fontsize=7,
        )
    fig.suptitle(
        "Checkpoint Proto #5 - K=2 best-of-K + single-crossing CLIP "
        f"(closure={cfg['closure']})",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    out = render_grid()
    print(f"saved {out}")


if __name__ == "__main__":
    main()
