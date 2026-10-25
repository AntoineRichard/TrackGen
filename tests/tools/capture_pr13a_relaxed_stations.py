"""Digests of the PR11 ``relaxed_stations`` gate pipelines at ed075f6 (roadmap PR13a).

PR13a moves the width stage before relax: a ``presets.gates(source="relaxed_stations")``
spec now holds ``width.constant`` (which launches nothing) before ``relax.xpbd``, and
``relax.xpbd`` reads its half-width from ``cl.half_width``. With the default
``spacing_metric="xy"`` / ``elevation_on="gates"`` the outputs must stay byte-identical to
PR11. This tool records, per case of :data:`CASES` (a ``presets.gates`` keyword set, the
same in both releases), the sha256 of every ``GateSequence`` field and of the anchors /
relaxed centerline taps; ``tests/goldens/pr13a_pr11_relaxed_stations.json`` keeps them,
keyed by the PR7 platform key (float results depend on the platform and the Warp
version). ``tests/test_stations_xyz.py`` rebuilds the cases with the current code and
compares.

Capture (from the repository root, against a checkout of ed075f6)::

    PYTHONPATH=<ed075f6 tree> python tests/tools/capture_pr13a_relaxed_stations.py

It adds or replaces this platform's entry.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import warp as wp

GOLDEN = Path(__file__).resolve().parents[1] / "goldens" / "pr13a_pr11_relaxed_stations.json"

E = 4
SEEDS = (0, 1, 7, 99991)
FIELDS = ("position", "tangent", "forward", "orientation", "half_size", "left", "right",
          "valid", "count", "fail_mask")
TAPS = ("anchors.xy", "anchors.count", "gates.xy", "gates.z")
_NOISE = {"z_base": 0.1, "z_max": 1.5, "z_noise_amplitude": 0.5, "z_noise_harmonics": 4}
#: case -> presets.gates keyword arguments (num_envs / device are the tool's).
CASES = {
    "bezier/noise": dict(generator="bezier", source="relaxed_stations", gates=12,
                         elevation=("noise", _NOISE)),
    "bezier/flat": dict(generator="bezier", source="relaxed_stations", gates=12),
    "hull/uniform/hw0.08": dict(generator="hull", source="relaxed_stations", gates=10,
                                half_width=0.08, elevation=("uniform", {"z_max": 1.0})),
    "bezier/noise/poison": dict(generator="bezier", source="relaxed_stations", gates=12,
                                elevation=("noise", _NOISE), on_invalid="poison"),
}


def platform_key() -> str:
    return f"{sys.platform}-{platform.machine()}-warp{wp.__version__}"


def _sha(arr: np.ndarray) -> str:
    a = np.ascontiguousarray(arr)
    h = hashlib.sha256(f"{a.dtype.str}{a.shape}".encode())
    h.update(a.tobytes())
    return h.hexdigest()


def capture(kw: dict) -> dict:
    """``name -> sha256`` of one ``generate()`` of the case (CPU)."""
    from track_gen import GateGenerator, PerEnvSeededRNG
    from track_gen.pipeline import presets

    spec = presets.gates(num_envs=E, device="cpu", **kw)
    rng = PerEnvSeededRNG(seeds=wp.array(np.asarray(SEEDS, np.int32), dtype=wp.int32,
                                         device="cpu"), num_envs=E, device="cpu")
    gen = GateGenerator(spec, rng)
    seq = gen.generate()
    wp.synchronize()
    out = {f: _sha(getattr(seq, f).numpy()) for f in FIELDS}
    for ch in TAPS:
        out[ch] = _sha(gen.program.tap(ch).numpy())
    out["cl.xy@relax"] = _sha(gen.program.tap("cl.xy", after="relax").numpy())
    return out


def load() -> dict:
    return json.loads(GOLDEN.read_text()) if GOLDEN.exists() else {}


def main() -> None:
    wp.config.quiet = True
    data = load()
    data[platform_key()] = {name: capture(kw) for name, kw in CASES.items()}
    GOLDEN.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    print(f"wrote {GOLDEN} [{platform_key()}]: {sorted(CASES)}")


if __name__ == "__main__":
    main()
