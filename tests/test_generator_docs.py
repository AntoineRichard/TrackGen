"""The generator docs stay in step with the code (roadmap PR7).

* The three code examples of ``docs/contributing/writing-a-generator.rst`` (the quick
  start's wobbly circle, the scatter gate generator and the star generator built from
  ``track_gen.ops``) run on CPU and produce valid tracks and gates.
* Every private ``_name`` the per-generator pages cite still exists in that generator's
  module (or in ``warp_pipeline``, for the shared stages).
"""
from __future__ import annotations

import importlib
import importlib.util
import pathlib
import re
import sys
import textwrap

import numpy as np
import pytest

from track_gen import (GateGenConfig, GateGenerator, PerEnvSeededRNG, TrackGenConfig,
                       TrackGenerator)
from track_gen.plugin import GeneratorSpec, registry

_DOCS = pathlib.Path(__file__).resolve().parent.parent / "docs"
_GUIDE = _DOCS / "contributing" / "writing-a-generator.rst"
_BLOCK = re.compile(r"\.\. code-block:: python\n\n((?:(?:   .*)?\n)+)")
_GATE_REGISTER = ('def register(reg):\n'
                  '    ...  # reg.add(GeneratorSpec(name="acme.wobbly", ...)) as above\n')


def _guide_block(marker: str) -> str:
    blocks = [textwrap.dedent(b) for b in _BLOCK.findall(_GUIDE.read_text())]
    hits = [b for b in blocks if marker in b]
    assert len(hits) == 1, f"expected one python block containing {marker!r}"
    return hits[0]


@pytest.fixture(scope="module")
def guide_module(tmp_path_factory):
    """The guide's examples as one importable module (Warp kernels need a source file,
    so the blocks are written out and imported rather than exec'd)."""
    quick = _guide_block("def generate(seeds_wp, config, out_centerline")
    gate = _guide_block("def generate_scatter_gates(")
    star = _guide_block("def generate_star(")
    assert _GATE_REGISTER in gate
    gate = gate.replace(_GATE_REGISTER, "def register_gates(reg):\n")
    path = tmp_path_factory.mktemp("guide") / "acme_guide_examples.py"
    path.write_text(quick + "\n\n" + gate + "\n\n" + star + "\n")
    spec = importlib.util.spec_from_file_location("acme_guide_examples", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
        yield mod
    finally:
        sys.modules.pop(spec.name, None)


def test_guide_examples_generate_valid_tracks_and_gates(guide_module):
    g, E = guide_module, 8
    with registry.scoped():
        g.register(registry)
        g.register_gates(registry)
        registry.add(GeneratorSpec(name="acme.star", alloc_scratch=g.alloc_star_scratch,
                                   generate=g.generate_star, accepts_ctx=True))
        for name in ("acme.wobbly", "acme.star"):
            cfg = TrackGenConfig(generator=name, num_envs=E, device="cpu")
            rng = PerEnvSeededRNG(seeds=0, num_envs=E, device="cpu")
            track = TrackGenerator(cfg, rng).generate()
            valid, count = track.valid.numpy(), track.count.numpy()
            assert valid.any(), name
            center = track.center.numpy().reshape(E, -1, track.center.numpy().shape[-1])
            for e in np.flatnonzero(valid):
                assert np.isfinite(center[e, :count[e]]).all(), (name, e)
        gcfg = GateGenConfig(generator="acme.scatter", num_envs=E, device="cpu")
        rng = PerEnvSeededRNG(seeds=0, num_envs=E, device="cpu")
        gates = GateGenerator(gcfg, rng).generate()
        count = gates.count.numpy()
        assert ((count >= g.MIN_GATES) & (count <= g.MAX_GATES)).all()
        assert gates.valid.numpy().any()


_GENERATOR_MODULES = {
    "bezier": "warp_generate", "hull": "warp_generate_hull", "polar": "warp_generate_polar",
    "voronoi": "warp_generate_voronoi", "checkpoint": "warp_generate_checkpoint",
    "repulsive": "warp_generate_repulsive",
}


@pytest.mark.parametrize("page", sorted(_GENERATOR_MODULES))
def test_generator_pages_cite_existing_private_names(page):
    mod = importlib.import_module(f"track_gen._src.{_GENERATOR_MODULES[page]}")
    pipe = importlib.import_module("track_gen._src.warp_pipeline")
    stale = []
    for i, line in enumerate((_DOCS / "generators" / f"{page}.rst").read_text().splitlines(), 1):
        for name in re.findall(r"``(_[A-Za-z0-9_]+)``", line):
            if not hasattr(mod, name) and name not in vars(pipe):
                stale.append(f"{page}.rst:{i}: {name}")
    assert not stale, stale
