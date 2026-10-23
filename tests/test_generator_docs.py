"""The generator docs stay in step with the code (roadmap PR7, PR8).

* The three code examples of ``docs/contributing/writing-a-generator.rst`` (the quick
  start's wobbly circle, the scatter gate generator Stage (roadmap PR11) and the star
  generator built from ``track_gen.ops``) run on CPU and produce valid tracks and gates.
* The guide's Stage quick start (``Wobble``) registers, generates, takes a dynamic
  ``set("amp", ...)``, and ``describe()`` reports the JSON schema the guide prints.
* ``docs/configuration/reference.rst`` lists every field of every built-in Params class
  in that class's table.
* Every private ``_name`` the per-generator pages cite still exists in that generator's
  module (or in ``warp_pipeline``, for the shared stages).
"""
from __future__ import annotations

import dataclasses
import importlib
import importlib.util
import json
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
_JSON_BLOCK = re.compile(r"\.\. code-block:: json\n\n((?:(?:   .*)?\n)+)")
# The gate example is a Stage class (PR11) that shares the guide's imports and SALT; its
# ``register`` (which elides the quick start's registrations) is dropped and the class
# registered directly.
_GATE_HEADER = ("import warp as wp\n\nfrom track_gen import ops, plugin\n\n"
                "SALT = 0x61636D65\n\n")
_REGISTER = re.compile(r"^def register\(reg\):\n(?:(?:    .*)?\n)*", flags=re.M)


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
    gate = _guide_block("class Scatter(plugin.Stage)")
    star = _guide_block("def generate_star(")
    assert _REGISTER.search(gate), "the gate example registers its Stage"
    gate = _REGISTER.sub("", gate)
    gate = gate + "\n\ndef register_gates(reg):\n    reg.add(Scatter)\n"
    path = tmp_path_factory.mktemp("guide") / "acme_guide_examples.py"
    path.write_text(quick + "\n\n" + star + "\n\n" + _GATE_HEADER + gate + "\n")
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


def _import_block(tmp_path_factory, source: str, name: str):
    path = tmp_path_factory.mktemp(name) / f"{name}.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def stage_guide_module(tmp_path_factory):
    """The guide's Stage quick start as its own module (it defines ``register`` too)."""
    name = "acme_guide_stage"
    try:
        yield _import_block(tmp_path_factory, _guide_block("class Wobble(plugin.Stage)"),
                            name)
    finally:
        sys.modules.pop(name, None)


def test_guide_stage_example_runs_and_matches_its_schema(stage_guide_module):
    g, E = stage_guide_module, 8
    with registry.scoped():
        g.register(registry)
        cfg = TrackGenConfig(generator="acme.wobble", num_envs=E, device="cpu",
                             generator_params={"lobes": 5})
        gen = TrackGenerator(cfg, PerEnvSeededRNG(seeds=0, num_envs=E, device="cpu"))
        track = gen.generate()
        assert track.valid.numpy().any()
        before = np.nan_to_num(track.center.numpy()).copy()
        gen.params["generator"].set("amp", 0.35)          # the guide's usage snippet
        after = gen.generate()                            # same seeds, the new amp
        assert not np.array_equal(before, np.nan_to_num(after.center.numpy()))
        assert gen.generate(advance=True).valid.numpy().any()
        schema = registry.describe("generator", "acme.wobble").params_schema
    printed = [json.loads(textwrap.dedent(b)) for b in _JSON_BLOCK.findall(_GUIDE.read_text())]
    assert [b for b in printed if b.get("title") == "WobbleParams"] == [schema]


def test_reference_lists_every_builtin_params_field():
    from track_gen._src import builtin_params as bp
    from track_gen.stages.external import ExternalParams

    # The built-in generators' Params: the flat ones, and (roadmap PR13b) the ingestion
    # generator's (a Stage: generator="external").
    classes = {name: getattr(bp, name) for name in bp.__all__}
    classes["ExternalParams"] = ExternalParams
    text = (_DOCS / "configuration" / "reference.rst").read_text()
    heads = list(re.finditer(r"^``(\w+Params)`` \(", text, flags=re.M))
    sections = {}
    for i, m in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else text.index("\n.. _", m.end())
        sections[m.group(1)] = text[m.start():end]
    assert set(sections) == set(classes)
    missing = [f"{name}.{f.name}" for name, cls in classes.items()
               for f in dataclasses.fields(cls)
               if f"``{f.name}``" not in sections[name]]
    assert not missing, missing


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
