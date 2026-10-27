import track_gen


def test_public_api_surface_is_exactly_curated():
    assert set(track_gen.__all__) == {
        "TrackGenerator",
        "TrackGenConfig",
        "Track",
        "GateGenerator",
        "GateGenConfig",
        "GateSequence",
        # Roadmap PR13b: the result layout, the capability protocols and their module.
        "Layout",
        "HasBand",
        "HasCenterline",
        "HasGateFrames",
        "HasGates",
        "HasGeneration",
        "results",
        # Roadmap PR14: the component protocol and adapters, the validity and checkpoint
        # protocols.
        "tools",
        "HasValidity",
        "HasCheckpoints",
        "PerEnvSeededRNG",
        "collision",
        "props",
        "checkpoints",
        "progress",
        "localize",
        "course",
        "heightfield",
        "__version__",
    }


def test_public_names_are_accessible():
    for name in track_gen.__all__:
        assert hasattr(track_gen, name), f"track_gen.{name} missing"


def test_oracle_internals_are_not_public():
    # The torch oracle lives in tests/_oracle, not the shipped package; none of these
    # internal names should be importable from track_gen.
    for gone in ("geometry", "relaxation", "inflation", "generators", "relax",
                 "safe_normalize", "polygon_area", "Centerline", "warp_pipeline",
                 "generate_tracks_warp"):
        assert not hasattr(track_gen, gone), f"track_gen.{gone} should not be public"


def test_import_track_gen_and_gate_registry_pull_no_torch_or_scipy():
    # The shipped runtime is pure NVIDIA Warp + numpy; torch is a dev-only dependency
    # (tests + the torch oracle). `import track_gen` and the lazy gate registry must NOT
    # pull torch or scipy into sys.modules.
    # Fresh interpreter, because other tests in this session import torch.
    import subprocess
    import sys

    code = "\n".join([
        "import sys, track_gen",
        "from track_gen.plugin import registry",
        "registry.available('gate_generator')",
        "expected = {",
        "    'track_gen._src.warp_generate_gates',",
        "    'track_gen._src.warp_generate_polar_gates',",
        "    'track_gen._src.warp_generate_voronoi_gates',",
        "    'track_gen._src.warp_generate_checkpoint_gates',",
        "}",
        "missing = sorted(name for name in expected if name not in sys.modules)",
        "assert not missing, missing",
        "leaked = sorted(",
        "    m for m in sys.modules",
        "    if m == 'torch' or m.startswith('torch.')",
        "    or m == 'scipy' or m.startswith('scipy.')",
        ")",
        "assert not leaked, leaked",
    ])
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_import_track_gen_loads_no_plugin_and_no_generator_module():
    # Plugin discovery is lazy: `import track_gen` (and `import track_gen.plugin`) must not
    # query the "track_gen.plugins" entry points, load the registry, or import a generator
    # module. The first registry lookup does all three, still without torch or scipy.
    # Fresh interpreter: other tests in this session already loaded the registry.
    import os
    import subprocess
    import sys

    code = "\n".join([
        "import importlib.metadata as md, sys",
        "queried = []",
        "real = md.entry_points",
        "def spy(**params):",
        "    queried.append(params.get('group'))",
        "    return real(**params)",
        "md.entry_points = spy",
        "def generator_modules():",
        "    return sorted(m for m in sys.modules if m.startswith('track_gen._src.warp_generate')",
        "                  or m == 'track_gen._src.builtin_plugins')",
        "import track_gen",
        "assert 'track_gen.plugins' not in queried, queried",
        "assert 'track_gen.plugin' not in sys.modules",
        "assert not generator_modules(), generator_modules()",
        "import track_gen.plugin",
        "assert 'track_gen.plugins' not in queried, queried",
        "assert not track_gen.plugin.registry._loaded",
        "assert not generator_modules(), generator_modules()",
        "names = track_gen.plugin.registry.available('generator')",
        "assert queried.count('track_gen.plugins') == 1, queried",
        "assert 'bezier' in names and 'track_gen._src.builtin_plugins' in sys.modules",
        "leaked = sorted(m for m in sys.modules if m.split('.')[0] in ('torch', 'scipy'))",
        "assert not leaked, leaked",
    ])
    # Discovery must run in the child even though tests/conftest.py defaults
    # TRACK_GEN_PLUGINS to "off" for this process.
    env = {k: v for k, v in os.environ.items() if k != "TRACK_GEN_PLUGINS"}
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       env=env)
    assert r.returncode == 0, r.stderr


def test_the_channel_carry_surface_is_public():
    """Roadmap PR13a: the carried-channel ops, the attribute description, the builder's
    dtype query and the new result fields."""
    import dataclasses
    import inspect

    from track_gen import ops, pipeline
    from track_gen.plugin import StageBuilder

    for name in ("ResampleTables", "resample_channels"):
        assert name in ops.__all__ and inspect.getdoc(getattr(ops, name)), name
    assert "AttributeInfo" in pipeline.__all__
    assert [f.name for f in dataclasses.fields(pipeline.AttributeInfo)] == [
        "key", "channel", "dtype", "interp", "padding", "domain"]
    assert inspect.getdoc(StageBuilder.dtype_of)
    params = inspect.signature(StageBuilder.write).parameters
    assert params["interp"].default is None and params["export"].default is False
    track = [f.name for f in dataclasses.fields(track_gen.Track)]
    assert {"half_width", "attributes", "layout"} <= set(track)
    assert {"attributes", "layout"} <= {
        f.name for f in dataclasses.fields(track_gen.GateSequence)}
    assert "tables" in inspect.signature(ops.resample_uniform).parameters
    assert "tables" in inspect.signature(ops.resample_constant_spacing).parameters
