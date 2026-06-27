import track_gen


def test_public_api_surface_is_exactly_curated():
    assert set(track_gen.__all__) == {
        "TrackGenerator",
        "TrackGenConfig",
        "Track",
        "GateGenerator",
        "GateGenConfig",
        "GateSequence",
        "PerEnvSeededRNG",
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
        "from track_gen._src import gate_generator_registry",
        "gate_generator_registry.available()",
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
