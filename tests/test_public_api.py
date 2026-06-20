import track_gen


def test_public_api_surface_is_exactly_curated():
    assert set(track_gen.__all__) == {
        "TrackGenerator",
        "TrackGenConfig",
        "Track",
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


def test_import_track_gen_pulls_no_torch():
    # The shipped runtime is pure NVIDIA Warp + numpy; torch is a dev-only dependency
    # (tests + the torch oracle). `import track_gen` must NOT pull torch into sys.modules.
    # Fresh interpreter, because other tests in this session import torch.
    import subprocess
    import sys

    code = (
        "import sys, track_gen\n"
        "leaked = sorted(m for m in sys.modules if m == 'torch' or m.startswith('torch.'))\n"
        "assert not leaked, leaked\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
