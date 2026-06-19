import track_gen


def test_public_api_surface_is_exactly_curated():
    assert set(track_gen.__all__) == {
        "TrackGenerator",
        "generate_tracks_warp",
        "generate_tracks_warp_graph",
        "TrackGenConfig",
        "Track",
        "PerEnvSeededRNG",
        "__version__",
    }


def test_public_names_are_accessible():
    for name in track_gen.__all__:
        assert hasattr(track_gen, name), f"track_gen.{name} missing"


def test_oracle_internals_are_not_public():
    # The torch oracle moved to tests/_oracle and is no longer part of the package.
    for gone in ("geometry", "relaxation", "inflation", "generators", "relax",
                 "safe_normalize", "polygon_area", "Centerline", "warp_pipeline"):
        assert not hasattr(track_gen, gone), f"track_gen.{gone} should not be public"


def test_import_track_gen_is_warp_free():
    # `import track_gen` must not pull in NVIDIA Warp; Warp loads only when a
    # Warp entry point is actually used. Fresh interpreter: other tests in this
    # session import warp, so sys.modules here would already contain it.
    import subprocess
    import sys

    code = (
        "import sys, track_gen\n"
        "leaked = sorted(m for m in sys.modules if m == 'warp' or m.startswith('warp.'))\n"
        "assert not leaked, leaked\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
