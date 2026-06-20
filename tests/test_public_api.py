import track_gen


def test_public_api_surface_is_exactly_curated():
    assert set(track_gen.__all__) == {
        "TrackGenerator",
        "generate_tracks_warp",
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
