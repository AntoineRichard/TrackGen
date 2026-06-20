def test_full_public_api_is_reexported():
    import track_gen

    for name in ("PerEnvSeededRNG", "TrackGenerator",
                 "TrackGenConfig", "Track"):
        assert hasattr(track_gen, name), f"track_gen.{name} is not exported"


def test_generate_tracks_warp_is_removed():
    import track_gen
    assert not hasattr(track_gen, "generate_tracks_warp"), \
        "generate_tracks_warp must not be part of the public API"
