def test_full_public_api_is_reexported():
    import track_gen

    for name in ("PerEnvSeededRNG", "TrackGenerator", "TrackGenConfig", "Track", "Centerline"):
        assert hasattr(track_gen, name), f"track_gen.{name} is not exported"
