def test_full_public_api_is_reexported():
    import track_gen

    for name in ("PerEnvSeededRNG", "TrackGenerator", "generate_tracks_warp",
                 "generate_tracks_warp_graph", "TrackGenConfig", "Track"):
        assert hasattr(track_gen, name), f"track_gen.{name} is not exported"
