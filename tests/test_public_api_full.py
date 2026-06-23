def test_full_public_api_is_reexported():
    import track_gen

    for name in ("PerEnvSeededRNG", "TrackGenerator",
                 "TrackGenConfig", "Track"):
        assert hasattr(track_gen, name), f"track_gen.{name} is not exported"


def test_generate_tracks_warp_is_removed():
    import track_gen
    assert not hasattr(track_gen, "generate_tracks_warp"), \
        "generate_tracks_warp must not be part of the public API"


def test_checkpoint_clip_grid_import_has_no_file_side_effects():
    import importlib
    import sys
    from pathlib import Path

    module_name = "track_gen._experimental.checkpoint_clip_grid"
    sys.modules.pop(module_name, None)

    out = Path("viz/out/checkpoint_k2clip_grid.png")
    existed = out.exists()
    before_mtime = out.stat().st_mtime_ns if existed else None

    side_effect = False
    try:
        importlib.import_module(module_name)
        side_effect = (not existed and out.exists()) or (
            existed and out.stat().st_mtime_ns != before_mtime
        )
    finally:
        if not existed and out.exists():
            out.unlink()

    assert not side_effect, "importing checkpoint_clip_grid must not render or write files"
