# TODO

## CUDA graph-capture thread race (crashes param explorer on cuda)

`TrackGenerator.generate()` / `GateGenerator.generate()` are not thread-safe on cuda:
warmup + `wp.ScopedCapture` + `capture_launch` coordinate via the module-global
`warp_pipeline._CAPTURING` flag ("single-threaded by construction"). The param explorer
violates this — `app.load(_generate)` and `app.load(_generate_gates)` run concurrently on
every page load — producing `Warp CUDA error 700/401/900` (700 = corrupted context,
unrecoverable). Reproduced deterministically with two threads; CPU path is immune.

- [x] Serialize the cuda branch of both generators — `runtime._CAPTURE_LOCK` held
      around seed copy + warmup + capture + `capture_launch` replay in
      `TrackGenerator`, `GateGenerator`, and `Course` (capture and replay paths).
- [ ] Optionally also serialize the UI's generation events (gradio `concurrency_id`)
      so overlapping page loads queue instead of racing — belt-and-braces; the
      library lock already makes the UI safe.
- [x] Regression test: `tests/test_generate_concurrent_cuda.py` (marked `cuda`,
      `slow`) — tracks+gates and tracks+tracks concurrent capture; verified to fail
      against the unfixed code and pass with the lock.

## PyPI release

Once `track_gen` is published to PyPI, update the docs and packaging accordingly:

- [ ] Publish `track_gen` to PyPI.
- [ ] `docs/getting-started/installation.rst` — "Add to an existing uv project": lead
      with `uv add track_gen` (PyPI) and demote the `git+https://…` form to an
      "install from source" alternative.
- [ ] `docs/getting-started/installation.rst` — "From scratch with uv": mention that a
      plain `uv pip install track_gen` (non-editable, from PyPI) works for users who
      don't need a source checkout.
- [ ] Add extras examples for the PyPI form (`uv add "track_gen[ui]"`,
      `uv add "track_gen[dev]"`).
- [ ] Double-check the PyPI package name matches `track_gen` (underscore vs. hyphen).
