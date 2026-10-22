"""Built-in generator outputs are byte-identical to 1077b18 (roadmap PR7).

The fingerprints in ``tests/goldens/pr7_fingerprints.json`` are sha256 digests of every
output field of every built-in track and gate generator, captured on CPU at 1077b18
(before the built-ins were ported to ``track_gen.ops``) with
``tests/tools/capture_pr7_fingerprints.py``. Float results depend on the platform and
the Warp version, so the test runs only where the file holds this platform's key and
skips elsewhere (capture one with the tool against a 1077b18 checkout).
"""
from __future__ import annotations

import pytest

from tests.tools.capture_pr7_fingerprints import compute, load, platform_key


def test_builtin_generators_match_the_1077b18_fingerprints():
    key = platform_key()
    want = load().get(key)
    if want is None:
        pytest.skip(f"no 1077b18 fingerprints for {key}")
    got = compute()
    assert sorted(got) == sorted(want)
    differ = sorted(k for k in want if got[k] != want[k])
    assert not differ, f"{len(differ)} fields differ from 1077b18: {differ[:20]}"
