"""Built-in generator outputs are byte-identical to 1077b18 (roadmap PR7).

The fingerprints in ``tests/goldens/pr7_fingerprints.json`` are sha256 digests of every
output field of every built-in track and gate generator, captured on CPU at 1077b18
(before the built-ins were ported to ``track_gen.ops``) with
``tests/tools/capture_pr7_fingerprints.py``. Float results depend on the platform and
the Warp version, so the test runs only where the file holds this platform's key and
skips elsewhere (capture one with the tool against a 1077b18 checkout).
"""
from __future__ import annotations

import os

import pytest

from tests.tools.capture_pr7_fingerprints import compute, load, platform_key

#: Fields whose bytes depend on the HOST, not only on the platform key: the checkpoint
#: track generator's path tracer is chaotic, so last-ulp differences of the C library's
#: transcendental functions or of the CPU's code generation between two x86_64 machines
#: grow into visible differences. ``TRACKGEN_SKIP_HOST_SENSITIVE_GOLDENS=1`` (set by CI,
#: whose runners are not the capture machines) leaves them out; every other field is
#: still compared there.
HOST_SENSITIVE = ("track/checkpoint/",)


def _host_sensitive_skipped() -> bool:
    return os.environ.get("TRACKGEN_SKIP_HOST_SENSITIVE_GOLDENS", "") not in ("", "0")


def test_builtin_generators_match_the_1077b18_fingerprints():
    key = platform_key()
    want = load().get(key)
    if want is None:
        pytest.skip(f"no 1077b18 fingerprints for {key}")
    got = compute()
    assert sorted(got) == sorted(want)
    if _host_sensitive_skipped():
        want = {k: v for k, v in want.items() if not k.startswith(HOST_SENSITIVE)}
    differ = sorted(k for k in want if got[k] != want[k])
    assert not differ, f"{len(differ)} fields differ from 1077b18: {differ[:20]}"
