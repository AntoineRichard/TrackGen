"""Built-in fingerprints after the 0.3 hashed-RNG rebaseline.

The historical PR7 filename retains coverage of every generator and output field.
Fragments are captured by tests.tools.capture_03_goldens on each platform; host-sensitive
checkpoint fields retain their explicit CI exclusion.
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


def test_builtin_generators_match_the_03_fingerprints():
    key = platform_key()
    want = load().get(key)
    if want is None:
        pytest.skip(f"no 0.3 fingerprints for {key}")
    got = compute()
    assert sorted(got) == sorted(want)
    if _host_sensitive_skipped():
        want = {k: v for k, v in want.items() if not k.startswith(HOST_SENSITIVE)}
    differ = sorted(k for k in want if got[k] != want[k])
    assert not differ, f"{len(differ)} fields differ from the 0.3 baseline: {differ[:20]}"
