#!/usr/bin/env python3
"""Drop Warp's log lines from a benchmark's raw output, so a docs page can show the rest.

Reads stdin, writes stdout. Warp prints its init banner (``Warp 1.18.0 initialized:`` and
the indented lines under it) and one ``Module <name> <hash> load on device ...`` line per
module to **stdout**, so ``2>/dev/null`` does not remove them, and their ``device`` would
match a ``literalinclude`` ``:start-at: device``. Leading blank lines are dropped too::

    python -m benchmarks.benchmark_collision --E 8192 2>&1 \\
        | python docs/_tools/strip_warp_log.py > docs/_data/collision-cuda-default.txt
"""
from __future__ import annotations

import re
import sys

_BANNER = re.compile(r"^Warp \d+\.\d+(\.\d+)?\S* initialized:\s*$")
_MODULE = re.compile(r"^Module \S+ \S+ load on device ")


def strip(text: str) -> str:
    out = []
    in_banner = False
    for line in text.splitlines():
        if _BANNER.match(line):
            in_banner = True
            continue
        if in_banner:
            if line.startswith((" ", "\t")):
                continue
            in_banner = False
        if _MODULE.match(line):
            continue
        out.append(line)
    while out and not out[0].strip():
        out.pop(0)
    return "\n".join(out) + ("\n" if out else "")


if __name__ == "__main__":
    sys.stdout.write(strip(sys.stdin.read()))
