"""Capture platform fragments, then merge one complete RNG rebaseline.

``capture OUTPUT`` runs all five geometry/utility reference families. Run it on
each host against identical source, then ``merge FRAGMENT...`` once. The merge
refuses mismatched source and discards stale platform entries from older keying.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tests.tools import (capture_pr7_fingerprints as pr7,
                         capture_pr11_gate_stages as pr11,
                         capture_pr13a_relaxed_stations as pr13,
                         make_pr14_goldens as pr14,
                         make_pr15_course_goldens as pr15)

ROOT = Path(__file__).resolve().parents[2]
FAMILIES = ("pr7_fingerprints.json", "pr11_gate_stages.json",
            "pr13a_pr11_relaxed_stations.json", "pr14_utilities.json", "pr15_course.json")


def source_digest() -> str:
    digest = hashlib.sha256()
    paths = list((ROOT / "track_gen").rglob("*.py"))
    paths += [p for p in (ROOT / "tests").rglob("*.py") if not p.name.startswith("test_")]
    for path in sorted(paths):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def capture(output: str) -> None:
    values = (pr7.compute(), pr11.compute()[0],
              {name: pr13.capture(kw) for name, kw in pr13.CASES.items()},
              pr14.compute(), pr15.compute())
    data = dict(source_sha256=source_digest(), platform=pr7.platform_key(),
                families=dict(zip(FAMILIES, values)))
    Path(output).write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    print(f"captured {len(values)} families for {data['platform']} -> {output}")


def merge(paths: list[str]) -> None:
    fragments = [json.loads(Path(path).read_text()) for path in paths]
    expected = source_digest()
    platforms = [f["platform"] for f in fragments]
    if len(platforms) != len(set(platforms)):
        raise ValueError("duplicate platform fragments")
    for fragment in fragments:
        if fragment["source_sha256"] != expected:
            raise ValueError(f"different source for {fragment['platform']}")
        if set(fragment["families"]) != set(FAMILIES):
            raise ValueError(f"incomplete families for {fragment['platform']}")
    # Validate every fragment before changing any golden file.
    merged = {}
    for family in FAMILIES:
        data = {f["platform"]: f["families"][family] for f in fragments}
        if family == "pr15_course.json":
            data[pr15.TOPOLOGY] = pr15.topology_of(fragments[0]["families"][family])
        merged[family] = json.dumps(data, indent=1, sort_keys=True) + "\n"
    for family, text in merged.items():
        (ROOT / "tests" / "goldens" / family).write_text(text)
    print(f"rebaselined {len(FAMILIES)} families for {', '.join(platforms)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("capture").add_argument("output")
    sub.add_parser("merge").add_argument("fragments", nargs="+")
    args = parser.parse_args()
    if args.command == "capture":
        capture(args.output)
    else:
        merge(args.fragments)
