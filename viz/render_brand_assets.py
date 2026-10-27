"""Export the original hand-designed Circuit identity; no runtime or font inputs.

Run ``python -m viz.render_brand_assets --out-dir docs/_static`` from the root.
An optional ``--source-revision`` records the committed renderer's full Git SHA.
Only the explicitly named assets below are overwritten; other files are retained.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import re

from PIL import Image

# Literal 5x7 letterforms and stepped route from the user-approved Circuit sketch.
GLYPHS = {
    "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "C": ("01111", "10000", "10000", "10000", "10000", "10000", "01111"),
    "K": ("10001", "10010", "10100", "11000", "10100", "10010", "10001"),
    "G": ("01111", "10000", "10000", "10111", "10001", "10001", "01111"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "N": ("10001", "11001", "11001", "10101", "10011", "10011", "10001"),
}
ROUTE = ((5, 9), (14, 9), (14, 4), (26, 4), (26, 17),
         (21, 17), (21, 26), (5, 26), (5, 9))
PALETTES = {
    "light": {"ink": "#243344", "accent": "#166baf"},
    "dark": {"ink": "#e7e9ed", "accent": "#65b8ff"},
    "black": {"ink": "#000000", "accent": "#000000"},
    "white": {"ink": "#ffffff", "accent": "#ffffff"},
}
FAVICON_COLOR = "#327fba"
PNG_SIZES = (16, 24, 32, 48, 64)


def _road() -> set[tuple[int, int]]:
    road = set()
    for (ax, ay), (bx, by) in zip(ROUTE, ROUTE[1:]):
        for x in range(min(ax, bx), max(ax, bx) + 1):
            for y in range(min(ay, by), max(ay, by) + 1):
                road.update((x + dx, y + dy)
                            for dx in range(-2, 3) for dy in range(-2, 3))
    return road


ROAD = _road()
CHECKER = {(x, y) for y in (19, 20) for x in range(3, 8)}


def _pixels(points: set[tuple[int, int]], color: str, *,
            x: int = 0, y: int = 0, scale: int = 1) -> str:
    # Separate closed pixel subpaths retain the sketch's exact integer geometry.
    path = "".join(f"M{x + px * scale} {y + py * scale}"
                   f"h{scale}v{scale}h{-scale}z" for px, py in sorted(points))
    return f'<path fill="{color}" d="{path}"/>'


def _svg(wordmark: bool, palette: dict[str, str]) -> bytes:
    width = 140 if wordmark else 32
    label = "TrackGen Circuit wordmark" if wordmark else "TrackGen Circuit icon"
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="32" '
        f'viewBox="0 0 {width} 32" role="img" aria-labelledby="title desc" '
        'shape-rendering="crispEdges">',
        f'<title id="title">{label}</title>',
        '<desc id="desc">A closed stepped pixel racetrack with a checkered start'
        + (' beside the TrackGen pixel lettering.' if wordmark else '.') + '</desc>',
        _pixels(ROAD - CHECKER, palette["ink"]),
        _pixels({(x, y) for x, y in CHECKER if (x + y) % 2}, palette["accent"]),
    ]
    if wordmark:
        for index, letter in enumerate("TRACKGEN"):
            points = {(index * 6 + col, row)
                      for row, line in enumerate(GLYPHS[letter])
                      for col, bit in enumerate(line) if bit == "1"}
            pieces.append(_pixels(points, palette["accent" if index >= 5 else "ink"],
                                  x=40, y=9, scale=2))
    return ("".join(pieces) + '</svg>\n').encode("ascii")


def render(out_dir: Path, source_revision: str | None = None) -> None:
    """Write transparent assets and deterministic, relative-path provenance."""
    outputs = {}

    def write(relative: str, data: bytes, size: list[int] | list[list[int]]) -> None:
        destination = out_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        outputs[relative] = {"size": size, "bytes": len(data), "sha256": sha256(data).hexdigest()}

    for theme, palette in PALETTES.items():
        folder = "media-dark" if theme == "dark" else "brand"
        suffix = f"-{theme}" if theme in ("black", "white") else ""
        for wordmark in (True, False):
            kind = "wordmark" if wordmark else "icon"
            write(f"{folder}/trackgen-circuit-{kind}{suffix}.svg",
                  _svg(wordmark, palette), [140 if wordmark else 32, 32])

    # Tiny exports omit the checker so every opaque pixel belongs to the same loop.
    base = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    for point in ROAD:
        base.putpixel(point, tuple(bytes.fromhex(FAVICON_COLOR[1:])) + (255,))
    icons = {size: base.resize((size, size), Image.Resampling.NEAREST) for size in PNG_SIZES}
    for size, icon in icons.items():
        data = BytesIO()
        icon.save(data, format="PNG", optimize=False)
        write(f"brand/trackgen-circuit-icon-{size}.png", data.getvalue(), [size, size])
    data = BytesIO()
    ico_sizes = (16, 32, 48)
    icons[48].save(data, format="ICO", sizes=[(n, n) for n in ico_sizes],
                   append_images=[icons[16], icons[32]])
    write("brand/trackgen-circuit.ico", data.getvalue(), [[n, n] for n in ico_sizes])

    manifest = {
        "schema_version": 1,
        "artwork": "Original hand-designed Circuit branding; not a simulation capture.",
        "source_revision": source_revision,
        "renderer": "viz/render_brand_assets.py",
        "renderer_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "logical_sizes": {"wordmark": [140, 32], "icon": [32, 32]},
        "palette": {**PALETTES, "favicon": FAVICON_COLOR},
        "outputs": outputs,
    }
    target = out_dir / "brand/circuit-manifest.json"
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="ascii")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True, help="Static asset root to mirror")
    parser.add_argument("--source-revision", help="Full 40-hex Git revision of the renderer source")
    args = parser.parse_args()
    if args.source_revision and not re.fullmatch(r"[0-9a-fA-F]{40}", args.source_revision):
        parser.error("--source-revision must be a full 40-hex Git SHA")
    render(args.out_dir, args.source_revision.lower() if args.source_revision else None)


if __name__ == "__main__":
    main()
