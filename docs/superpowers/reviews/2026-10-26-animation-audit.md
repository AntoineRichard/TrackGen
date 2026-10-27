# Animation audit: jitter, motion and readability

Status: final captures, selective registration, independent review, strict docs and
browser acceptance and the complete CPU lane have passed. The results below
cover this documentation follow-up; the rc3 release tag remains immutable. This internal review is excluded
from the published site by `docs/conf.py`'s `superpowers/**` rule.

## Scope and findings

All nine published movies were inspected in both real light and dark variants
(18 files), together with their posters and the corresponding renderer/data paths.
The review distinguishes explanatory reveals, deliberately illustrated inputs and
actual generator/solver motion. Every movie and video poster uses native 16:9;
theme changes must preserve geometry and data.

| Movie | Inspection and disposition |
| --- | --- |
| `generator-bezier-construction.mp4` | Retain the approved ten-point, 8×8, cell-bounded construction illustration. Its curve/road stages are real; ordinary sampler behavior and the separate real default-seed-97 fallback remain unchanged. Both posters remained byte-identical through the Hull repair. |
| `generator-hull-construction.mp4` | Finding: default seed 7's canonical anchor offsets reach 0.47966, about 4.8 cells at pitch 0.1, obscuring the point-selection lesson. Replace the primary teaching input with an explicitly labelled ten-anchor, 8×8, cell-bounded illustration. Keep real radial midpoint displacement, Catmull–Rom, raw resampling and the road pipeline. Show the unchanged real default-seed-23 whole-loop fallback separately. |
| `generator-polar-construction.mp4` | Recorded radial and angular choices; no artificial or excessive teaching jitter found. |
| `generator-voronoi-construction.mp4` | Recorded site/control selection and curve construction; no artificial jitter found. |
| `generator-checkpoint-construction.mp4` | Recorded steering candidates and selection; no artificial jitter found. |
| `generator-repulsive-growth.mp4` | Root inspected all 64 decoded frames per theme. Motion reflects actual solver states; no artificial jitter introduced by the movie. |
| `pipeline-sampling.mp4` | Root inspected 12 keyframes per theme. The raw, target-spacing, XPBD and final redistribution steps show their recorded geometries; no artificial jitter found. |
| `relaxation-iterations.mp4` | All frames of both variants inspected. Motion is actual solver-prefix output, displayed after the documented uniformisation/inflation. Finding: a distance box obscured the bottom of the seed-21 road; move that metric into the panel title. |
| `relaxation-gates-iterations.mp4` | All frames of both variants inspected. Motion is actual solver-prefix output; no corrective change needed. |

The Polar/Voronoi/Checkpoint audit inspected 84 decoded frames and six posters.
The XPBD audit covered all frames of four files (track and gates, two themes).
These visual inspections supplement array/replay tests; they do not establish
universal stochastic yield, smoothness or runtime performance.

## Hull correction and preservation boundary

Hull shares Bézier's grid sampler: its independent absolute offsets are drawn in
`[-0.5, 0.5)` before scale and size, rather than being bounded to a grid cell.
It retains every active anchor in centroid-angle order; it does not compute a
geometric convex hull. It then inserts one radially displaced midpoint per edge.
That midpoint layer is a separate operation and remains untouched.

Only the construction renderer chooses the illustration: seed 7, ten anchors,
`min_point_distance=0.0625`, eight addresses per side and pitch 0.125. Hull's own
canonical cell IDs, RNG salt and draws are retained. Maintenance-only input mapping
uses `origin + (original_noise + 0.5) * pitch` before the ordinary angle-sort launch.
All midpoint, spline, crossing/fallback and road stages then run in the actual CPU
Program. No library source, sampler, RNG stream, default or public parameter changes.

The figure/video and page disclose the illustration. The manifest records
`illustrative_inputs=true`, `production_sampler_unchanged=true` and
`observation_preserves_outputs=false`; it does not present the mapped output as
an ordinary production sample. The observer is confined to its local enqueue.
Two illustrated replays agree, then ordinary generation restores all public arrays
and raw output byte for byte. Direct actual-sample capture/drawing stays available.
The illustrative result has 256 raw samples and a valid 57-bead finished road.

Scoped RED→GREEN checks cover bounded anchors, Hull's actual salt/IDs/noise,
unchanged midpoint-kernel replay, full dense Catmull–Rom/raw replay, real fallback,
ordinary output restoration, both themes and the unchanged Bézier example.
The 26 storyboard/dispatcher tests passed locally. The final combined validation
is recorded below.

Hull source repair: `3857656`. XPBD track-label repair: `c3ed4b8`.
Both selective final captures use clean source
`c3ed4b8810d936721413c6843cdfc1c9cd2bdd13`. Unchanged assets keep their previous
per-file provenance; do not relabel them with the new capture revision.

## Reproduce the selective captures

Run from the repository root with the rendering dependencies installed and a clean
committed source tree. For exact reproduction, use the capture source above.
The following maintenance API writes only the Hull MP4, poster and a selectively
updated copy of its generator manifest under an external output directory:

```python
import json
from pathlib import Path
from viz import render_generator_guide as guide
from viz.racing_theme import get_palette, render_variants

out = Path("build/animation-audit/hull")

def capture(target):
    theme = get_palette().name
    original = (Path("docs/assets/racing/generators-manifest.json")
                if theme == "light" else
                Path("docs/_static/media-dark/generators-manifest.json"))
    manifest = json.loads(original.read_text())
    source = guide._source_record()
    assert not source["worktree_dirty"] and not source["source_tree_dirty"]
    old = manifest["animations"]["hull"]["sample"]
    sample = guide.sample_batch("hull", seeds=[old["seed"]],
        overrides=guide._current_overrides(old["config"]))[0]
    record = guide.render_construction(target, sample, movies=True)
    record["source"] = guide._family_source(source, "hull")
    record["command"] = "Selective Hull Python API capture; theme=" + theme
    manifest["animations"]["hull"] = record
    paths = [target / ("generator-hull-construction" + suffix)
             for suffix in (".png", ".mp4")]
    guide._register_refreshed_assets(manifest, paths,
                                    record["source"], record["command"])
    (target / "generators-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return record

render_variants(out, "both", capture)
```

`render_construction` reaches `storyboard_bezier_hull.render`, whose selection
boundary supplies the labelled Hull illustration. It leaves default galleries,
Bézier and every unrelated asset untouched. Final manifests contain actual source,
per-file hashes and paired plotted geometry; check them before promoting files.

Reproduce the repaired track movie and both posters with the original budget,
seed pool and selection:

```sh
python -m viz.render_relaxation_video --device cpu --batch 24 --seed 0 \
  --sweeps 50 --rounds 16 --smooth-passes 5 --polish 10 \
  --out-dir build/animation-audit/track --theme both
```

The gate movie needs no refresh. Its equivalent reproduction command is:

```sh
python -m viz.render_relaxation_video --gates --device cpu --batch 24 --seed 0 \
  --sweeps 50 --rounds 16 --smooth-passes 5 --polish 10 \
  --out-dir build/animation-audit/gates --theme both
```

## Final integration evidence

- Independent source/provenance review: no actionable findings. Runtime source,
  package metadata and goldens remain identical to immutable `v0.3.0rc3`.
- The four final movies decode completely: Hull has 144 frames at 6 fps (24 s,
  1696×954); track has 90 frames at 8 fps (11.25 s, 1216×684). Their posters match
  their native dimensions. All are 16:9.
- Independent final visual review inspected 78 keyframes spanning every Hull
  transition and track phase in both themes, plus all four native posters. Labels
  are clear of geometry; final Hull assets match the approved previews byte for byte.
- Track's complete 66-state capture digest, all frame metrics, configuration and
  selected environments/seeds match the preceding capture exactly. The digest is
  `83560707c85bfd765c08780c823377cf3d3c95b338c8bbd713cef6e7302cf033`.
- Only four registry entries changed (two assets for each corrected movie). All
  78 other entries, both-theme files and per-file provenance remain byte-identical.
  All 164 registered file hashes verify. Generator-level historical capture records
  are retained; only the refreshed files receive the clean `c3ed4b8` provenance.
- Spark: 62 rendering, storyboard, provenance and theme tests passed in 6.28 s
  against a clean archive of `c3ed4b8`.
- Strict Sphinx `-W --keep-going -E` HTML build passed with the default 82-pair
  registry enforced; internal records stay excluded from the site.
- Browser acceptance passed 16 route/theme/viewport cases (Hull, visual tour and
  track convergence), 32 served-file hash checks, native 16:9 playback and theme
  switching with no console/network errors or horizontal page overflow. Final
  screenshots wait for an actually presented video frame, avoiding native loading
  overlays. The benchmark table remains available.
- Complete CPU lane, including executable documentation snippets: 3,763 passed,
  four optional UI skips and 837 deselected in 552.48 s; exit 0.
