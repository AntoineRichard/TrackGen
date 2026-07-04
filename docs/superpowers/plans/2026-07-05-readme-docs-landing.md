# README Rework & Utilities Docs Section Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Promote the runtime utilities to a first-class `docs/utilities/` section (mirroring `docs/generators/`), retire the runtime-utilities tutorial into it, and replace the 392-line README with a ~140-line attractive landing that defers depth to the docs site.

**Architecture:** Pure docs/content work, no runtime changes. Task 1 builds the docs section by redistributing existing prose (tutorial + api.rst Performance rubric) into five new pages plus landing-page integration, gated by the CI-equivalent `-W` docs build. Task 2 rewrites the README from the full text in this plan and polishes the repo About metadata.

**Tech Stack:** Sphinx (furo + sphinx-design), GitHub-flavored Markdown, `gh` CLI.

**Spec:** `docs/superpowers/specs/2026-07-04-readme-docs-landing-design.md`

## Global Constraints

- No runtime code changes; no figure/asset regeneration; `tests/test_readme_assets.py` untouched and passing.
- Docs must build clean under CI semantics: `python3 -m sphinx -W --keep-going -b html docs <out>` (CI uses `-W`; a single new warning breaks the deploy).
- README image paths stay RELATIVE (`docs/assets/...`) so GitHub renders them; all docs links in the README are ABSOLUTE to `https://antoinerichard.github.io/TrackGen/` with `.html` page paths.
- The tutorial `docs/tutorials/runtime-utilities.rst` is deleted only after every piece of its content has a home in `docs/utilities/*`; grep for inbound references (`runtime-utilities`) and repoint ALL of them.
- Display name `TrackGen`; package name `track_gen` (existing convention).
- Suite runs: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/ -q` (GPU present; cuda tests run).
- Work on branch `feature/readme-docs-landing` off main (Task 1 Step 0).

---

### Task 1: `docs/utilities/` section + landing integration + tutorial retirement

**Files:**
- Create: `docs/utilities/overview.rst`, `docs/utilities/collision.rst`, `docs/utilities/props.rst`, `docs/utilities/progress.rst`, `docs/utilities/course.rst`
- Modify: `docs/index.rst` (nav card, feature card, gallery figure, toctree)
- Modify: `docs/reference/api.rst` (Performance subsection moves out; `:doc:` refs repointed)
- Modify: `docs/contributing/rendering-assets.rst` (only if it references the tutorial page — grep)
- Delete: `docs/tutorials/runtime-utilities.rst` (content redistributed)

**Interfaces:**
- Consumes: the existing prose in `docs/tutorials/runtime-utilities.rst` (section headings named below) and the "Performance" subsection of `docs/reference/api.rst`.
- Produces: page names Task 2's README links depend on — `utilities/overview.html`, `utilities/collision.html`, `utilities/props.html`, `utilities/progress.html`, `utilities/course.html`.

- [ ] **Step 0: Create the feature branch**

```bash
git checkout -b feature/readme-docs-landing
```

- [ ] **Step 1: Write `docs/utilities/overview.rst`**

Full content (the "after regeneration" table and the capture paragraph are
LIFTED from `docs/tutorials/runtime-utilities.rst` — take the current text
of its "Stable buffers and CUDA graphs" section and the regeneration
list-table verbatim where indicated):

```rst
Runtime utilities
=================

Beyond generating tracks and gates, ``track_gen`` ships a family of
GPU-batched runtime utilities for the sim loop:

- :doc:`Out-of-bounds & obstacle collision </utilities/collision>` —
  oriented boxes vs the drivable band (exact or SDF backends) and vs disc
  obstacles such as gate posts.
- :doc:`Boundary props </utilities/props>` — cone lines and wall pieces
  along the boundaries, for rendering-only instancing.
- :doc:`Checkpoints & progress </utilities/progress>` — ordered course
  goals from gate sequences or subsampled track centerlines, with pass/lap
  events and reward-ready distances.
- :doc:`The Course facade </utilities/course>` — one object bundling
  generation, collision, and progress per mode.

.. figure:: ../assets/utilities-overview.png
   :alt: Overview of the collision and props utilities on one track.

   Cones and walls placed by ``track_gen.props``, the effect of spacing,
   and ``track_gen.collision``'s SDF field with boxes classified by the
   exact backend.

Family conventions
------------------

Every utility follows the same contracts, so learning one teaches all:

- **Flat batched layouts** — arrays are flat ``[E * stride]`` Warp arrays,
  NaN-padded past each env's real count.
- **Preallocated in-place results** — per-step methods return the SAME
  result object every call, overwritten in place; call ``clone()`` for a
  snapshot.
- **Eager, NaN-proof validation** — shapes, dtypes, and devices are checked
  when you construct or bind, never in the hot path.
- **Input binding** — latch a tool onto your sim's stable pose buffers once
  (constructor kwargs or ``bind``/``bind_inputs``) and call
  ``update()``/``query()`` with no arguments thereafter.
- **Undefined for invalid envs** — gate on ``valid`` from the generator
  result, as everywhere in the library.

CUDA graphs
-----------

[LIFT: the current "Stable buffers and CUDA graphs" section body from
tutorials/runtime-utilities.rst, including the standalone-capture paragraph
that references ``track_gen.set_capturing``.]

After regenerating
------------------

[LIFT: the current "after regeneration" list-table and its intro sentence
from tutorials/runtime-utilities.rst.]
```

- [ ] **Step 2: Write `docs/utilities/collision.rst`**

Structure and new connective text; lift markers name exact existing
sections:

```rst
Out-of-bounds & obstacle collision
==================================

``track_gen.collision`` answers two questions every batched sim asks: *did
the agent leave the drivable band?* and *did it hit a point obstacle?*

Out-of-bounds checking
----------------------

[LIFT: the "Out-of-bounds collision" section body from
tutorials/runtime-utilities.rst — backend narrative and guidance.]

Performance
-----------

[LIFT: the entire "Performance" subsection currently under "Collision
queries" in docs/reference/api.rst — the measured table, accuracy summary,
rule of thumb, and Reproduce block — verbatim.]

Disc obstacles (gate posts, cones)
----------------------------------

[LIFT: the "Gate posts & point obstacles" section body from
tutorials/runtime-utilities.rst, including the recipe code block and its
snapshot caveat.]

.. figure:: ../assets/disc-collision.png
   :alt: Gate posts as disc obstacles with boxes colored by hit.

   Gate posts as discs; the same checker makes cones physical.

API: :class:`track_gen.collision.CollisionChecker`,
:class:`track_gen.collision.DiscChecker` — see the
:doc:`API reference </reference/api>`.
```

- [ ] **Step 3: Write `docs/utilities/props.rst`**

```rst
Boundary props
==============

[LIFT: the "Boundary props (rendering-only instancing)" section body from
tutorials/runtime-utilities.rst.]

Both modes snap the requested spacing per environment —
``n = clamp(round(perimeter / spacing), 3, max_props)`` at effective step
``perimeter / n`` — so every closed ring places props with no seam gap or
doubled prop. ``PropSet.count``/``truncated``/``step`` report what was
actually placed; see the left panels of the overview figure on
:doc:`the utilities overview </utilities/overview>`.

Props are rendering-only by design. To make point props physical (cones a
vehicle can clip), feed their positions to
:class:`track_gen.collision.DiscChecker` — see
:doc:`collision </utilities/collision>`.

API: :class:`track_gen.props.PropSampler`,
:class:`track_gen.props.PropSet` — see the
:doc:`API reference </reference/api>`.
```

- [ ] **Step 4: Write `docs/utilities/progress.rst`**

```rst
Checkpoints & progress
======================

Progress logic is identical for drone-racing gates and car-racing tracks:
discrete pass events plus a distance-to-next-goal signal. The utilities
split it into a shared goal contract and a stateful tracker.

Checkpoints: one contract, two sources
--------------------------------------

[LIFT: the "Checkpoints: one contract, two sources" section body from
tutorials/runtime-utilities.rst, including the code block.]

.. figure:: ../assets/checkpoints-overview.png
   :alt: Track-sourced virtual gates beside gate-sourced checkpoints.

   The same ``CheckpointSet`` contract from a subsampled track (left) and a
   gate sequence (right).

Progress tracking & rewards
---------------------------

[LIFT: the "Progress tracking & rewards" section body from
tutorials/runtime-utilities.rst, including the reward snippet with its
reset caveat and the reset/teleport paragraph.]

.. figure:: ../assets/progress-tracking.png
   :alt: Agent path colored by progress with a dist_to_next lower panel.

   A scripted agent threading track checkpoints; the lower panel shows the
   ``dist_to_next`` sawtooth your negative-delta reward differentiates.

API: :class:`track_gen.checkpoints.CheckpointSet`,
:class:`track_gen.checkpoints.CheckpointSampler`,
:class:`track_gen.progress.ProgressTracker` — see the
:doc:`API reference </reference/api>`.
```

- [ ] **Step 5: Write `docs/utilities/course.rst`**

```rst
The Course facade
=================

[LIFT: the "Putting it together: the Course facade" section body from
tutorials/runtime-utilities.rst — intro, code block, whole-batch vs
per-env paragraph, gates-mode example, and the capture paragraph.]

Under the hood, two CUDA graphs do the heavy lifting on ``cuda`` devices:
the generator's own pipeline graph (captured on the first ``generate()``)
and a facade-owned refresh graph covering the post-generation work — SDF
rebake, checkpoint resample, gate-post rebuild, and the full progress
reset. Without a ``seeds=`` argument, ``generate()`` reproduces the
identical batch (the generators are deterministic under an unchanged RNG);
pass seeds to vary the courses. When ``max_checkpoints`` is auto-derived,
check ``course.checkpoint_sampler.truncated`` after regenerating onto
much longer tracks.

API: :class:`track_gen.course.Course`,
:class:`track_gen.course.CourseConfig` — see the
:doc:`API reference </reference/api>`.
```

- [ ] **Step 6: Integrate into `docs/index.rst`**

(a) Toctree — insert a new block AFTER the generators toctree block:

```rst
.. toctree::
   :maxdepth: 1
   :caption: Runtime utilities
   :hidden:

   utilities/overview
   utilities/collision
   utilities/props
   utilities/progress
   utilities/course
```

(match the surrounding blocks' exact options — check whether they use
`:hidden:` and `:caption:` and mirror them).

(b) Nav-card grid: add a card next to the Generators card:

```rst
   .. grid-item-card:: Runtime utilities
      :link: utilities/overview
      :link-type: doc

      Collision, props, checkpoints & progress, and the Course facade.
```

(c) Feature-card grid (the six-card `grid: 1 2 2 3`): replace the
"RL-ready output" card's body so the utilities are visible, e.g.:

```rst
   .. grid-item-card:: RL-ready runtime

      Out-of-bounds collision, checkpoint progress and rewards, prop
      instancing — and one Course object that bundles them.
```

(keep six cards total; do not grow the grid).

(d) Gallery: after the existing `readme-generator-grid.png` figure, add:

```rst
.. figure:: assets/progress-tracking.png
   :alt: Progress tracking on a generated track.

   Runtime utilities in action: checkpoint progress on a generated track
   with the reward-ready ``dist_to_next`` signal.
```

(e) Remove `tutorials/runtime-utilities` from the tutorials toctree.

- [ ] **Step 7: Repoint api.rst and retire the tutorial**

- In `docs/reference/api.rst`: delete the "Performance" subsection under
  "Collision queries" (it moved to `utilities/collision.rst`) and replace
  it with:

```rst
Performance
~~~~~~~~~~~

Measured numbers, backend trade-offs, and the reproduce command live in
:doc:`the collision utility page </utilities/collision>`.
```

- Repoint every other reference: `grep -rn "runtime-utilities" docs/ README.md`
  — the api.rst `:doc:` link in the Course facade section goes to
  `/utilities/course`; anything in `docs/contributing/rendering-assets.rst`
  naming the tutorial page is reworded to name the utilities section.
- `git rm docs/tutorials/runtime-utilities.rst`.
- Final grep must show zero remaining `runtime-utilities` references
  outside `docs/superpowers/` (archived specs/plans keep their history).

- [ ] **Step 8: Build docs under CI semantics and inspect**

Run: `python3 -m sphinx -W --keep-going -b html docs /tmp/claude-1000/-home-antoine-Documents-track-gen/a3819d36-c82d-4063-bcba-b7abbecf061d/scratchpad/docs-build-ci`
Expected: exit 0, zero warnings. Then open
`.../docs-build-ci/utilities/overview.html` etc. with the Read tool on the
generated HTML? No — instead verify structurally: the five pages exist in
the output dir, `tutorials/runtime-utilities.html` does NOT, and
`index.html` contains the new card text (grep the built HTML).

- [ ] **Step 9: Run the suite (readme-assets tests must be untouched)**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/ -q`
Expected: all PASS.

- [ ] **Step 10: Commit**

```bash
git add docs/
git commit -m "docs: runtime utilities become a first-class section (generators-style)"
```

---

### Task 2: README rework + repo About polish

**Files:**
- Rewrite: `README.md` (full replacement text below)
- Command: `gh repo edit` (description)

**Interfaces:**
- Consumes: Task 1's page URLs (`/utilities/*.html`).

- [ ] **Step 1: Replace `README.md` with exactly this content**

````markdown
# TrackGen

**GPU-batched race tracks, gate courses, and the runtime signals to race
them — pure [NVIDIA Warp](https://github.com/NVIDIA/warp), CUDA-graph
native.**

[![Documentation](https://img.shields.io/badge/docs-antoinerichard.github.io%2FTrackGen-blue)](https://antoinerichard.github.io/TrackGen/)
[![docs build](https://github.com/AntoineRichard/TrackGen/actions/workflows/docs.yml/badge.svg)](https://github.com/AntoineRichard/TrackGen/actions/workflows/docs.yml)
![Python](https://img.shields.io/badge/python-%E2%89%A5%203.10-blue)
![Warp](https://img.shields.io/badge/warp--lang-%E2%89%A5%201.14-76b900)

Given a batch of per-environment seeds, TrackGen generates `E` closed-loop
tracks (or gate sequences) in parallel — and then keeps working at sim
time: out-of-bounds collision, checkpoint progress and rewards, prop
instancing, all as Warp kernels over the same batched buffers.

```
seeds[E] ─► generator ─► resample ─► XPBD relax ─► inflate ─► Track ─► collision · progress · props
            bezier/checkpoint/hull/polar/voronoi                       (runtime utilities)
```

![Representative phase-1 outputs by generator](docs/assets/readme-generator-strip.png)

## What's in the box

| | | docs |
|---|---|---|
| **Five track generators** | Bezier, checkpoint-steering, hull, polar, Voronoi — one config, per-env styles | [Generators](https://antoinerichard.github.io/TrackGen/generators/overview.html) |
| **Gate sequences** | Batched gate courses with tangent frames and collision-solved spacing | [Tutorial](https://antoinerichard.github.io/TrackGen/tutorials/gate-sequences.html) |
| **Out-of-bounds collision** | Oriented boxes vs the drivable band — exact scan or baked SDF — plus disc obstacles (gate posts, cones) | [Collision](https://antoinerichard.github.io/TrackGen/utilities/collision.html) |
| **Boundary props** | Cone lines and wall pieces along any boundary, seam-free, for instanced rendering | [Props](https://antoinerichard.github.io/TrackGen/utilities/props.html) |
| **Checkpoints & progress** | Ordered goals from gates *or* subsampled centerlines; pass/lap events and `dist_to_next` rewards | [Progress](https://antoinerichard.github.io/TrackGen/utilities/progress.html) |
| **The Course facade** | One object: `generate()` → `bind()` → `step()`/`reset(mask)`, CUDA-graph replays included | [Course](https://antoinerichard.github.io/TrackGen/utilities/course.html) |

![Runtime utilities on one generated track](docs/assets/utilities-overview.png)

## Install

Python ≥ 3.10; `numpy` and `warp-lang` are the only runtime deps. Runs on
the Warp `cpu` device (tests/CI) and `cuda` (production).

```bash
uv venv --python 3.12 && uv pip install -e ".[dev]"     # or:
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
```

Extras: `dev` (tests, torch oracles, matplotlib) · `ui` (Gradio parameter
explorer) · `docs` (Sphinx). Details:
[installation](https://antoinerichard.github.io/TrackGen/getting-started/installation.html).

## Quickstart

Generate a batch of tracks:

```python
import warp as wp
wp.init()
from track_gen import TrackGenerator, TrackGenConfig, PerEnvSeededRNG

E, device = 64, "cuda"                       # or "cpu"
config = TrackGenConfig(num_envs=E, half_width=0.03, device=device)
rng = PerEnvSeededRNG(seeds=0, num_envs=E, device=device)
track = TrackGenerator(config, rng).generate()
# track.outer / center / inner: [E*N_max] vec2f, NaN-padded, count[e] real points
```

…or let one object run the whole loop — generation, out-of-bounds checks,
checkpoint progress:

```python
from track_gen.course import Course, CourseConfig

course = Course(CourseConfig(mode="track", gen=config, seeds=42,
                             collision="segments", checkpoint_spacing=0.6))
course.bind(position=robot_pos, yaw=robot_yaw, half_extents=robot_he)
course.generate()                            # whole batch + coherent refresh
res = course.step()                          # events + contacts, every sim step
course.reset(done_mask)                      # respawn finished envs
```

More: [batch generation](https://antoinerichard.github.io/TrackGen/tutorials/batch-of-tracks.html) ·
[runtime utilities](https://antoinerichard.github.io/TrackGen/utilities/overview.html) ·
[CUDA graphs in a sim](https://antoinerichard.github.io/TrackGen/tutorials/cuda-graph-in-a-sim.html)

## Documentation

| | |
|---|---|
| [Getting started](https://antoinerichard.github.io/TrackGen/getting-started/quickstart.html) | install, first batch, parameter explorer |
| [Tutorials](https://antoinerichard.github.io/TrackGen/tutorials/batch-of-tracks.html) | batches, gates, generator choice, CUDA graphs |
| [Generators](https://antoinerichard.github.io/TrackGen/generators/overview.html) | the five families, quality benchmarks |
| [Runtime utilities](https://antoinerichard.github.io/TrackGen/utilities/overview.html) | collision, props, checkpoints & progress, Course |
| [How it works](https://antoinerichard.github.io/TrackGen/how-it-works/resample.html) | resample → XPBD → inflation internals |
| [Configuration](https://antoinerichard.github.io/TrackGen/configuration/reference.html) | every knob, tuning guidance |
| [API reference](https://antoinerichard.github.io/TrackGen/reference/api.html) | the complete public surface |

## Development

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q -m "not slow and not benchmark and not cuda"  # fast lane
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q                                              # full suite
python -m sphinx -W -b html docs docs/_build/html                                                 # docs
```

Contributing guides (dev setup, writing a generator, rendering the
committed figures) live in the
[docs](https://antoinerichard.github.io/TrackGen/contributing/dev-setup.html).
````

NOTE to implementer: verify each absolute URL resolves against the built
docs from Task 1 (`ls` the built HTML tree for each path used above —
getting-started/installation.html, getting-started/quickstart.html,
tutorials/*.html, generators/overview.html, utilities/*.html,
how-it-works/resample.html, configuration/reference.html,
reference/api.html). If a filename differs (e.g. quickstart page name),
fix the LINK to match the real page, not the page.

- [ ] **Step 2: Verify README renders sanely**

Run: `grep -c "antoinerichard.github.io" README.md` (expect ~18) and
`grep -n "docs/assets" README.md` (expect the two relative figure paths).
Read README.md once fully to check formatting (tables well-formed, no
stray fences).

- [ ] **Step 3: Set the repo description**

```bash
gh repo edit AntoineRichard/TrackGen --description "GPU-batched race-track & gate generation with runtime collision, progress and instancing utilities — pure NVIDIA Warp, CUDA-graph native."
```

- [ ] **Step 4: Run the suite once more**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/ -q`
Expected: all PASS (README asset tests unaffected).

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "docs: lean README landing — pitch, feature table, dual quickstart, docs links"
```

---

## Self-Review Notes (completed during planning)

- **Spec coverage:** Part A (hero/badges/table/install/quickstarts/nav/dev, cuts) = Task 2 with full text; Part B (five pages, toctree, cards, gallery, tutorial retirement, api.rst Performance move, ref repointing) = Task 1; Part C (description; homepage already set; badge carries the label) = Task 2 Step 3. Out-of-scope respected (no assets, no code).
- **Placeholder scan:** the `[LIFT: ...]` markers reference exact existing section titles in files present in the repo — they are move instructions, not TBDs; every NEW sentence is written out.
- **Consistency:** page filenames used in Task 1 match every URL in Task 2's README text and the api.rst pointer; figure paths `../assets/...` used from `docs/utilities/` match sibling sections' convention (verify against `docs/generators/*.rst` usage when writing).
