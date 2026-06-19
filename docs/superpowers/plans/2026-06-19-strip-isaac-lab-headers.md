# Strip Isaac Lab license headers Implementation Plan

> **For agentic workers:** small mechanical task — single edit pass + verification, one commit.

**Goal:** Remove the copied "Isaac Lab Project Developers / BSD-3-Clause" license header from every source file and the README, leaving no copyright/license notice behind (no replacement header, no LICENSE file).

**Decision (confirmed):** Remove entirely. No NVIDIA/own-header replacement, no LICENSE/NOTICE added.

## Scope

34 `.py` files carry the header (auto-discovered by the copyright line), in two byte-variants, plus the README License section. No `LICENSE` file exists. Nothing under `docs/superpowers/spikes/` has the header.

- **Variant A (4 lines):**
  ```
  # Copyright (c) 2022-2025, The Isaac Lab Project Developers.
  # All rights reserved.
  #
  # SPDX-License-Identifier: BSD-3-Clause
  ```
- **Variant B (2 lines):**
  ```
  # Copyright (c) 2022-2025, The Isaac Lab Project Developers.
  # SPDX-License-Identifier: BSD-3-Clause
  ```
- **Shebang files** (`viz/make_report.py`, `viz/param_explorer.py`, `viz/plot_ablations.py`, `viz/plot_tracks.py`): `#!/usr/bin/env python3` on line 1 precedes a Variant-A header — the shebang MUST be preserved.

## Global Constraints

- No header replacement; no `LICENSE`/`NOTICE` added.
- Preserve shebang lines.
- Comment-only change → `pytest -q` must stay **233 passed** on the Warp `cpu` device.
- GPG signing fails in this env → commit with `--no-gpg-sign`.
- Do this on a branch off `main` (not directly on `main`).

---

### Task 1: Strip the headers

**Files:** all 34 header-bearing `.py` files (auto-scoped by the script) + `README.md`.

- [ ] **Step 1: Branch off main**

```bash
cd /home/antoiner/Documents/TrackGen
git checkout main && git checkout -b chore/strip-isaac-lab-headers
```

- [ ] **Step 2: Remove the header from every `.py` file**

Run (substring removal — handles both variants, preserves shebangs, and drops one immediately-following blank line so files don't start blank):

```bash
cd /home/antoiner/Documents/TrackGen
.venv/bin/python - <<'PY'
import glob, pathlib

HEADER_A = ("# Copyright (c) 2022-2025, The Isaac Lab Project Developers.\n"
            "# All rights reserved.\n#\n# SPDX-License-Identifier: BSD-3-Clause\n")
HEADER_B = ("# Copyright (c) 2022-2025, The Isaac Lab Project Developers.\n"
            "# SPDX-License-Identifier: BSD-3-Clause\n")

files = [p for p in glob.glob("**/*.py", recursive=True)
         if not (p.startswith(".venv") or ".worktrees" in p or ".git" in p)]
changed = 0
for p in files:
    s = pathlib.Path(p).read_text()
    if "The Isaac Lab Project Developers" not in s:
        continue
    # Variant A first (more specific); HEADER_B is NOT a substring of A files.
    for variant in (HEADER_A, HEADER_B):
        if variant in s:
            if variant + "\n" in s:                 # header followed by a blank line
                s = s.replace(variant + "\n", "", 1)
            else:                                    # header immediately followed by content
                s = s.replace(variant, "", 1)
            break
    pathlib.Path(p).write_text(s)
    changed += 1
print(f"stripped headers from {changed} files")
PY
```
Expected: `stripped headers from 34 files`.

- [ ] **Step 3: Remove the README License section**

```bash
cd /home/antoiner/Documents/TrackGen
.venv/bin/python - <<'PY'
import pathlib
p = pathlib.Path("README.md")
s = p.read_text()
block = ("\n## License\n\n"
         "BSD-3-Clause. Copyright (c) 2022-2025, The Isaac Lab Project Developers.\n")
assert block in s, "README License block not found verbatim — inspect tail of README.md"
s = s.replace(block, "")
if not s.endswith("\n"):
    s += "\n"
p.write_text(s)
print("removed README License section")
PY
```

- [ ] **Step 4: Verify no header residue anywhere**

```bash
cd /home/antoiner/Documents/TrackGen
echo "Isaac Lab refs (expect 0):"
grep -rn "The Isaac Lab Project Developers" --include=*.py --include=*.md . | grep -vE "\.venv/|\.worktrees/|\.git/|docs/superpowers/" | wc -l
echo "SPDX lines in .py (expect 0):"
grep -rn "SPDX-License-Identifier" --include=*.py . | grep -vE "\.venv/|\.worktrees/|\.git/" | wc -l
echo "Files now starting with a blank line (expect none):"
for f in track_gen/__init__.py track_gen/_src/warp_pipeline.py viz/make_report.py benchmarks/benchmark_pipeline.py; do
  head -1 "$f" | grep -qE '^\s*$' && echo "BLANK FIRST LINE: $f"; done
echo "Shebangs preserved (expect 4):"
grep -rln "^#!/usr/bin/env python3" --include=*.py viz/ | wc -l
```
Expected: `0`, `0`, no `BLANK FIRST LINE` output, `4`.

- [ ] **Step 5: Confirm files still import / parse and suite is green**

```bash
cd /home/antoiner/Documents/TrackGen
.venv/bin/python -c "import track_gen; print(sorted(track_gen.__all__))"
.venv/bin/python -m pytest -q
```
Expected: the 7-name public API prints; **233 passed** (header removal is comment-only).

- [ ] **Step 6: Commit**

```bash
cd /home/antoiner/Documents/TrackGen
git add -u
git add README.md
git status   # confirm only the 34 .py + README.md are staged, nothing else
git commit --no-gpg-sign -m "chore: remove Isaac Lab license headers from sources and README"
```

---

## Self-Review

- **Scope:** auto-discovered by the copyright line (34 files) → no hardcoded list to drift; both byte-variants handled; shebangs preserved; README section removed by verbatim match (asserts presence).
- **No placeholders:** exact scripts and expected outputs given.
- **Safety:** comment-only change; verification greps must read 0/0 and the suite must stay 233.

## Notes / implications

- After this, source files carry **no copyright or license notice** and the repo has no `LICENSE`. That's the requested state (suitable for private/proprietary or not-yet-licensed). If this code is later open-sourced — especially via NVIDIA OSRB — headers + a `LICENSE`/`NOTICE` will need to be added back (the `osrb-apache2-opensource-compliance` skill checks for exactly that).
- `pyproject.toml` has no license metadata to change. The `viz/*` shebangs and all module docstrings are untouched.
