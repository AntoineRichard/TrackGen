# docs/_data: generated numbers shown in the docs

Every benchmark number the docs show lives here and is pulled into the pages with
`csv-table :file:` (tables), `literalinclude` (raw benchmark output) or `include` (the
machine line of a caption). Nothing here is typed by hand: each file is written by the
command below, and prose never retypes a GPU number. This directory is not a Sphinx
source (`exclude_patterns`).

These datasets predate the 0.3 built-in hashed-RNG switch and are historical reference
measurements. Running their commands against current source measures the new seed
mapping and poison default; it does not reproduce these stored numbers. The 0.3 RNG
distribution gate is documented separately in `docs/upgrading/from-0.2.rst`.

CPU files: every column but the timings (`gen_ms_per_call`, the collision `ms` lines)
is deterministic for a fixed source revision, Warp version and platform; the
timings depend on the machine and its load.

GPU files: measured on the machine named in `machine-cuda.txt` (an NVIDIA GB10, an
integrated-memory GPU: absolute timings differ on a discrete GPU; re-run the commands on
your machine to compare). Before a GPU file is measured it can hold a placeholder: the
header the command writes plus one row (or line) reading `pending GPU re-measurement`,
written by the same tool with `--placeholder`.

| File | Status | Command (run from the repository root) |
|---|---|---|
| `metrics-cpu.csv` | CPU, measured | `python -m benchmarks.compare_generators --E 512 --seed 0 --generators bezier checkpoint hull polar voronoi \| python docs/_tools/md_table_to_csv.py > docs/_data/metrics-cpu.csv` |
| `collision-accuracy-cpu.txt` | CPU, measured | `python -m benchmarks.benchmark_collision --E 8192 --cpu 2>&1 \| python docs/_tools/strip_warp_log.py > docs/_data/collision-accuracy-cpu.txt` |
| `collision-racing-accuracy-cpu.txt` | CPU, measured | `python -m benchmarks.benchmark_collision --racing --E 100 --B 1024 --no-timing --cpu 2>&1 \| python docs/_tools/strip_warp_log.py > docs/_data/collision-racing-accuracy-cpu.txt` |
| `machine-cpu.txt` | CPU | one line: CPU model, OS, Warp version and device, date of the CPU measurements |
| `metrics-repulsive-cuda.csv` | GPU, measured | `python -m benchmarks.compare_generators --E 512 --seed 0 --cuda --generators repulsive \| python docs/_tools/md_table_to_csv.py > docs/_data/metrics-repulsive-cuda.csv` |
| `cost-cuda-e4096.csv` | GPU, measured | `python -m benchmarks.compare_generators --E 4096 --seed 0 --cuda \| python docs/_tools/md_table_to_csv.py --columns generator,yield,gen_ms_per_call > docs/_data/cost-cuda-e4096.csv` |
| `cost-cuda-e8192.csv` | GPU, measured | `python -m benchmarks.compare_generators --E 8192 --seed 0 --cuda --generators bezier checkpoint hull polar voronoi \| python docs/_tools/md_table_to_csv.py --columns generator,yield,gen_ms_per_call > docs/_data/cost-cuda-e8192.csv` |
| `cost-repulsive-cuda.csv` | GPU, measured | `python -m benchmarks.compare_generators --E 64 --seed 0 --cuda --generators repulsive \| python docs/_tools/md_table_to_csv.py --columns generator,gen_ms_per_call --add-column E=64 > docs/_data/cost-repulsive-cuda.csv` then `python -m benchmarks.compare_generators --E 8192 --seed 0 --cuda --generators repulsive \| python docs/_tools/md_table_to_csv.py --columns generator,gen_ms_per_call --add-column E=8192 --no-header >> docs/_data/cost-repulsive-cuda.csv` |
| `collision-cuda-default.txt` | GPU, measured | `python -m benchmarks.benchmark_collision --E 8192 2>&1 \| python docs/_tools/strip_warp_log.py > docs/_data/collision-cuda-default.txt` |
| `collision-cuda-racing.txt` | GPU, measured | `python -m benchmarks.benchmark_collision --racing 2>&1 \| python docs/_tools/strip_warp_log.py > docs/_data/collision-cuda-racing.txt` |
| `collision-cuda-racing-b16.txt` | GPU, measured | `python -m benchmarks.benchmark_collision --racing --B 16 2>&1 \| python docs/_tools/strip_warp_log.py > docs/_data/collision-cuda-racing-b16.txt` |
| `machine-cuda.txt` | GPU, measured | one line written from `nvidia-smi --query-gpu=name,driver_version --format=csv,noheader` and `python -c "import warp; print(warp.__version__)"`, plus the date |

The placeholders were written with, for example:

```bash
python docs/_tools/md_table_to_csv.py --columns generator,yield,gen_ms_per_call \
    --placeholder "pending GPU re-measurement" > docs/_data/cost-cuda-e4096.csv
```

`strip_warp_log.py` removes Warp's init banner and its `Module ... load on device ...`
lines, which go to stdout (so `2>/dev/null` does not remove them, and their `device` would
match a `literalinclude :start-at: device`). The collision outputs then start at their own
`device ...` line.
