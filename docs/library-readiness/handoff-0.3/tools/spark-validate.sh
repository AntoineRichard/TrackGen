#!/bin/bash
# Same-machine A/B on spark-home: baselines from v0.2.0.dev5, then PR14 and PR15.
T=~/code/TrackGen-test; PY=~/venvs/trackgen/bin/python; S=$T/snapshots
cd $T/dev5 && export PYTHONPATH=$PWD
echo "=== DEV5 BASELINES"
timeout 1500 $PY -W ignore ../tools/full_snapshot.py cuda:0 $S/full-dev5.npz 2>&1 | tail -1
timeout 1500 $PY -W ignore ../tools/full_snapshot.py cuda:0 $S/full-dev5b.npz 2>&1 | tail -1
$PY ../tools/full_snapshot.py --compare $S/full-dev5.npz $S/full-dev5b.npz
timeout 2500 $PY -W ignore ../tools/gate_snapshot.py cuda:0 $S/gates-dev5.npz 2>&1 | grep arrays
for n in pr14 pr15; do
  cd $T/$n && export PYTHONPATH=$PWD
  echo "=== $n SNAPSHOTS vs dev5"
  timeout 1500 $PY -W ignore ../tools/full_snapshot.py cuda:0 $S/full-$n.npz 2>&1 | tail -1
  $PY ../tools/full_snapshot.py --compare $S/full-dev5.npz $S/full-$n.npz
  timeout 2500 $PY -W ignore ../tools/gate_snapshot.py cuda:0 $S/gates-$n.npz 2>&1 | grep arrays
  $PY ../tools/gate_snapshot.py --compare $S/gates-dev5.npz $S/gates-$n.npz
  echo "=== $n NEW CUDA TESTS"
  f=tests/test_components_cuda.py; [ $n = pr15 ] && f="tests/test_components_cuda.py tests/test_course_composition_cuda.py"
  timeout 3000 $PY -m pytest -q -p no:cacheprovider -s $f 2>&1 | grep -iE "ratio|probe|cost|passed|failed|error" | tail -25
  echo "=== $n CUDA LANE"
  timeout 4000 $PY -m pytest -q -p no:cacheprovider -m cuda tests 2>&1 | grep -E "^(FAILED|ERROR)|passed|failed" | tail -30
  echo "=== $n FULL SUITE"
  timeout 9000 $PY -m pytest -q -p no:cacheprovider tests 2>&1 | grep -E "^(FAILED|ERROR)|passed|failed" | tail -40
done
