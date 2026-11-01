#!/bin/bash
# Build sim/lander_sim*.so (C++ flights for training; see sim/lander_sim.cpp). Box2D 2.3.x from box2d-py 2.3.5.
# -ffp-contract=off: no fused multiply-adds, so float maths rounds like the reference build of Box2D.
set -e
cd "$(dirname "$0")"
PY=../.venv/bin/python
INC="$($PY -c 'import pybind11; print(pybind11.get_include())') $($PY -c 'import sysconfig; print(sysconfig.get_paths()["include"])')"
OUT=${OUTDIR:-.}/lander_sim$($PY -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')
FLAGS="${OPT:--O2} -fPIC -ffp-contract=off -fno-fast-math -I. $(for i in $INC; do echo -n "-I$i "; done)"
BUILD=${BUILD:-build}; mkdir -p $BUILD
objs=()
for f in $(find Box2D -name '*.cpp'); do
  o=$BUILD/$(echo "$f" | tr '/' '_' | sed 's/\.cpp$/.o/')
  [ "$o" -nt "$f" ] || g++ $FLAGS -c "$f" -o "$o" &
  objs+=("$o")
  [ $(jobs -r | wc -l) -ge ${JOBS:-2} ] && wait -n
done
wait
g++ $FLAGS -std=c++17 -shared lander_sim.cpp "${objs[@]}" -o "$OUT"
echo "built sim/$OUT"
