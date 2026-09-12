#!/bin/sh
# arm64 JIT code-memory decisions (tests/jitmem/main.cpp): builds and runs.
# Header-only and arithmetic-only -- maps, writes and executes no code -- so it
# runs on any host. Runnable from any directory.
set -e
DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT=$(CDPATH= cd -- "$DIR/../.." && pwd)
CXX=${CXX:-c++}
"$CXX" -std=c++17 -O1 -g -Wall -I "$ROOT" -I "$ROOT/libretro/libretro-common/include" \
  -o "$DIR/jitmem_test" "$DIR/main.cpp"
"$DIR/jitmem_test"
