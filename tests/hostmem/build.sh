#!/bin/sh
# Shared-memory aliasing gate (tests/hostmem/main.cpp): builds and runs.
# Links the real common/HostSys.cpp out of a cmake build of the core --
# LRPS2_BUILD names that build directory (default: build/macos-arm64, what
# cores/build.sh leaves on an Apple-silicon Mac). Runnable from any directory.
set -e
DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT=$(CDPATH= cd -- "$DIR/../.." && pwd)
BUILD=${LRPS2_BUILD:-$ROOT/build/macos-arm64}
CXX=${CXX:-c++}
LIB="$BUILD/common/libcommon.a"
[ -f "$LIB" ] || { echo "hostmem: no $LIB -- build the core first (cmake --build <dir> --target pcsx2_libretro)"; exit 1; }

CC=${CC:-cc}
INC="-I $ROOT -I $ROOT/common -I $ROOT/libretro/libretro-common/include"
EXTRA=""
case "$(uname -s)" in
  Darwin) EXTRA="-framework CoreFoundation -framework Foundation" ;;
  *)      EXTRA="-lpthread -lrt" ;;
esac

# libcommon.a's threading sits on libretro-common's rthreads, which the core's
# own target compiles; compiled here from the same source.
"$CC" -O1 -g -c $INC -o "$DIR/rthreads.o" "$ROOT/libretro/libretro-common/rthreads/rthreads.c"
"$CXX" -std=c++17 -O1 -g -Wall $INC \
  -o "$DIR/hostmem_test" "$DIR/main.cpp" "$LIB" "$DIR/rthreads.o" $EXTRA
"$DIR/hostmem_test"
