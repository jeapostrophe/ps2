#!/bin/sh
# Shared-memory aliasing gate (tests/hostmem/main.cpp): builds and runs.
# Links the real common/HostSys.cpp, rebuilt from the working tree on every run
# (`cmake --build --target common`), so it never runs over a stale libcommon.a.
# LRPS2_BUILD names the cmake build directory (default: build/tests-host, this
# gate's own, configured on first use). On Apple it also compiles the GS's
# wrapped memory (pcsx2/GS/GSWrappedMemoryDarwin.cpp) at GSLocalMemory's own
# size and repeat. Runnable from any directory.
set -e
DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT=$(CDPATH= cd -- "$DIR/../.." && pwd)
BUILD=${LRPS2_BUILD:-$ROOT/build/tests-host}
CXX=${CXX:-c++}
CC=${CC:-cc}
JOBS=$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)

if [ ! -f "$BUILD/CMakeCache.txt" ]; then
  mkdir -p "$BUILD"
  cmake -S "$ROOT" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.18 \
    >"$BUILD/hostmem-configure.log" 2>&1 \
    || { echo "hostmem: configuring $BUILD failed -- see $BUILD/hostmem-configure.log"; exit 1; }
fi
cmake --build "$BUILD" --target common --parallel "$JOBS" >"$BUILD/hostmem-build.log" 2>&1 \
  || { tail -30 "$BUILD/hostmem-build.log"; echo "hostmem: building libcommon in $BUILD failed"; exit 1; }
LIB="$BUILD/common/libcommon.a"
[ -f "$LIB" ] || { echo "hostmem: the build produced no $LIB"; exit 1; }

INC="-I $ROOT -I $ROOT/common -I $ROOT/libretro/libretro-common/include"
EXTRA=""
GS=""
case "$(uname -s)" in
  Darwin)
    EXTRA="-framework CoreFoundation -framework Foundation"
    # The GS's size and repeat, read from where the GS says them.
    VMSIZE=$(sed -n 's/.*static constexpr int m_vmsize = \(.*\);.*/\1/p' "$ROOT/pcsx2/GS/GSLocalMemory.h" | tr -d ' ')
    REPEAT=$(sed -n 's/.*GSAllocateWrappedMemory(m_vmsize, \([0-9]*\)).*/\1/p' "$ROOT/pcsx2/GS/GSLocalMemory.cpp")
    [ -n "$VMSIZE" ] && [ -n "$REPEAT" ] \
      || { echo "hostmem: could not read the GS's m_vmsize/repeat from pcsx2/GS/GSLocalMemory.{h,cpp}"; exit 1; }
    GS="-DHOSTMEM_GS_VMSIZE=($VMSIZE) -DHOSTMEM_GS_REPEAT=$REPEAT $ROOT/pcsx2/GS/GSWrappedMemoryDarwin.cpp"
    ;;
  *) EXTRA="-lpthread -lrt" ;;
esac

# libcommon.a's threading sits on libretro-common's rthreads, which the core's
# own target compiles; compiled here from the same source.
"$CC" -O1 -g -c $INC -o "$DIR/rthreads.o" "$ROOT/libretro/libretro-common/rthreads/rthreads.c"
"$CXX" -std=c++17 -O1 -g -Wall $INC \
  -o "$DIR/hostmem_test" "$DIR/main.cpp" $GS "$LIB" "$DIR/rthreads.o" $EXTRA
"$DIR/hostmem_test"
