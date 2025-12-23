#!/usr/bin/env bash
set -e

# ===== default values =====
BUILD_DIR=build
BUILD_TYPE=Release
JOBS=$(nproc 2>/dev/null || sysctl -n hw.ncpu)
GENERATOR=""
CLEAN=0
VERBOSE=0
EXTRA_CMAKE_ARGS=""

# ===== help info =====
usage() {
  cat <<EOF
Usage: ./build.sh [options]

Options:
  -d, --debug            Build Debug (default: Release)
  -r, --release          Build Release
  -c, --clean            Remove build directory before build
  -j, --jobs N           Parallel build jobs (default: auto)
  -B, --build-dir DIR    Build directory (default: build)
  -G, --generator GEN    CMake generator (e.g. Ninja)
  -v, --verbose          Verbose build
  -h, --help             Show this help

Examples:
  ./build.sh
  ./build.sh -d
  ./build.sh -c -j 8
  ./build.sh -B build-debug -d
  ./build.sh -G Ninja
EOF
}

while [[ $# -gt 0 ]]; do
  case $1 in
    -d|--debug)
      BUILD_TYPE=Debug
      ;;
    -r|--release)
      BUILD_TYPE=Release
      ;;
    -c|--clean)
      CLEAN=1
      ;;
    -j|--jobs)
      JOBS="$2"
      shift
      ;;
    -B|--build-dir)
      BUILD_DIR="$2"
      shift
      ;;
    -G|--generator)
      GENERATOR="$2"
      shift
      ;;
    -v|--verbose)
      VERBOSE=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      EXTRA_CMAKE_ARGS="$EXTRA_CMAKE_ARGS $1"
      ;;
  esac
  shift
done

# ===== clean =====
if [[ $CLEAN -eq 1 ]]; then
  echo "[build.sh] Cleaning ${BUILD_DIR}"
  rm -rf "${BUILD_DIR}"
fi

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# ===== cmake =====
CMAKE_CMD="cmake .. -DCMAKE_BUILD_TYPE=${BUILD_TYPE}"

if [[ -n "${GENERATOR}" ]]; then
  CMAKE_CMD+=" -G \"${GENERATOR}\""
fi

if [[ $VERBOSE -eq 1 ]]; then
  CMAKE_CMD+=" -DCMAKE_VERBOSE_MAKEFILE=ON"
fi

CMAKE_CMD+=" ${EXTRA_CMAKE_ARGS}"

echo "[build.sh] Configure:"
echo "  ${CMAKE_CMD}"
eval "${CMAKE_CMD}"

# ===== build =====
echo "[build.sh] Build (${JOBS} jobs)"
cmake --build . -j "${JOBS}"

echo "[build.sh] Done ✔"
