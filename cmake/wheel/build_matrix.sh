#!/usr/bin/env bash
#
# Build a VIAME wheel for each python version, on Linux.
#
#   cmake/wheel/build_matrix.sh --source ~/Dev/viame-lite/src --build ~/wheels
#
# One wheel per interpreter, because a wheel with compiled extensions is
# specific to the version that built it: the modules use the full CPython API
# and link `libpython3.X.so`, so there is no `abi3` wheel to build instead.
# See docs/wheels.md.
#
# The expensive part is VIAME's own C++, and only the quarter of it that
# touches the python API is recompiled per version -- nothing third party is
# rebuilt, torch included, which comes from pip at install time.
#
# A version that fails does not stop the others. Each one's log is kept and
# the summary at the end says which wheels exist.

set -u -o pipefail

VERSIONS_DEFAULT="3.10 3.11 3.12 3.13 3.14"

usage()
{
  sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
  cat <<USAGE
Options:
  --source DIR      VIAME source tree            (default: this script's repo)
  --build DIR       where the per-version trees and wheels go  (required)
  --versions "..."  space separated              (default: $VERSIONS_DEFAULT)
  --jobs N          make -j                      (default: nproc/2, min 1)
  --cmake-arg ARG   passed to every configure; repeatable
  --keep-going      no effect; failures never stop the matrix (kept for clarity)
  --help
USAGE
}

SOURCE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )"
BUILD_ROOT=""
VERSIONS="$VERSIONS_DEFAULT"
JOBS=""
CMAKE_ARGS=()

while [ $# -gt 0 ]; do
  case "$1" in
    --source)     SOURCE_DIR="$2"; shift 2 ;;
    --build)      BUILD_ROOT="$2"; shift 2 ;;
    --versions)   VERSIONS="$2";   shift 2 ;;
    --jobs)       JOBS="$2";       shift 2 ;;
    --cmake-arg)  CMAKE_ARGS+=("$2"); shift 2 ;;
    --keep-going) shift ;;
    --help|-h)    usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [ -z "$BUILD_ROOT" ]; then
  echo "--build is required" >&2; usage >&2; exit 2
fi
if [ ! -f "$SOURCE_DIR/CMakeLists.txt" ]; then
  echo "not a VIAME source tree: $SOURCE_DIR" >&2; exit 2
fi

# Half the cores by default. These machines are shared, and a wheel build that
# takes the whole box is a wheel build someone kills.
if [ -z "$JOBS" ]; then
  JOBS=$(( $(nproc 2>/dev/null || echo 2) / 2 ))
  [ "$JOBS" -lt 1 ] && JOBS=1
fi

mkdir -p "$BUILD_ROOT"
WHEELHOUSE="$BUILD_ROOT/wheelhouse"
mkdir -p "$WHEELHOUSE"

echo "source     : $SOURCE_DIR"
echo "build root : $BUILD_ROOT"
echo "versions   : $VERSIONS"
echo "jobs       : $JOBS"
echo

declare -a RESULTS=()

for version in $VERSIONS; do
  python_exe="$(command -v "python${version}" 2>/dev/null || true)"

  if [ -z "$python_exe" ]; then
    echo "== python${version}: not installed, skipping"
    RESULTS+=("${version}|skipped|no python${version} on PATH")
    continue
  fi

  tree="$BUILD_ROOT/build-py${version}"
  log="$BUILD_ROOT/build-py${version}.log"
  prefix="$tree/install"

  echo "== python${version} ($python_exe)"
  echo "   tree $tree"
  echo "   log  $log"

  mkdir -p "$tree"
  start=$(date +%s)

  {
    echo "### configure"
    cmake -S "$SOURCE_DIR" -B "$tree" \
          -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_INSTALL_PREFIX="$prefix" \
          -DPYTHON_EXECUTABLE="$python_exe" \
          -DPython3_EXECUTABLE="$python_exe" \
          -DVIAME_ENABLE_PYTHON=ON \
          "${CMAKE_ARGS[@]+"${CMAKE_ARGS[@]}"}" &&
    echo "### build" &&
    cmake --build "$tree" --parallel "$JOBS" &&
    echo "### install" &&
    cmake --install "$tree" 2>/dev/null || cmake --build "$tree" --target install &&
    echo "### wheel" &&
    cmake --build "$tree" --target wheel
  } > "$log" 2>&1

  status=$?
  elapsed=$(( $(date +%s) - start ))

  if [ $status -ne 0 ]; then
    echo "   FAILED after ${elapsed}s -- see $log"
    RESULTS+=("${version}|failed|${elapsed}s, $(grep -m1 -iE 'error:' "$log" | cut -c1-70)")
    continue
  fi

  # `make wheel` prints the path; take it from the tree rather than parsing.
  wheel="$(ls -t "$tree"/wheel/*.whl 2>/dev/null | head -1)"

  if [ -z "$wheel" ]; then
    echo "   built, but produced no wheel -- see $log"
    RESULTS+=("${version}|failed|built but no .whl")
    continue
  fi

  cp "$wheel" "$WHEELHOUSE/"
  echo "   ok in ${elapsed}s -> $(basename "$wheel")"
  RESULTS+=("${version}|ok|$(basename "$wheel")")
done

echo
echo "================================ summary ================================"
printf "  %-6s %-8s %s\n" "python" "result" "wheel"
ok=0
for row in "${RESULTS[@]+"${RESULTS[@]}"}"; do
  IFS='|' read -r v r d <<< "$row"
  printf "  %-6s %-8s %s\n" "$v" "$r" "$d"
  [ "$r" = "ok" ] && ok=$(( ok + 1 ))
done
echo
echo "  $ok wheel(s) in $WHEELHOUSE"

# Non-zero when nothing was produced, so CI notices; a partial matrix is a
# pass, since a version failing is usually that version not being supported
# yet rather than the tree being broken.
[ "$ok" -gt 0 ]
