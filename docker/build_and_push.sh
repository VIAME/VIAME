#!/bin/bash
# Build the web and default GPU images, gate both on the CRITICAL example
# tests, and push them only if those pass.
#
# The tests run here, against the finished images with --gpus, rather than
# inside the Dockerfiles. docker build gets the daemon's default runtime,
# normally runc, so a RUN step has no device and every pipeline falls back to
# the CPU: measure_via_default_fish takes 52s on a GPU and does not finish in
# 540s without one, which reports a broken image when nothing is wrong.
#
# Usage: build_and_push.sh [--test-only] [--no-push]
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
# Overridable so the gate can be exercised against a scratch tag.
WEB=${VIAME_WEB_IMAGE:-kitware/viame:gpu-algorithms-web}
DEFAULT=${VIAME_DEFAULT_IMAGE:-kitware/viame:gpu-algorithms-default}

TEST_ONLY=0; NO_PUSH=0
for arg in "$@"; do
  case "$arg" in
    --test-only) TEST_ONLY=1 ;;
    --no-push)   NO_PUSH=1 ;;
    *) echo "unknown option: $arg"; exit 2 ;;
  esac
done

# The CRITICAL set from tests/examples/CMakeLists.txt: test file, class,
# examples subdirectory. Keep in step with add_critical_example_test() there.
CRITICAL_TESTS=(
  "test_image_enhancement.py|TestEnhance|image_enhancement"
  "test_object_detection.py|TestGenericProposalsExample|object_detection"
  "test_object_detection.py|TestFishDetectorExample|object_detection"
  "test_object_tracking.py|TestRunFishTracker|object_tracking"
  "test_object_tracking.py|TestRunGenericTracker|object_tracking"
  "test_size_measurement.py|TestMeasureViaDefaultFish|size_measurement"
  "test_object_detector_training.py|TestTrainNetharnCfrnnFromViameCsv|object_detector_training"
)

# run_critical <image> <require_all_ran>
#
# Example tests skip themselves when the pipeline they drive is not installed,
# which is what lets the same suite gate the lean web image. That also means a
# skip is indistinguishable from a pass, so the default image -- whose whole
# purpose is to carry those model packs -- is required to actually run all of
# them. Otherwise an add-on download that quietly installed nothing would ship.
run_critical() {
  local image="$1" require_all_ran="$2" cid rc=0 ran=0 skipped=0

  if ! docker run --rm --gpus all --entrypoint true "$image" 2>/dev/null; then
    echo "ERROR: no GPU available, refusing to pass $image untested"
    return 1
  fi

  cid=$(docker run -d --gpus all -v "$SRC_DIR/tests:/src_tests:ro" \
          --entrypoint sleep "$image" infinity) || return 1
  # shellcheck disable=SC2064
  trap "docker rm -f $cid >/dev/null 2>&1" RETURN

  docker exec "$cid" bash -c \
    'cp -r /src_tests /tests && python -m pip install --break-system-packages -q pytest' \
    >/dev/null 2>&1 || { echo "ERROR: could not stage pytest in $image"; return 1; }

  for entry in "${CRITICAL_TESTS[@]}"; do
    IFS='|' read -r file cls dir <<< "$entry"
    local out
    out=$(docker exec "$cid" bash -c "
      I=/opt/noaa/viame
      PYV=\$(python -c 'import sys;print(f\"{sys.version_info.major}.{sys.version_info.minor}\")')
      export PYTHONPATH=\"\$I/python:\$I/lib/python\$PYV/site-packages:/tests/common:/tests/examples\"
      export VIAME_INSTALL=\"\$I\"
      cd \"\$I/examples/$dir\" && \
      python -m pytest /tests/examples/$file -k $cls -q -p no:cacheprovider" 2>&1)
    if [ $? -ne 0 ]; then
      echo "  FAIL  $cls"
      echo "$out" | tail -15 | sed 's/^/        /'
      rc=1
    elif grep -q "skipped" <<< "$out"; then
      echo "  SKIP  $cls  (pipeline not installed)"
      skipped=$((skipped + 1))
    else
      echo "  PASS  $cls"
      ran=$((ran + 1))
    fi
  done

  echo "  -> $ran ran, $skipped skipped"
  if [ "$require_all_ran" = "1" ] && [ "$skipped" -ne 0 ]; then
    echo "ERROR: $image skipped $skipped test(s); its model packs are missing"
    rc=1
  fi
  return $rc
}

if [ "$TEST_ONLY" -eq 0 ]; then
  echo "=== building $WEB ==="
  docker image rm -f "$WEB" >/dev/null 2>&1
  docker build --no-cache -t "$WEB" -f "$SRC_DIR/docker/viame_gpu_web.docker" \
    "$SRC_DIR/docker" > web_build.log 2>&1 \
    || { echo "web build failed, see web_build.log"; exit 1; }

  echo "=== building $DEFAULT (web plus model packs) ==="
  docker image rm -f "$DEFAULT" >/dev/null 2>&1
  docker build --no-cache -t "$DEFAULT" -f "$SRC_DIR/docker/viame_gpu_default.docker" \
    "$SRC_DIR/docker" > default_build.log 2>&1 \
    || { echo "default build failed, see default_build.log"; exit 1; }
fi

# The web image ships no add-on models, so its model-driven tests skip.
echo "=== CRITICAL tests: $WEB ==="
run_critical "$WEB" 0 || { echo "web image FAILED its tests, nothing pushed"; exit 1; }

echo "=== CRITICAL tests: $DEFAULT ==="
run_critical "$DEFAULT" 1 || { echo "default image FAILED its tests, nothing pushed"; exit 1; }

if [ "$NO_PUSH" -eq 1 ]; then
  echo "all tests passed; --no-push given, stopping here"
  exit 0
fi

echo "=== pushing both images ==="
docker push "$WEB" || exit 1
docker push "$DEFAULT" || exit 1
echo "done: $WEB and $DEFAULT built, tested and pushed"
