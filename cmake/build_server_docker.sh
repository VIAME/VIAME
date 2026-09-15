#!/bin/bash

# VIAME's build inside a Docker image: the `build` stage of docker/Dockerfile.
#
# One script for what were four -- build_server_docker_web.sh,
# build_server_docker_web_cu11.sh, build_server_docker_web_ifremer.sh and
# build_server_docker_everything.sh -- which differed in a preset, a CUDA
# architecture list, a few -D flags, a job limit, a legacy mmdet tarball and
# whether the build tree was kept. Those are the arguments now, from the
# Dockerfile's build args:
#
#   VIAME_PRESET        configure preset (CMakePresets.json)   docker-web
#   CUDA_ARCHITECTURES  CUDA architectures to compile for       the preset's
#   VIAME_CMAKE_ARGS    extra -D flags                          none
#   VIAME_BUILD_JOBS    make -j                                 nproc
#   VIAME_LEGACY_MMDET  install the old mmdet plugin tarball    OFF
#   VIAME_KEEP_BUILD    keep /viame and its build tree          OFF
#
# Dropped from the old scripts: `download_opencv_extras`, OpenCV's aux files,
# which nothing has used since P7 removed OpenCV, and `fix_libsvm_symlink`,
# for a libsvm.so.2 that libsvm, vendored and static since P1, no longer
# installs.

set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/build_common_functions.sh"

VIAME_PRESET="${VIAME_PRESET:-docker-web}"

install_system_deps apt
install_cmake

# The build context is a checkout with its submodules checked out; a context
# that carries .git can still bring them up to date.
if [ -d /viame/.git ]; then
  update_git_submodules /viame
fi
setup_build_directory /viame

setup_basic_build_environment /viame/build/install /usr/local/cuda

configure_args=()
if [ -n "${CUDA_ARCHITECTURES:-}" ]; then
  configure_args+=("-DCUDA_ARCHITECTURES:STRING=${CUDA_ARCHITECTURES}")
fi
# Word-split on purpose: a list of -D flags.
# shellcheck disable=SC2206
configure_args+=(${VIAME_CMAKE_ARGS:-})

cmake -S .. -B . --preset "$VIAME_PRESET" "${configure_args[@]}"

if [ -n "${VIAME_BUILD_JOBS:-}" ]; then
  export VIAME_BUILD_JOBS
  export MAX_JOBS="$VIAME_BUILD_JOBS"
fi

run_build build_log.txt true

# The old mmdet plugin 1-2 models use, a cp310 binary: only for images whose
# python is 3.10. VIAME-Web does not handle binary code in add-ons.
if [ "${VIAME_LEGACY_MMDET:-OFF}" = "ON" ]; then
  wget https://viame.kitware.com/api/v1/file/685cd1a5a2df48d3c1ae8604/download
  tar -xvf download
  cp -r lib install
  rm -rf lib download
fi

# Only meaningful with a GPU, which `docker build` does not have; see
# build_server_docker_default.sh for the gate that runs with --gpus.
CRITICAL_TESTS_STATUS=0
if nvidia-smi -L >/dev/null 2>&1; then
  run_critical_tests /viame/build /viame/build/install || CRITICAL_TESTS_STATUS=$?
else
  echo "No GPU visible to the build, skipping CRITICAL tests"
  CRITICAL_TESTS_SKIPPED=1
fi

if [ "${VIAME_KEEP_BUILD:-OFF}" = "ON" ]; then
  finalize_docker_install /viame/build false
  ln -s /opt/noaa/viame /viame/build/install
else
  finalize_docker_install /viame/build
fi

# Mark the image rather than fail the RUN, which would discard the whole build
if [ -n "${CRITICAL_TESTS_SKIPPED:-}" ]; then
  echo "CRITICAL_TESTS_SKIPPED" > /opt/noaa/viame/CRITICAL_TESTS_SKIPPED
  echo "CRITICAL tests skipped, validate this image with --gpus"
elif [ "$CRITICAL_TESTS_STATUS" -ne 0 ]; then
  echo "CRITICAL_TESTS_FAILED" > /opt/noaa/viame/CRITICAL_TESTS_FAILED
  echo "CRITICAL tests FAILED -- image is marked broken"
else
  echo "All CRITICAL tests passed"
fi
