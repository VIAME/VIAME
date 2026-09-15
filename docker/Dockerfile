# syntax=docker/dockerfile:1.5
##############################################################################
# VIAME's Docker images, from one Dockerfile.
#
# Build from the top of a VIAME checkout: the checkout is the build context, so
# the image is built from this tree rather than from a fresh clone of main.
# The root `.dockerignore` keeps build trees and caches out of the context and
# keeps `.git` in, so the build checks out any submodules the checkout has not.
#
#   web          VIAME-Web's image: the docker-web preset, no model packs
#     docker build -f docker/Dockerfile --target web -t kitware/viame:gpu-algorithms-web .
#
#   default      web plus the default model packs
#     docker build -f docker/Dockerfile --target default -t kitware/viame:gpu-algorithms-default .
#
#   everything   the docker preset with training and the larger packs, kept
#                on the development image it was built in
#     docker build -f docker/Dockerfile --target everything \
#       --build-arg VIAME_PRESET=docker \
#       --build-arg VIAME_CMAKE_ARGS="-DVIAME_ENABLE_PYTORCH-LEARN=ON -DVIAME_ENABLE_WEB_EXCLUDES=ON -DVIAME_DOWNLOAD_MODELS-ARCTIC-SEAL=ON -DVIAME_DOWNLOAD_MODELS-HABCAM=ON -DVIAME_DOWNLOAD_MODELS-MOUSS-DEEP7=ON" \
#       -t kitware/viame:gpu-algorithms-everything .
#
# The IFREMER web image is the web target with other arguments. The CUDA
# 11.8 image is not buildable from this tree: it built PyTorch from source,
# which is gone, and there is no cu118 lock or PyTorch 2.12 cu118 wheel.
#
#   --build-arg CUDA_ARCHITECTURES=7.5 --build-arg VIAME_BUILD_JOBS=4 \
#   --build-arg VIAME_LEGACY_MMDET=ON --build-arg VIAME_KEEP_BUILD=ON
#
# The CRITICAL tests are not run here: `docker build` gets no GPU, and the
# heavy pipelines time out on the CPU. `cmake/build_server_docker_default.sh`
# runs them against the finished images with --gpus.
##############################################################################

# The CUDA version has to be one the PyTorch wheels in the lock were built
# for (cuda12.lock names the cu126 index; cuda13.lock is picked from 13.0);
# configure stops on any other.
ARG CUDA_IMAGE=nvidia/cuda:12.6.3-cudnn
ARG BASE_OS=ubuntu24.04

# ---------------------------------------------------------------------------
FROM ${CUDA_IMAGE}-devel-${BASE_OS} AS build

ARG VIAME_PRESET=docker-web
ARG CUDA_ARCHITECTURES="7.0 7.5 8.0 8.6 8.9 9.0 10.0 12.0"
ARG VIAME_CMAKE_ARGS=
ARG VIAME_BUILD_JOBS=
ARG VIAME_LEGACY_MMDET=OFF
ARG VIAME_KEEP_BUILD=OFF

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

USER root
WORKDIR /home

COPY . /viame

RUN VIAME_PRESET="${VIAME_PRESET}" \
    CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
    VIAME_CMAKE_ARGS="${VIAME_CMAKE_ARGS}" \
    VIAME_BUILD_JOBS="${VIAME_BUILD_JOBS}" \
    VIAME_LEGACY_MMDET="${VIAME_LEGACY_MMDET}" \
    VIAME_KEEP_BUILD="${VIAME_KEEP_BUILD}" \
    bash /viame/cmake/build_server_docker.sh

# libcupti, for the runtime image, which is not in the CUDA runtime images.
# Copied out with its symlinks resolved: /usr/local/cuda is a chain of links
# (cuda -> /etc/alternatives/cuda -> cuda-<version>), and COPY --from does not
# follow a link in a glob's directory -- the old Dockerfiles spelled out the
# CUDA version instead, so each worked for one.
RUN mkdir -p /opt/cupti \
  && cp -L /usr/local/cuda/targets/x86_64-linux/lib/libcupti.so* /opt/cupti/

ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# ---------------------------------------------------------------------------
# Directly after `build`, not at the end: the legacy builder builds every
# stage in the file up to the target, so an `everything` placed after `default`
# would also build the runtime image and download the model packs it discards.
FROM build AS everything

# ---------------------------------------------------------------------------
FROM ${CUDA_IMAGE}-runtime-${BASE_OS} AS web

# Not inherited from the build stage: without it a package that asks a
# question, tzdata's time zone, waits for an answer that never comes.
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y \
  && apt-get install -y wget unzip tar openssl curl bzip2 gfortran \
       libharfbuzz0b libxcb-shm0 libsm6 libice6 libxext6 \
       python3-pip python-is-python3 \
  && rm -rf /var/lib/apt/lists/* \
  && ( python -m pip install --break-system-packages numpy==2.0.2 \
       || python -m pip install numpy==2.0.2 )

COPY --from=build /opt/cupti/ /usr/local/cuda/lib64/
COPY --from=build /opt/noaa/viame /opt/noaa/viame

# ---------------------------------------------------------------------------
FROM web AS default

ARG VIAME_ADDONS="DEFAULT-FISH GENERIC"

RUN /opt/noaa/viame/bin/download_viame_addons.sh ${VIAME_ADDONS}
