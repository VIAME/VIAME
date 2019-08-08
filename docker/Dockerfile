## === === === === ===  Base image with lined deps === === === === ===
ARG BASE_IMAGE=nvidia/cuda:9.1-cudnn7-devel-ubuntu16.04
FROM ${BASE_IMAGE} as base

ARG DEBIAN_FRONTEND=noninteractive

# System Deps for VIAME
# Fletch system dependencies
RUN apt-get update -qq &&\
    apt-get install -qq \
    curl \
    &&\
    apt-get install -qq --no-install-recommends \
        apt-transport-https \
        ca-certificates \
        git \
        openssl \
        software-properties-common \
        wget \
        zip unzip \
        libgl1-mesa-dev \
        libexpat1-dev \
        libgtk2.0-dev \
        libxt-dev \
        libxml2-dev \
        libssl-dev \
        liblapack-dev \
        python3-dev \
        zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

## === === === === ===  Compile-time dependencies === === === === ===
FROM base as compiler

# Need CMake >= 3.11.4 for VIAME,
# Thus, install latest stable CMake 3.14.1
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | apt-key add - &&\
    apt-add-repository 'deb https://apt.kitware.com/ubuntu/ xenial main' &&\
    apt-get update -qq &&\
    apt-get install -qq \
        cmake \
        python3-setuptools \
    &&\
    apt-get install -qq --no-install-recommends \
        build-essential\
        ninja-build \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

## === === === === ===  Building VIAME Deps === === === === ===
FROM compiler as viame_repo

# Get VIAME
WORKDIR /root/viame
ARG VIAME_VARIANT=default
ENV VIAME_VARIANT=$VIAME_VARIANT

# We have to be rather precise here because there are multiple possible origins
# normally one would just git pull origin
# todo: more granular dependencies
COPY . .
RUN git config --global user.email "dockerfile@kitware.com" &&\
    git config --global user.name "Dockerfile dummy git" &&\
    git pull https://github.com/VIAME/VIAME.git --recurse-submodules &&\
    git submodule update --init --recursive

## == == == == == == == ==  Building Fletch == == == == == == == ==
RUN cd /root/viame/packages/fletch &&\
    mkdir -p build && \
    cd build &&\
    cmake \
        -Dfletch_ENABLE_ALL_PACKAGES=OFF \
        -G Ninja \
        .. &&\
    ninja

## === === === === ===  Building VIAME in full === === === === ===
FROM viame_repo as viame

# build the full system
RUN mkdir -p build && \
    export FLAGS="$(./configs/parse_flags.sh \
        ./configs/build-flags/common.txt \
        ./configs/build-flags/${VIAME_VARIANT}.txt )" &&\
    printf "Variant ${VIAME_VARIANT}\nCMAKE flags: \n$(echo ${FLAGS} | tr ' ' '\n' )" &&\
    cd build &&\
    cmake "${FLAGS}" \
        -G Ninja \
        .. &&\
    ninja

## === === === === ===  Copy in build artifacts === === === === ===

FROM base

COPY --from=viame /root/viame/build/install /root/viame
WORKDIR /root/viame

