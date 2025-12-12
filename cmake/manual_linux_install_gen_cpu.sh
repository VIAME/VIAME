#!/bin/bash

# clean build env
docker stop viame_installer_zip || true && docker rm --force viame_installer_zip || true
rm -rf viame-src-clone || true

git clone https://github.com/VIAME/VIAME.git viame-src-clone
cd viame-src-clone

# Extract version from RELEASE_NOTES.md (first token of first line)
VIAME_VERSION=$(head -n 1 RELEASE_NOTES.md | awk '{print $1}')

# stand up a new docker build env
docker pull nvidia/cuda:12.6.3-cudnn-devel-rockylinux8
chmod +x cmake/build_server_rocky.sh
docker run -td --runtime=nvidia --name viame_installer_zip nvidia/cuda:12.6.3-cudnn-devel-rockylinux8 bash
cd ../
docker cp viame-src-clone viame_installer_zip:/viame/

# run the build script in the fresh docker environment
docker exec -i viame_installer_zip ./viame/cmake/build_server_rocky_cpu.sh

# copy out final installer and build log
docker cp viame_installer_zip:/viame/build/VIAME-CPU-${VIAME_VERSION}-Linux-64Bit.tar.gz .
docker cp viame_installer_zip:/viame/build/build_log.txt .
