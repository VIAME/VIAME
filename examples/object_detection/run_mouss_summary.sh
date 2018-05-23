#!/bin/sh

# Setup VIAME Paths (no need to run multiple times if you already ran it)

export VIAME_INSTALL_DIR=./../..
export VIAME_SCRIPT_DIR=${VIAME_INSTALL_DIR}/configs

source ${VIAME_INSTALL_DIR}/setup_viame.sh

rm -rf database
mkdir database

python ${VIAME_SCRIPT_DIR}/ingest_video.py -d $1 \
  --detection-plots -species fish -threshold 0.25 -frate 1
