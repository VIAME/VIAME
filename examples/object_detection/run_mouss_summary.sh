#!/bin/sh

# Setup VIAME Paths (no need to run multiple times if you already ran it)

export VIAME_INSTALL=./../..

source ${VIAME_INSTALL}/setup_viame.sh

rm -rf database
mkdir database

python ${VIAME_INSTALL}/configs/ingest_video.py -d $1 \
  --detection-plots -species fish -threshold 0.25 -frate 1
