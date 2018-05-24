#!/bin/bash

export VIAME_INSTALL_DIR=/opt/noaa/viame
export VIAME_SCRIPT_DIR=${VIAME_INSTALL_DIR}/configs

source ${VIAME_INSTALL_DIR}/setup_viame.sh

python ${VIAME_SCRIPT_DIR}/ingest_video.py --init -d input_videos --build-index \
  -frate 2 -fbatch 50 -fskip 200
