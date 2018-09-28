#!/bin/bash

# Path to VIAME installation
export VIAME_INSTALL=/opt/noaa/viame

# Frame rate (in hz, or frames per second) to process data
export FRAME_RATE=5

# Setup paths and run command
source ${VIAME_INSTALL}/setup_viame.sh

python ${VIAME_INSTALL}/configs/ingest_video.py --init -v input_list.txt \
  --build-index --ball-tree -p pipelines/index_default.res.pipe \
  -install ${VIAME_INSTALL}
