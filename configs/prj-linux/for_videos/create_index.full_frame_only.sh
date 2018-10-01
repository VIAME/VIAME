#!/bin/bash

# Path to VIAME installation
export VIAME_INSTALL=/opt/noaa/viame

# Processing options
export INPUT_DIRECTORY=videos
export FRAME_RATE=5
export MAX_IMAGE_WIDTH=1400
export MAX_IMAGE_HEIGHT=1000

# Setup paths and run command
source ${VIAME_INSTALL}/setup_viame.sh

python ${VIAME_INSTALL}/configs/ingest_video.py --init \
  -d ${INPUT_DIRECTORY} \
  -p pipelines/index_full_frame.res.pipe \
  --build-index --ball-tree -frate ${FRAME_RATE} \
  -archive-width ${MAX_IMAGE_WIDTH} -archive-height ${MAX_IMAGE_HEIGHT} \
  -install ${VIAME_INSTALL}
