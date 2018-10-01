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
  -d ${INPUT_DIRECTORY} -frate ${FRAME_RATE} \
  -p pipelines/index_full_frame.res.pipe \
  -archive-width ${MAX_IMAGE_WIDTH} -archive-height ${MAX_IMAGE_HEIGHT} \
  --build-index --ball-tree \
  -install ${VIAME_INSTALL}
