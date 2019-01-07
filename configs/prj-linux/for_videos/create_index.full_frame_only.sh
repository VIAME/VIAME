#!/bin/bash

# Path to VIAME installation
export VIAME_INSTALL=/opt/noaa/viame

# Core processing options
export INPUT_DIRECTORY=videos
export FRAME_RATE=5
export MAX_IMAGE_WIDTH=1400
export MAX_IMAGE_HEIGHT=1000

# Extra resource utilization options
export TOTAL_GPU_COUNT=1
export PIPES_PER_GPU=1

# Setup paths and run command
source ${VIAME_INSTALL}/setup_viame.sh

python ${VIAME_INSTALL}/configs/process_video.py --init \
  -d ${INPUT_DIRECTORY} -frate ${FRAME_RATE} \
  -p pipelines/index_full_frame.res.pipe \
  -gpus ${TOTAL_GPU_COUNT} -pipes-per-gpu ${PIPES_PER_GPU} \
  -archive-width ${MAX_IMAGE_WIDTH} -archive-height ${MAX_IMAGE_HEIGHT} \
  --build-index --ball-tree -install ${VIAME_INSTALL}
