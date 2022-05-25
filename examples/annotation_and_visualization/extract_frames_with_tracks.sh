#!/bin/bash

# Path to VIAME installation
export VIAME_INSTALL=/home/matt/Dev/viame/build/install

# Core processing options
export INPUT=videos
export OUTPUT=frames
export FRAME_RATE=5

# Setup paths and run command
source ${VIAME_INSTALL}/setup_viame.sh

python ${VIAME_INSTALL}/configs/process_video.py \
  -i ${INPUT} -o ${OUTPUT} -frate ${FRAME_RATE} \
  -p "pipelines/filter_tracks_only.pipe" \
  -auto-detect-gt viame_csv
