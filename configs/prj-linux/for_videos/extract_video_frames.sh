#!/bin/bash

# Path to VIAME installation
export VIAME_INSTALL=/opt/noaa/viame

# Core processing options
export INPUT_DIRECTORY=videos
export FRAME_RATE=5
export START_TIME=00:00:00.00
export DURATION=99:99:99.99
export OUTPUT_DIR=frames
export METHOD=kwiver

# Setup paths and run command
source ${VIAME_INSTALL}/setup_viame.sh

python ${VIAME_INSTALL}/configs/extract_video_frames.py \
  -d ${INPUT_DIRECTORY} -o ${OUTPUT_DIR} \
  -r ${FRAME_RATE} -s ${START_TIME} -t ${DURATION}
