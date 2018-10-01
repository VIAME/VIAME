#!/bin/bash

# Path to VIAME installation
export VIAME_INSTALL=/opt/noaa/viame

# Processing options
export VIDEO_NAME=[video_name]
export FRAME_RATE=5
export START_TIME=00:00:00.00
export CLIP_DURATION=00:05:00.00
export OUTPUT_DIR=images

# Setup paths and run command
source ${VIAME_INSTALL}/setup_viame.sh

mkdir -p ${OUTPUT_DIR}

ffmpeg -i ${VIDEO_NAME} -r ${FRAME_RATE} \
  -ss ${START_TIME} -t ${CLIP_DURATION} \
  ${OUTPUT_DIR}/frame%06d.png
