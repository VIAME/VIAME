#!/bin/bash

# Path to VIAME installation
export VIAME_INSTALL=/opt/noaa/viame

# Core processing options
export INPUT=videos
export OUTPUT=video_clips
export FRAME_RATE=5
export MAX_DURATION=05:00.00

# Setup paths and run command
source ${VIAME_INSTALL}/setup_viame.sh

viame run \
  -i ${INPUT} -o ${OUTPUT} -frate ${FRAME_RATE} \
  -p "pipelines/transcode_default.pipe" \
  -s "video_writer:maximum_length="${MAX_DURATION}
