#!/bin/bash

# Path to VIAME installation
export VIAME_INSTALL=/opt/noaa/viame

# Core processing options - 
export INPUT_DIRECTORY=videos
export DEFAULT_FRAME_RATE=5
export MAX_DURATION=05:00.00
export OUTPUT_DIR=video_clips

# Setup paths and run command
source ${VIAME_INSTALL}/setup_viame.sh

python ${VIAME_INSTALL}/configs/extract_video_frames.py \
  -p "pipelines/transcode_default.pipe" \
  -d ${INPUT_DIRECTORY} -o ${OUTPUT_DIR} \
  -r ${DEFAULT_FRAME_RATE} \
  -s "output:maximum_length="${MAX_DURATION}
