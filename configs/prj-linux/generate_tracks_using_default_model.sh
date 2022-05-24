#!/bin/bash

# Path to VIAME installation
export VIAME_INSTALL=/opt/noaa/viame

# Core processing options
export INPUT=videos
export OUTPUT=output
export FRAME_RATE=5
export TRACKER_MODEL=pipelines/tracker_fish.pipe

# Extra resource utilization options
export TOTAL_GPU_COUNT=1
export PIPES_PER_GPU=1

# Setup paths and run command
source ${VIAME_INSTALL}/setup_viame.sh

python ${VIAME_INSTALL}/configs/process_video.py \
  -i ${INPUT} -o ${OUTPUT} -frate ${FRAME_RATE} \
  -p ${TRACKER_MODEL} --no-reset-prompt \
  -gpus ${TOTAL_GPU_COUNT} -pipes-per-gpu ${PIPES_PER_GPU}
