#!/bin/bash

# Path to VIAME installation
export VIAME_INSTALL=/opt/noaa/viame

# Core processing options
export INPUT=videos
export OUTPUT=output
export FRAME_RATE=5

# Extra resource utilization options
export TOTAL_GPU_COUNT=1
export PIPES_PER_GPU=1

# Setup paths and run command
source ${VIAME_INSTALL}/setup_viame.sh

# Set current directory for project folder pipe
export VIAME_PROJECT_DIR="$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)"

python ${VIAME_INSTALL}/configs/process_video.py \
  -i ${INPUT} -o ${OUTPUT} -frate ${FRAME_RATE} \
  -p pipelines/tracker_project_folder.pipe --no-reset-prompt \
  -gpus ${TOTAL_GPU_COUNT} -pipes-per-gpu ${PIPES_PER_GPU}
