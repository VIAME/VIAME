#!/bin/bash

# Path to VIAME installation
export VIAME_INSTALL=/opt/noaa/viame

# Core processing options
export INPUT_DIRECTORY=videos
export CACHE_DIRECTORY=database
export FRAME_RATE=5

# Setup paths and run command
source ${VIAME_INSTALL}/setup_viame.sh

# Set current directory for project folder pipe
export VIAME_PROJECT_DIR="$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)"

python ${VIAME_INSTALL}/configs/launch_annotation_interface.py \
  -d ${INPUT_DIRECTORY} -c ${CACHE_DIRECTORY} -frate ${FRAME_RATE} 
