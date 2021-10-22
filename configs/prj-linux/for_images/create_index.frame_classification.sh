#!/bin/bash

# Path to VIAME installation
export VIAME_INSTALL=/opt/noaa/viame

# Core processing options
export INPUT_LIST=input_list.txt

# Note: Frame rates are specified in hertz, aka frames per second. If the
# input frame rate is 1 and the process frame rate is also 1, then every
# input image in the list will be processed. If the process frame rate
# is changed to 0.5, then every other image will be processed.
export INPUT_FRAME_RATE=1
export PROCESS_FRAME_RATE=1

# Extra resource utilization options
export TOTAL_GPU_COUNT=1
export PIPES_PER_GPU=1

# Setup paths and run command
source ${VIAME_INSTALL}/setup_viame.sh

python ${VIAME_INSTALL}/configs/process_video.py --init \
  -l ${INPUT_LIST} -ifrate ${INPUT_FRAME_RATE} -frate ${PROCESS_FRAME_RATE} \
  -p pipelines/index_frame.pipe -o database \
  -gpus ${TOTAL_GPU_COUNT} -pipes-per-gpu ${PIPES_PER_GPU} \
  --build-index -install ${VIAME_INSTALL}
