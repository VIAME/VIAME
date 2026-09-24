#!/bin/bash

# Path to VIAME installation
export VIAME_INSTALL="$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)/../.."

# Core processing options
export INPUT_LIST=input_list.txt
export OUTPUT_DIRECTORY=output
export INPUT_FRAME_RATE=1
export PROCESS_FRAME_RATE=1

# Note: Frame rates are specified in hertz, aka frames per second. If the
# input frame rate is 1 and the process frame rate is also 1, then every
# input image in the list will be processed. If the process frame rate
# is changed to 0.5, then every other image will be processed.

# Extra resource utilization options
export TOTAL_GPU_COUNT=1
export PIPES_PER_GPU=1

# Trained model: the pack written by training, or an unpacked folder
if [ -f trained_model.zip ]; then
  export TRAINED_MODEL=trained_model.zip/detector.pipe
elif [ -d trained_model ]; then
  export TRAINED_MODEL=trained_model/detector.pipe
else
  export TRAINED_MODEL=category_models/detector.pipe
fi

# Setup paths and run command
source ${VIAME_INSTALL}/setup_viame.sh

# Set current directory for project folder pipe
export VIAME_PROJECT_DIR="$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)"

viame run \
  -l ${INPUT_LIST} -ifrate ${INPUT_FRAME_RATE} -frate ${PROCESS_FRAME_RATE} \
  -p ${TRAINED_MODEL} -o ${OUTPUT_DIRECTORY} --no-reset-prompt \
  -gpus ${TOTAL_GPU_COUNT} -pipes-per-gpu ${PIPES_PER_GPU}
