#!/bin/bash

# Path to VIAME installation
export VIAME_INSTALL="$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)/../.."

# Input and output folders
export INPUT=videos
export OUTPUT=frames
export PIPELINE=pipelines/utility_extract_chips.pipe

# The default frame rate is only used when the csvs alongside videos
# do not contain frame rates, otherwise the CSV frame rate is used.
export DEFAULT_FRAME_RATE=5

# Setup paths and run command
source ${VIAME_INSTALL}/setup_viame.sh

python ${VIAME_INSTALL}/configs/process_video.py \
  -i ${INPUT} -o ${OUTPUT} -frate ${DEFAULT_FRAME_RATE} \
  -p ${PIPELINE} -auto-detect-gt viame_csv
