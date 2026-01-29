#!/bin/bash

# Path to VIAME installation
export VIAME_INSTALL="$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)/../.."

# Input and output folders
export INPUT=videos
export OUTPUT=frames

# The default frame rate is only used when the csvs alongside videos
# do not contain frame rates, otherwise the CSV frame rate is used.
export DEFAULT_FRAME_RATE=5

# Set to true to output CSVs for the extracted frames (renumbered to match
# the output frame numbering). The output CSVs will be in the output folder.
export OUTPUT_CSVS=true

# Setup paths and run command
source ${VIAME_INSTALL}/setup_viame.sh

if [ "${OUTPUT_CSVS}" = "true" ]; then
  PIPELINE="pipelines/filter_tracks_only_adjust_csv.pipe"
else
  PIPELINE="pipelines/filter_tracks_only.pipe"
fi

python ${VIAME_INSTALL}/configs/process_video.py \
  -i ${INPUT} -o ${OUTPUT} -frate ${DEFAULT_FRAME_RATE} \
  -p ${PIPELINE} \
  -auto-detect-gt viame_csv
