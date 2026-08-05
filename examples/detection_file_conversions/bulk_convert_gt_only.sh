#!/bin/bash

# Path to VIAME installation
export VIAME_INSTALL="$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)/../.."

# Core processing options
export INPUT_FOLDER=../object_detector_training/training_data_mouss
export INPUT_FORMAT=viame_csv
export OUTPUT_FOLDER=example_output
export OUTPUT_FORMAT=coco_json
export OUTPUT_EXTENSION=json

# Setup paths, pipeline, and run the command. Unlike bulk_convert_gt_plus_data
# this does not require the source imagery or videos to be present, and uses
# the image names or timestamps stored in the input annotation files instead.
export PIPELINE=pipelines/convert_${INPUT_FORMAT}_to_${OUTPUT_FORMAT}_gt_only.pipe

source ${VIAME_INSTALL}/setup_viame.sh

python ${VIAME_INSTALL}/configs/process_video.py \
  -i ${INPUT_FOLDER} -o ${OUTPUT_FOLDER} \
  -p ${PIPELINE} -output-ext ${OUTPUT_EXTENSION} \
  -auto-detect-gt ${INPUT_FORMAT} --gt-only --no-reset-prompt
