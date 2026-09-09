#!/bin/bash

# Convert every annotation file under a folder into another format using only
# the annotation files themselves: any imagery next to them is ignored, and the
# frame names and numbers stored in the annotations carry through.

export VIAME_INSTALL="$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)/../.."

export INPUT_FOLDER=../object_detector_training/training_data_mouss
export OUTPUT_FOLDER=example_output
export OUTPUT_FORMAT=coco

source ${VIAME_INSTALL}/setup_viame.sh

viame convert ${INPUT_FOLDER} ${OUTPUT_FOLDER} \
  -o ${OUTPUT_FORMAT} --no-images
