#!/bin/bash

# Convert every annotation file under a folder into another format.
#
# The convert tool recognises each annotation file from its extension and
# content, and uses any imagery found next to it (images in the same folder,
# or a video) for the frame names, frame count and timing of the output. Run
# "viame convert --list-formats" for the readers and writers available.

export VIAME_INSTALL="$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)/../.."

export INPUT_FOLDER=../object_detector_training/training_data_mouss
export OUTPUT_FOLDER=example_output
export OUTPUT_FORMAT=coco
export DEFAULT_FRAME_RATE=5

source ${VIAME_INSTALL}/setup_viame.sh

viame convert ${INPUT_FOLDER} ${OUTPUT_FOLDER} \
  -o ${OUTPUT_FORMAT} --frame-rate ${DEFAULT_FRAME_RATE}
