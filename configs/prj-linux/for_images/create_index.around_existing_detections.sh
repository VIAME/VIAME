#!/bin/bash

# Path to VIAME installation
export VIAME_INSTALL=/opt/noaa/viame

# Processing options
export INPUT_LIST=input_list.txt
export INPUT_DETECTIONS=input_detections.csv
export INPUT_FRAME_RATE=1
export PROCESS_FRAME_RATE=1

# Note: Frame rates are specified in hertz, aka frames per second. If the
# input frame rate is 1 and the process frame rate is also 1, then every
# input image in the list will be processed. If the process frame rate
# is 2, then every other image will be processed.

# Setup paths and run command
python ${VIAME_INSTALL}/configs/process_video.py --init \
  -l ${INPUT_LIST} -id ${INPUT_DETECTIONS} \
  -ifrate ${INPUT_FRAME_RATE} -frate ${PROCESS_FRAME_RATE} \
  -p pipelines/index_existing_detections.res.pipe --build-index --ball-tree \
  -install ${VIAME_INSTALL}
