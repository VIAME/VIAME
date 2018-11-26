#!/bin/bash

# Path to VIAME installation
export VIAME_INSTALL=/opt/noaa/viame

# Processing options
export INPUT_LIST=input_list.txt
export INPUT_FRAME_RATE=1
export PROCESS_FRAME_RATE=1

# Note: Frame rates are specified in hertz, aka frames per second. If the
# input frame rate is 1 and the process frame rate is also 1, then every
# input image in the list will be processed. If the process frame rate
# is 2, then every other image will be processed.

# Setup paths and run command
source ${VIAME_INSTALL}/setup_viame.sh

pipeline_runner -p ${VIAME_INSTALL}/configs/pipelines/tracker_use_svm_models.tut.pipe \
                -s input:video_filename=${INPUT_LIST} \
                -s input:frame_time=${INPUT_FRAME_RATE} \
                -s downsampler:target_frame_rate=${PROCESS_FRAME_RATE} 
