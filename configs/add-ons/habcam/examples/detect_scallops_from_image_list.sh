#!/bin/sh

# Setup VIAME Paths (no need to run multiple times if you already ran it)

export VIAME_INSTALL="/opt/noaa/viame"
export INPUT_LIST="input_image_list.txt"
export PIPELINE="detector_habcam_scallop_one_class.pipe"

source ${VIAME_INSTALL}/setup_viame.sh 

# Run pipeline

viame ${VIAME_INSTALL}/configs/pipelines/${PIPELINE} \
      -s input:video_filename=${INPUT_LIST}
