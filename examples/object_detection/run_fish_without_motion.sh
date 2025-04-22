#!/bin/sh

# Setup VIAME Paths (no need to run multiple times if you already ran it)

export VIAME_INSTALL="$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)/../.."

source ${VIAME_INSTALL}/setup_viame.sh 

# Run pipeline

kwiver runner ${VIAME_INSTALL}/configs/pipelines/detector_default_fish.pipe \
              -s input:video_filename=input_image_list_small_set.txt
