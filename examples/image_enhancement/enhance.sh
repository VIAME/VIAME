#!/bin/sh

# Setup VIAME Paths (no need to run multiple times if you already ran it)

export VIAME_INSTALL="$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)/../.."

source ${VIAME_INSTALL}/setup_viame.sh 

# Run pipeline

viame ${VIAME_INSTALL}/configs/pipelines/filter_enhance.pipe \
      -s input:video_filename=input_list_rgb_images.txt
