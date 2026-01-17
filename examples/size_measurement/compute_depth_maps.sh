#!/bin/sh

# VIAME Installation Location
export VIAME_INSTALL="$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)/../.."

# Setup VIAME Paths (no need to run multiple times if you already ran it)
source ${VIAME_INSTALL}/setup_viame.sh

# Run pipeline
viame ${VIAME_INSTALL}/configs/pipelines/filter_stereo_depth_map.pipe \
      -s input:video_filename=input_list.txt
