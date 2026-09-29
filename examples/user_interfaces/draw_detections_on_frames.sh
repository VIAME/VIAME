#!/bin/sh

# Setup VIAME Paths (no need to run multiple times if you already ran it)
export VIAME_INSTALL="$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)/../.."

source ${VIAME_INSTALL}/setup_viame.sh 

# Run draw pipeline
viame ${VIAME_INSTALL}/configs/pipelines/filter_draw_dets.pipe \
  -s input:video_filename=example_image_list.txt \
  -s detection_reader:file_name=example_detections.csv
