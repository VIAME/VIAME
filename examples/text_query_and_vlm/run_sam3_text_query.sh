#!/bin/sh

# Setup VIAME Paths (no need to run multiple times if you already ran it)

export VIAME_INSTALL="$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)/../.."

source ${VIAME_INSTALL}/setup_viame.sh

# Run pipeline (requires the SAM3 add-on)

viame ${VIAME_INSTALL}/configs/pipelines/tracker_sam3_animals.pipe \
      -s input:video_filename=input_list.txt \
      -s tracker:refiner:sam3:text_query="fish"
