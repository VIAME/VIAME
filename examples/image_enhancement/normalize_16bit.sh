#!/bin/sh

# Setup VIAME Paths (no need to run multiple times if you already ran it)

export VIAME_INSTALL="$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)/../.."

source ${VIAME_INSTALL}/setup_viame.sh

# Run pipeline to normalize 16-bit imagery to 8-bit

viame ${VIAME_INSTALL}/configs/pipelines/filter_normalize_16bit.pipe \
      -s input:video_filename=input_list_16bit_images.txt
