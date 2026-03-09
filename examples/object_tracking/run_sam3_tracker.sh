#!/bin/sh

# Setup VIAME Paths (no need to run multiple times if you already ran it)

export VIAME_INSTALL="$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)/../.."

source ${VIAME_INSTALL}/setup_viame.sh

# Run SAM3 single-target tracker on user-initialized detections
#
# SAM3 (Segment Anything Model 3) extends SAM2 with Grounding DINO support
# for text-based object queries. It produces high-quality segmentation masks
# with temporal tracking memory.
#
# Prerequisites:
#   - The SAM3 add-on package must be installed
#   - A CUDA-capable GPU is required
#
# Input: A VIAME CSV file containing single-state detections (bounding boxes
# or points) to initialize tracking from. Multi-state tracks (length > 1)
# are passed through unmodified.

viame ${VIAME_INSTALL}/configs/pipelines/utility_track_selections_sam3.pipe \
      -s input:video_filename=input_list.txt
