#!/bin/sh

# Setup VIAME Paths (no need to run multiple times if you already ran it)

export VIAME_INSTALL="$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)/../.."

source ${VIAME_INSTALL}/setup_viame.sh

# Run SAM2 single-target tracker on user-initialized detections
#
# SAM2 (Segment Anything Model 2) produces high-quality segmentation masks
# and uses temporal memory for consistent tracking across frames.
#
# Prerequisites:
#   - The SAM2 add-on package must be installed
#   - A CUDA-capable GPU is required
#
# Input: A VIAME CSV file containing single-state detections (bounding boxes
# or points) to initialize tracking from. Multi-state tracks (length > 1)
# are passed through unmodified.

viame ${VIAME_INSTALL}/configs/pipelines/utility_track_selections_sam2.pipe \
      -s input:video_filename=input_list.txt
