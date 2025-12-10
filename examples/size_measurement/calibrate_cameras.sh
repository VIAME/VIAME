#!/bin/sh

# VIAME Installation Location
export VIAME_INSTALL="$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)/../.."

# Setup VIAME Paths (no need to run multiple times if you already ran it)
source ${VIAME_INSTALL}/setup_viame.sh

# Run calibration tool
#
# Usage Mode 1 - Stitched stereo images (left and right horizontally concatenated):
#   ./calibrate_cameras.sh calibration_video.mp4
#   ./calibrate_cameras.sh "calibration_images/*.png"
#
# Usage Mode 2 - Separate left/right images:
#   ./calibrate_cameras.sh --left left_video.mp4 --right right_video.mp4
#   ./calibrate_cameras.sh --left ./left_images/ --right ./right_images/
#   ./calibrate_cameras.sh --left "left/*.png" --right "right/*.png"
#
# Additional options: -x GRID_X -y GRID_Y -q SQUARE_SIZE_MM -s FRAME_STEP -g (gui)

python ${VIAME_INSTALL}/tools/calibrate_cameras.py -j calibration_matrices.json "$@"
