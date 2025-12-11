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
# Additional options:
#   -a              Auto-detect grid size (recommended)
#   -x GRID_X       Number of inner corners in grid width (default: 6)
#   -y GRID_Y       Number of inner corners in grid height (default: 5)
#   -q SQUARE_SIZE  Width of calibration square in mm (default: 85)
#   -s FRAME_STEP   Process every Nth frame (default: 1)
#   -g              Show GUI with detection results

python ${VIAME_INSTALL}/tools/calibrate_cameras.py -a -o calibration_matrices.json "$@"
