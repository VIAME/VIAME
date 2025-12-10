#!/bin/sh

# VIAME Installation Location
export VIAME_INSTALL="$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)/../.."

# Setup VIAME Paths (no need to run multiple times if you already ran it)
source ${VIAME_INSTALL}/setup_viame.sh

# Run calibration tool
# Usage: Provide a video file or image glob pattern as the first argument
# Example: ./calibrate_cameras.sh calibration_video.mp4
# Example: ./calibrate_cameras.sh "calibration_images/*.png"

python ${VIAME_INSTALL}/tools/calibrate_cameras.py -j calibration_matrices.json "$@"
