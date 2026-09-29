#!/bin/sh

# VIAME Installation Location
export VIAME_INSTALL="$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)/../.."

# Setup VIAME Paths (no need to run multiple times if you already ran it)
source ${VIAME_INSTALL}/setup_viame.sh

# Run calibration pipeline
#
# Usage: ./calibrate_single_camera.sh calibration_images.txt [square_size]
#
# The first argument is a list of images of the calibration target, or a
# video of it. The second is the width of a checkerboard square in the units
# that lengths should be reported in (default: 80).

viame ${VIAME_INSTALL}/configs/pipelines/utility_calibrate_single_camera.pipe \
      -s input:video_filename="${1:-calibration_images.txt}" \
      -s global:square_size="${2:-80}"
