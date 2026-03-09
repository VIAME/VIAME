#!/bin/sh

# Setup VIAME Paths (no need to run multiple times if you already ran it)

export VIAME_INSTALL="$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)/../.."

source ${VIAME_INSTALL}/setup_viame.sh

# Run GMM motion detector
#
# The GMM (Gaussian Mixture Model) motion detector identifies moving objects
# by building a background model and detecting foreground regions. Best for
# stationary camera scenarios with moving objects.

viame ${VIAME_INSTALL}/configs/pipelines/detector_gmm_motion.pipe \
      -s input:video_filename=input_list.txt
