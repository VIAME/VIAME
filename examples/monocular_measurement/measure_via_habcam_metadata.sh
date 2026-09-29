#!/bin/sh

# VIAME Installation Location
export VIAME_INSTALL="$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)/../.."

# Setup VIAME Paths (no need to run multiple times if you already ran it)
source ${VIAME_INSTALL}/setup_viame.sh

# Run pipeline (requires the HabCam add-on)
viame ${VIAME_INSTALL}/configs/pipelines/detector_habcam_measure_scallops_one_class_metadata.pipe \
      -s input:video_filename=input_image_list_habcam.txt
