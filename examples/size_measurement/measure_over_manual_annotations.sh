#!/bin/sh

# VIAME Installation Location
export VIAME_INSTALL="$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)/../.."

# Setup VIAME Paths (no need to run multiple times if you already ran it)
source ${VIAME_INSTALL}/setup_viame.sh

# Run pipeline
kwiver runner ${VIAME_INSTALL}/configs/pipelines/measurement_from_user_annotations.pipe
