#!/bin/bash

# Path to VIAME installation
export VIAME_INSTALL=/opt/noaa/viame

# Setup paths and run command
source ${VIAME_INSTALL}/setup_viame.sh

export DIVE_VIAME_INSTALL_PATH="${VIAME_INSTALL}"

exec ${VIAME_INSTALL}/dive/vue-media-annotator
