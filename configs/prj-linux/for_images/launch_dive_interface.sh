#!/bin/bash

# Path to VIAME installation
export VIAME_INSTALL=/opt/noaa/viame

# Setup paths and run command
source ${VIAME_INSTALL}/setup_viame.sh 

exec ${VIAME_INSTALL}/bin/dive/vue-media-annotator
