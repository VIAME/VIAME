#!/bin/bash

# Path to VIAME installation
export VIAME_INSTALL=/opt/noaa/viame

# Setup paths and run command
source ${VIAME_INSTALL}/setup_viame.sh 

exec sealtk \
  --pipeline-directory ${VIAME_INSTALL}/configs/pipelines/embedded_dual_stream \
  --theme ${VIAME_INSTALL}/configs/gui-params/dark_gui_settings.ini
