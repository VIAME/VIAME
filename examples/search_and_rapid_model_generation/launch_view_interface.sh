#!/bin/sh

# Setup VIAME Paths (no need to run multiple times if you already ran it)

export VIAME_INSTALL="$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)/../.."

source ${VIAME_INSTALL}/setup_viame.sh 

# Run vpView annotation GUI

exec sealtk \
  --pipeline-directory ${VIAME_INSTALL}/configs/pipelines/embedded_dual_stream \
  --theme ${VIAME_INSTALL}/configs/gui-params/dark_gui_settings.ini
