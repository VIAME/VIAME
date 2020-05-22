#!/bin/sh

# Setup VIAME Paths (set path if script moved to another directory)

export VIAME_INSTALL="$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)/../.."

source ${VIAME_INSTALL}/setup_viame.sh

# Launch the GUI
exec sealtk \
  --pipeline-directory ${VIAME_INSTALL}/configs/pipelines/embedded_dual_stream \
  --theme ${VIAME_INSTALL}/configs/gui-params/dark_gui_settings.ini
