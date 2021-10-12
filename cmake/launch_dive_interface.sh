#!/bin/bash

# Setup VIAME Paths (set path if script moved to another directory)
export VIAME_INSTALL="$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)"

source ${VIAME_INSTALL}/setup_viame.sh || exit $?

# Run the DIVE GUI
export DIVE_VIAME_INSTALL_PATH="${VIAME_INSTALL}"

exec ${VIAME_INSTALL}/dive/vue-media-annotator
