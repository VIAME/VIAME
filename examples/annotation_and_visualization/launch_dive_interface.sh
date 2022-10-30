#!/bin/sh

# Setup VIAME Paths (no need to run multiple times if you already ran it)

export VIAME_INSTALL="$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)/../.."

source ${VIAME_INSTALL}/setup_viame.sh 

# Run the DIVE GUI

export DIVE_VIAME_INSTALL_PATH="${VIAME_INSTALL}"

exec ${VIAME_INSTALL}/dive/dive-desktop
