#!/bin/bash

# Path to VIAME installation
export VIAME_INSTALL=/opt/noaa/viame

# Setup paths and run command
source ${VIAME_INSTALL}/setup_viame.sh

# Set current directory for project folder pipe
export VIAME_PROJECT_DIR="$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)"

# Set fixed path to VIAME algorithms for DIVE
export DIVE_VIAME_INSTALL_PATH="${VIAME_INSTALL}"

exec ${VIAME_INSTALL}/dive/dive-desktop
