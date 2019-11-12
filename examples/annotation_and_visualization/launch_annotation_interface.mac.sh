#!/bin/sh

# Setup VIAME Paths (no need to run multiple times if you already ran it)

export VIAME_INSTALL=./../..

source ${VIAME_INSTALL}/setup_viame.sh 

# Run vpView annotation GUI

${VIAME_INSTALL}/vpView.app/Contents/MacOS/vpView
