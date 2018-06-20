#!/bin/bash

export VIAME_INSTALL_DIR=./../..

source ${VIAME_INSTALL_DIR}/setup_viame.sh

python ${VIAME_INSTALL_DIR}/configs/launch_track_viewer.py
