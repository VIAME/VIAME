#!/bin/bash

export VIAME_INSTALL=./../..

source ${VIAME_INSTALL}/setup_viame.sh

python ${VIAME_INSTALL}/configs/launch_timeline_interface.py
