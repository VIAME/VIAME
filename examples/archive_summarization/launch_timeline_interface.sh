#!/bin/bash

export VIAME_INSTALL="$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)/../.."

source ${VIAME_INSTALL}/setup_viame.sh

python ${VIAME_INSTALL}/configs/launch_timeline_interface.py
