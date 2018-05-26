#!/bin/bash

export VIAME_INSTALL_DIR=~/Dev/viame/build/install
export VIAME_SCRIPT_DIR=${VIAME_INSTALL_DIR}/configs

source ${VIAME_INSTALL_DIR}/setup_viame.sh
python ${VIAME_SCRIPT_DIR}/launch_query_gui.py -qp pipelines/query_retrieval_and_iqr.res.pipe
