#!/bin/bash

export VIAME_INSTALL_DIR=./../../..
export VIAME_SCRIPT_DIR=${VIAME_INSTALL_DIR}/configs

source ${VIAME_INSTALL_DIR}/setup_viame.sh

pipeline_runner -p ${VIAME_INSTALL}/configs/pipelines/detector_use_svm_models.pipe \
                -s input:image_list_file=input_list.txt

