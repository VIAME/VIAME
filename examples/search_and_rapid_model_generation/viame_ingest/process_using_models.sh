#!/bin/bash

export VIAME_INSTALL=./../../..

source ${VIAME_INSTALL}/setup_viame.sh

pipeline_runner -p ${VIAME_INSTALL}/configs/pipelines/detector_use_svm_models.pipe \
                -s input:image_list_file=input_list.txt

