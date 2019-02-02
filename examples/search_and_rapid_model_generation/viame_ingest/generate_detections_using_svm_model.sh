#!/bin/bash

export VIAME_INSTALL=./../../..

source ${VIAME_INSTALL}/setup_viame.sh

pipeline_runner -p ${VIAME_INSTALL}/configs/pipelines/detector_svm_models.pipe \
                -s input:video_filename=input_list.txt

