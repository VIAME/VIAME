#!/bin/bash

# Path to VIAME installation
export VIAME_INSTALL=/opt/noaa/viame

# Setup paths and run command
source ${VIAME_INSTALL}/setup_viame.sh

kwiver runner ${VIAME_INSTALL}/configs/pipelines/database_apply_svm_models.pipe \
              -s reader:reader:db:video_name=input_list \
              -s descriptors:video_name=input_list
