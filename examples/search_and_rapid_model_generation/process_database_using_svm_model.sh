#!/bin/bash

export VIAME_INSTALL="$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)/../.."

source ${VIAME_INSTALL}/setup_viame.sh

kwiver runner ${VIAME_INSTALL}/configs/pipelines/database_apply_svm_models.pipe \
              -s reader:reader:db:video_name=input_list \
              -s descriptors:video_name=input_list
                

