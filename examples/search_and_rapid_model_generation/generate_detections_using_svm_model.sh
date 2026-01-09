#!/bin/bash

export VIAME_INSTALL="$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)/../.."

source ${VIAME_INSTALL}/setup_viame.sh

kwiver runner ${VIAME_INSTALL}/configs/pipelines/detector_svm_models.pipe \
              -s input:video_filename=ingest_list.txt

