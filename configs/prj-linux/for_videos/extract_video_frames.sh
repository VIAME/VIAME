#!/bin/bash

export VIAME_INSTALL=/opt/noaa/viame

export VIDEO_NAME=[video_name]
export FRAME_RATE=5
export OUTPUT_DIR=images

source ${VIAME_INSTALL}/setup_viame.sh

mkdir -p ${OUTPUT_DIR}

ffmpeg -i ${VIDEO_NAME} -r ${FRAME_RATE} ${OUTPUT_DIR}/frame%06d.png
