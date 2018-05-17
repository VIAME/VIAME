#!/bin/bash

export VCATInstallDir=/opt/kitware/video-cat
export VCATScriptDir=${VCATInstallDir}/run_scripts

source ${VCATInstallDir}/setup_viame.sh

python ${VCATScriptDir}/ingest_video.py --init -d input_videos --build-index
