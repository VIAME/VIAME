#!/bin/bash

export VCATInstallDir=/opt/kitware/video-cat
export VCATScriptDir=${VCATInstallDir}/run_scripts

source ${VCATInstallDir}/setup_viame.sh

python ${VCATScriptDir}/database_tool.py start
python ${VCATScriptDir}/ingest_video.py --build-index
