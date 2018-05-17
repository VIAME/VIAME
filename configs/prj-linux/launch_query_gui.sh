#!/bin/bash

export VCATInstallDir=/opt/kitware/video-cat
export VCATScriptDir=${VCATInstallDir}/run_scripts

source ${VCATInstallDir}/setup_viame.sh
python ${VCATScriptDir}/launch_query_gui.py
