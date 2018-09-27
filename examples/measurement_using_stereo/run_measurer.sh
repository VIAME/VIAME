#!/bin/sh

# VIAME Installation Location
export VIAME_INSTALL=../..

# Setup VIAME Paths (no need to run multiple times if you already ran it)
source ${VIAME_INSTALL}/setup_viame.sh

# Extra path setup, in a future iteration this line will be deprecated
export SPROKIT_PYTHON_MODULES=kwiver.processes:viame.processes:camtrawl_processes

# Run pipeline
pipeline_runner -p ${VIAME_INSTALL}/configs/pipelines/measurement_default.gmm.tut.pipe
