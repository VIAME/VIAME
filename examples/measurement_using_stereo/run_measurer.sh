#!/bin/sh

# VIAME Installation Location
export VIAME_INSTALL=../..

# Setup VIAME Paths (no need to run multiple times if you already ran it)
source ${VIAME_INSTALL}/setup_viame.sh 

# Run pipeline
pipeline_runner -p measurement_example.pipe
