#!/bin/sh

# VIAME Installation Location
export VIAME_INSTALL="$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)/../.."

# Setup VIAME Paths (no need to run multiple times if you already ran it)
source ${VIAME_INSTALL}/setup_viame.sh

# Extra path setup, in a future iteration this line will be deprecated
export SPROKIT_PYTHON_MODULES=kwiver.processes:viame.processes:camtrawl_processes

# Run pipeline
kwiver runner ${VIAME_INSTALL}/configs/pipelines/measurement_gmm_only.pipe
