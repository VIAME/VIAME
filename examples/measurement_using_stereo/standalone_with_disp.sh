#!/bin/sh

# VIAME Installation Location
export VIAME_INSTALL="$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)/../.."

# Setup VIAME Paths (no need to run multiple times if you already ran it)
source ${VIAME_INSTALL}/setup_viame.sh

# Extra path setup, in a future iteration this line will be deprecated
export SPROKIT_PYTHON_MODULES=kwiver.processes:viame.processes:camtrawl_processes

# Run standalone script
python -m viame.processes.camtrawl.demo \
       --left=../example_imagery/camtrawl_example_image_set1/left \
       --right=../example_imagery/camtrawl_example_image_set1/right \
       --cal=calibration_matrices.npz \
       --out=out --draw -f
