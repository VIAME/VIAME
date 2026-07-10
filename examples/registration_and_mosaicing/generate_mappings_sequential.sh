#!/bin/bash

# Path to VIAME installation
export VIAME_INSTALL="$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)/../.."

# Core processing options. INPUT is either a single folder of images or a
# multi-camera rig folder containing PORT/STAR/CENTER subfolders.
export INPUT=insert_foldername_here
export OUTPUT=output

# Setup paths and run command
source ${VIAME_INSTALL}/setup_viame.sh

# Sequential registration + prior coverage WITHOUT any GPS metadata.
# detect_prior_coverage.py chains affine frame-to-frame registrations from an
# anchor frame, estimates the rig-constant cross-camera transform by robust
# cluster consensus, carries a moving average of the chained motion across
# featureless open-water gaps, and pseudo-georeferences the site from the
# chains so within-site revisits are still detected. Writes
# prior_coverage.csv (polygon classes prior_coverage_sequential /
# _cross_camera / _revisit), revisits.csv, coverage_map.png and a thumbnail
# visualization into ${OUTPUT}.
python ${VIAME_INSTALL}/configs/detect_prior_coverage.py "${INPUT}" \
  --method hybrid --output "${OUTPUT}"
