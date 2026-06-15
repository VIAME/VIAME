#!/bin/bash

# Path to VIAME installation
export VIAME_INSTALL="$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)/../.."

# Core processing options. INPUT is either a single folder of images or a
# multi-camera rig folder containing PORT/STAR/CENTER subfolders.
export INPUT=insert_foldername_here
export OUTPUT=output

# Setup paths and run command
source ${VIAME_INSTALL}/setup_viame.sh

# Sequential homography registration + coverage with the recommended settings.
# Frame-to-frame homographies are chained from an anchor frame (no global bundle
# adjustment); the cross-camera transform is a robust per-rig consensus. These
# settings work for BOTH land-heavy and water-heavy scenes:
#   --affine          constrained 6-DOF model; rejects the false perspective
#                     warps that repetitive water texture otherwise produces
#   --xcam-robust     mode-seeking (cluster) cross-camera consensus instead of a
#                     median, which is corrupted by chain drift over water
#   --xcam-low-drift  picks cross-camera pairs nearest both chain anchors so the
#                     consensus is corroborated, not a single unverified pair
#   --consistency-filter  validate water-frame registrations against land motion
python ${VIAME_INSTALL}/configs/reconstruct_3d.py "${INPUT}" \
  --output "${OUTPUT}" \
  --planar --coverage-class suppressed --visualize \
  --affine --consistency-filter --xcam-robust --xcam-low-drift
