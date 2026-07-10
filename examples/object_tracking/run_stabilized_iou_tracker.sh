#!/bin/sh

# Setup VIAME Paths (no need to run multiple times if you already ran it)

export VIAME_INSTALL="$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)/../.."

source ${VIAME_INSTALL}/setup_viame.sh

# Run stabilized IOU tracker for moving camera scenarios
#
# This tracker uses feature-based image registration (SURF + homography) to
# compensate for camera motion, then links detections using IOU in the
# stabilized coordinate frame. Best for aerial imagery or benthic tow cameras.
#
# Note: Requires a domain-specific add-on that includes a detector and the
# stabilized IOU tracker pipeline (e.g. default-fish add-on provides
# tracker_stabilized_iou.pipe). The common_stabilized_iou_tracker.pipe
# component is available in all installations.

viame ${VIAME_INSTALL}/configs/pipelines/tracker_stabilized_iou.pipe \
      -s input:video_filename=input_list.txt
