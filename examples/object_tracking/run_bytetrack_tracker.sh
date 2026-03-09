#!/bin/sh

# Setup VIAME Paths (no need to run multiple times if you already ran it)

export VIAME_INSTALL="$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)/../.."

source ${VIAME_INSTALL}/setup_viame.sh

# Run ByteTrack multi-target tracker with generic proposals
#
# The default tracker (common_default_tracker.pipe) uses ByteTrack. This
# script runs the generic proposal detector followed by ByteTrack tracking.
#
# ByteTrack parameters can be overridden from the command line using the
# -s flag, e.g.:
#   -s tracker:track_objects:bytetrack:high_thresh=0.5
#   -s tracker:track_objects:bytetrack:track_buffer=60

viame ${VIAME_INSTALL}/configs/pipelines/tracker_generic_proposals.pipe \
      -s input:video_filename=input_list.txt
