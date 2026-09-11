#!/bin/bash
# Train keypoints from detection boxes and named (kp) attributes.
VIAME_INSTALL="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${VIAME_INSTALL}/setup_viame.sh"
viame train -i "${1:-training_data}" \
  -c "${VIAME_INSTALL}/configs/pipelines/train_reclassifier_sleap_head_tail.conf" \
  --threshold 0.0
