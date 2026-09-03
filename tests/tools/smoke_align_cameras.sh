#!/bin/bash
# This file is part of VIAME, and is distributed under an OSI-approved
# BSD 3-Clause License. See either the root top-level LICENSE file or
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.
#
# Smoke test for the utility_align_cameras pipes. Requires an installed
# VIAME (source setup_viame.sh first) with the ALIGN-CAMERAS add-on
# (minima_loftr.ckpt) present. Generates a small synthetic 2-camera image
# set (camera 2 is a known homography warp of camera 1), runs the 2-cam
# pipe, and asserts a registration JSON with a solved pair comes out.
#
# Usage: ./smoke_align_cameras.sh [work_dir]

set -euo pipefail

WORK_DIR="${1:-$(mktemp -d)}"
PIPE="${VIAME_INSTALL:?source setup_viame.sh first}/configs/pipelines/utility_align_cameras_2-cam.pipe"

if [ ! -f "$PIPE" ]; then
    echo "SKIP: $PIPE not installed (enable VIAME_DOWNLOAD_MODELS-ALIGN-CAMERAS)"
    exit 0
fi

echo "Working in $WORK_DIR"
cd "$WORK_DIR"

python - <<'EOF'
import cv2
import numpy as np

# Textured synthetic scenes; camera 2 sees the same scene through a known
# homography (mild scale + shift), one frame is blank to exercise the
# prefilter skip path.
rng = np.random.default_rng(0)
H = np.array([[0.9, 0.02, 20.0], [-0.01, 0.88, 12.0], [0.0, 0.0, 1.0]])
with open('cam1_images.txt', 'w') as list1, \
     open('cam2_images.txt', 'w') as list2:
    for k in range(6):
        if k == 3:
            scene = np.zeros((512, 640), np.uint8)  # blank: prefilter skip
        else:
            scene = rng.integers(0, 255, (64, 80), np.uint8)
            scene = cv2.resize(scene, (640, 512),
                               interpolation=cv2.INTER_CUBIC)
        warped = cv2.warpPerspective(scene, H, (640, 512))
        cv2.imwrite(f'cam1_{k:04d}.png', scene)
        cv2.imwrite(f'cam2_{k:04d}.png', warped)
        list1.write(f'cam1_{k:04d}.png\n')
        list2.write(f'cam2_{k:04d}.png\n')
EOF

viame run "$PIPE" \
    -s input1:video_filename=cam1_images.txt \
    -s input2:video_filename=cam2_images.txt \
    -s register:output_directory="$WORK_DIR" \
    -s register:max_frames=4

python - <<'EOF'
import json

with open('registration.json') as f:
    data = json.load(f)
assert data['type'] == 'dive-camera-registration', data['type']
assert data['version'] == 2, data['version']
pairs = data['pairs']
assert len(pairs) == 1, len(pairs)
assert 'leftToRight' in pairs[0], 'pooled fit did not solve'
enabled = [o for o in pairs[0]['observations'] if o['enabled']]
skipped = [o for o in pairs[0]['observations'] if not o['enabled']]
assert enabled, 'no successful observations'
assert skipped, 'expected the blank frame to be skipped'
print(f"OK: {len(enabled)} observations pooled, "
      f"{len(skipped)} skipped, rms {pairs[0]['stats']['rmsPx']} px")
EOF

echo "Smoke test passed"
