#!/usr/bin/env python
# This file is part of VIAME, and is distributed under an OSI-approved
# BSD 3-Clause License. See either the root top-level LICENSE file or
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.

"""
Interactive Alignment Service backend (auto-align).

A thin stdio-protocol wrapper over viame.core.alignment_core, which holds
the actual matching and fitting logic (shared with the ``align_cameras``
pipeline process). Hosted by viame.core.interactive_service; same
newline-delimited JSON protocol. The model (11.5M params, ~44 MB weights)
loads lazily on the first ``register_images`` request and stays resident
for the process lifetime.

Commands:
  register_images       {image_path_a, image_path_b, options?}
                        -> {homography (3x3, A native px -> B native px),
                            inliers [[ax, ay, bx, by], ...] (spatially spread,
                            native px), num_matches, num_inliers,
                            inlier_ratio, image_size_a, image_size_b,
                            model, elapsed_ms}
  get_alignment_status  -> {available, weights_path, loaded, device}

Failures the caller can act on are returned as ``success: False`` with a
machine-readable ``code``:
  insufficient_matches   scene has too little structure for the matcher
  low_confidence         matches found but RANSAC consensus is too weak
  degenerate_homography  a consensus exists but the fit is not a sane warp
"""

from pathlib import Path
from typing import Any, Dict, Optional

from viame.core.alignment_core import (  # noqa: F401 -- re-exported for compat
    DEFAULT_OPTIONS,
    MATCH_SIZE,
    LoftrMatcher,
    find_alignment_weights,
    register_image_pair,
)


class InteractiveAlignmentService:
    """Auto-align backend: MINIMA-LoFTR matching + MAGSAC homography."""

    def __init__(
        self,
        weights_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self._matcher = LoftrMatcher(
            weights_path=weights_path, device=device, log=self._log)

    def _log(self, message: str) -> None:
        import sys
        print(f"[InteractiveAlignment] {message}", file=sys.stderr, flush=True)

    def unload(self) -> Dict[str, Any]:
        """Drop the matcher model and release its device memory.

        The host service calls this when a competing feature (segmentation or
        stereo) is about to use its own model, so the ~44 MB matcher (plus
        whatever CUDA/MPS cache it held) yields its device memory rather than
        sitting resident alongside them. The model reloads lazily on the next
        ``register_images``. A no-op when nothing is loaded."""
        return {"success": True, "unloaded": self._matcher.unload()}

    def auto_align(
        self,
        image_path_a: str,
        image_path_b: str,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return register_image_pair(
            self._matcher, image_path_a, image_path_b, options)

    # -------------------------------------------------------------- routing
    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        command = request.get("command")
        if command == "register_images":
            for key in ("image_path_a", "image_path_b"):
                if not request.get(key):
                    raise ValueError(f"register_images requires '{key}'")
                if not Path(request[key]).exists():
                    raise ValueError(f"Image not found: {request[key]}")
            return self.auto_align(
                request["image_path_a"],
                request["image_path_b"],
                request.get("options"),
            )
        if command == "get_alignment_status":
            return self.status()
        raise ValueError(f"Unknown alignment command: {command}")

    def status(self) -> Dict[str, Any]:
        weights_path = self._matcher.weights_path
        available = bool(weights_path and Path(weights_path).exists())
        return {
            "success": True,
            "available": available,
            "weights_path": weights_path,
            "loaded": self._matcher.loaded,
            "device": self._matcher.device,
        }
