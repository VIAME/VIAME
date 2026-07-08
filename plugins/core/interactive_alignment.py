#!/usr/bin/env python
# This file is part of VIAME, and is distributed under an OSI-approved
# BSD 3-Clause License. See either the root top-level LICENSE file or
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.

"""
Interactive Alignment Service backend (auto-align).

Computes a homography between two camera images of the same scene — typically
different modalities (EO/RGB vs thermal IR) — using the vendored MINIMA-LoFTR
matcher (viame.pytorch.minima_loftr), a LoFTR fine-tuned on multimodal data
that is robust to cross-spectral appearance changes.

Hosted by viame.core.interactive_service; same newline-delimited JSON
protocol. The model (11.5M params, ~44 MB weights) loads lazily on the first
``auto_align`` request and stays resident for the process lifetime.

Commands:
  auto_align            {image_path_a, image_path_b, options?}
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

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Matcher input size: long side after resize, dims floored to /8 (LoFTR df).
MATCH_SIZE = 640
# 16-bit (or generally high-dynamic-range single-channel) inputs are
# percentile-normalized so low-contrast IR frames still spread over 8 bits.
NORM_PERCENTILES = (2.0, 98.0)

DEFAULT_OPTIONS = {
    # RANSAC reprojection threshold in *matcher-resolution* pixels (<=640).
    "ransac_threshold": 2.0,
    # Quality gate: reject before/after RANSAC.
    "min_matches": 100,
    "min_inliers": 30,
    "min_inlier_ratio": 0.15,
    # How many spatially-spread inlier correspondences to return.
    "top_k": 24,
    # LoFTR coarse match confidence threshold.
    "match_threshold": 0.2,
}


def find_alignment_weights() -> Optional[str]:
    """Locate minima_loftr.ckpt in the VIAME install (or override via env)."""
    override = os.environ.get("VIAME_ALIGNMENT_WEIGHTS")
    if override and Path(override).exists():
        return override
    viame_install = os.environ.get("VIAME_INSTALL")
    if viame_install:
        candidate = (
            Path(viame_install) / "configs" / "pipelines" / "models"
            / "minima_loftr.ckpt"
        )
        if candidate.exists():
            return str(candidate)
    return None


def _resolve_device(device: Optional[str]) -> str:
    import torch

    if device in (None, "", "auto"):
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None \
                and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return device


def _load_gray_norm(path: str) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Load any 8/16-bit gray/color image as normalized 8-bit grayscale.

    Returns (gray uint8 HxW, (native_width, native_height)).
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    h, w = img.shape[:2]
    if img.ndim == 3:
        # Color inputs are standard 8-bit imagery (BGR from cv2).
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX) \
                .astype(np.uint8)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif img.dtype == np.uint8:
        gray = img
    else:
        # Single-channel high-bit-depth (16-bit IR): percentile normalize so
        # frame-to-frame dynamic range differences don't crush contrast.
        arr = img.astype(np.float32)
        lo, hi = np.percentile(arr, NORM_PERCENTILES)
        arr = np.clip((arr - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        gray = (arr * 255.0).astype(np.uint8)
    return gray, (w, h)


def _spread_inliers(
    kpts_a: np.ndarray,
    kpts_b: np.ndarray,
    conf: np.ndarray,
    mask: np.ndarray,
    size_a: Tuple[int, int],
    top_k: int,
    grid: int = 6,
) -> List[List[float]]:
    """Pick up to top_k inliers spatially spread over image A (grid buckets),
    preferring higher-confidence matches within each bucket."""
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return []
    idx = idx[np.argsort(-conf[idx])]  # best confidence first
    w, h = size_a
    picked: List[int] = []
    seen = set()
    for i in idx:
        cell = (
            min(int(kpts_a[i, 0] / w * grid), grid - 1),
            min(int(kpts_a[i, 1] / h * grid), grid - 1),
        )
        if cell not in seen:
            seen.add(cell)
            picked.append(int(i))
        if len(picked) >= top_k:
            break
    for i in idx:  # top up if the grid didn't fill top_k
        if len(picked) >= top_k:
            break
        if int(i) not in picked:
            picked.append(int(i))
    return [
        [float(kpts_a[i, 0]), float(kpts_a[i, 1]),
         float(kpts_b[i, 0]), float(kpts_b[i, 1])]
        for i in picked
    ]


def _homography_is_sane(
    H: np.ndarray, size_a: Tuple[int, int], size_b: Tuple[int, int]
) -> bool:
    """Reject non-finite / non-invertible / wildly-warping homographies."""
    if H is None or not np.all(np.isfinite(H)):
        return False
    if abs(np.linalg.det(H)) < 1e-12:
        return False
    w, h = size_a
    corners = np.array(
        [[0, 0], [w, 0], [w, h], [0, h]], np.float64
    ).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(corners, H).reshape(4, 2)
    if not np.all(np.isfinite(warped)):
        return False
    # Warped corners must remain a convex quad with consistent winding.
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    signs = [
        np.sign(cross(warped[i], warped[(i + 1) % 4], warped[(i + 2) % 4]))
        for i in range(4)
    ]
    if len(set(signs)) != 1 or signs[0] == 0:
        return False
    # Area must be within a plausible range of the target image's area.
    area = 0.5 * abs(sum(
        warped[i][0] * warped[(i + 1) % 4][1]
        - warped[(i + 1) % 4][0] * warped[i][1]
        for i in range(4)
    ))
    wb, hb = size_b
    ratio = area / float(wb * hb)
    return 1e-2 < ratio < 1e2


class InteractiveAlignmentService:
    """Auto-align backend: MINIMA-LoFTR matching + MAGSAC homography."""

    def __init__(
        self,
        weights_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self._weights_path = weights_path or find_alignment_weights()
        self._requested_device = device
        self._model = None
        self._device: Optional[str] = None

    # ---------------------------------------------------------------- model
    def _log(self, message: str) -> None:
        import sys
        print(f"[InteractiveAlignment] {message}", file=sys.stderr, flush=True)

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        if not self._weights_path or not Path(self._weights_path).exists():
            raise ValueError(
                "Alignment model weights not found (minima_loftr.ckpt); "
                "expected in $VIAME_INSTALL/configs/pipelines/models or via "
                "$VIAME_ALIGNMENT_WEIGHTS")
        import torch
        from copy import deepcopy
        from viame.pytorch.minima_loftr import LoFTR, default_cfg

        self._device = _resolve_device(self._requested_device)
        self._log(
            f"Loading MINIMA-LoFTR ({self._weights_path}) on {self._device}...")
        start = time.time()
        config = deepcopy(default_cfg)
        state = torch.load(
            self._weights_path, map_location="cpu", weights_only=False)
        model = LoFTR(config=config)
        model.load_state_dict(state["state_dict"], strict=True)
        self._model = model.eval().to(self._device)
        self._log(f"Alignment model ready in {time.time() - start:.1f}s")
        return self._model

    # ------------------------------------------------------------- matching
    @staticmethod
    def _to_match_tensor(gray: np.ndarray, device: str):
        """Resize long side to MATCH_SIZE (dims /8) -> [1,1,H,W] in [0,1].

        Returns (tensor, (sx, sy)) where s maps matcher coords -> native."""
        import torch

        h, w = gray.shape
        scale = MATCH_SIZE / max(h, w)
        nh = max(8, int(h * scale) // 8 * 8)
        nw = max(8, int(w * scale) // 8 * 8)
        resized = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(resized)[None, None].float().div(255.0)
        return tensor.to(device), (w / nw, h / nh)

    def auto_align(
        self,
        image_path_a: str,
        image_path_b: str,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        import torch

        opts = dict(DEFAULT_OPTIONS)
        opts.update(options or {})
        start = time.time()

        model = self._ensure_model()
        # CoarseMatching reads thr at forward time from its own attribute.
        model.coarse_matching.thr = float(opts["match_threshold"])

        device = self._device or "cpu"
        gray_a, size_a = _load_gray_norm(image_path_a)
        gray_b, size_b = _load_gray_norm(image_path_b)
        img_a, scale_a = self._to_match_tensor(gray_a, device)
        img_b, scale_b = self._to_match_tensor(gray_b, device)

        batch = {"image0": img_a, "image1": img_b}
        with torch.no_grad():
            model(batch)
        kpts_a = batch["mkpts0_f"].cpu().numpy()  # matcher coords, image A
        kpts_b = batch["mkpts1_f"].cpu().numpy()
        conf = batch["mconf"].cpu().numpy()
        num_matches = len(kpts_a)

        def failure(code: str, message: str, **extra) -> Dict[str, Any]:
            result = {
                "success": False,
                "code": code,
                "error": message,
                "num_matches": num_matches,
                "model": "minima_loftr",
                "elapsed_ms": int((time.time() - start) * 1000),
            }
            result.update(extra)
            return result

        if num_matches < int(opts["min_matches"]):
            return failure(
                "insufficient_matches",
                f"Only {num_matches} matches found (need "
                f">={opts['min_matches']}); the scene may lack distinctive "
                "structure — try a frame with more visual features.")

        # RANSAC at matcher resolution (both images <= MATCH_SIZE) so the
        # pixel threshold means the same thing regardless of native sizes.
        cv2.setRNGSeed(0)
        H_match, mask = cv2.findHomography(
            kpts_a.astype(np.float64),
            kpts_b.astype(np.float64),
            cv2.USAC_MAGSAC,
            ransacReprojThreshold=float(opts["ransac_threshold"]),
            confidence=0.999999,
            maxIters=10000,
        )
        if H_match is None or mask is None:
            return failure(
                "low_confidence",
                "Could not find a consistent alignment among the matches.")
        mask = mask.ravel().astype(bool)
        num_inliers = int(mask.sum())
        inlier_ratio = num_inliers / float(num_matches)
        if num_inliers < int(opts["min_inliers"]) \
                or inlier_ratio < float(opts["min_inlier_ratio"]):
            return failure(
                "low_confidence",
                f"Alignment consensus too weak ({num_inliers} inliers, "
                f"{inlier_ratio:.0%} of matches) — try a frame with more "
                "distinctive structure.",
                num_inliers=num_inliers,
                inlier_ratio=round(inlier_ratio, 4))

        # Map inliers to native pixel coordinates and refit there (plain
        # least squares over inliers only, mirroring what a client-side fit
        # over the returned correspondences would produce).
        native_a = kpts_a * np.asarray(scale_a)
        native_b = kpts_b * np.asarray(scale_b)
        H_native, _ = cv2.findHomography(
            native_a[mask].astype(np.float64),
            native_b[mask].astype(np.float64),
            0,
        )
        if not _homography_is_sane(H_native, size_a, size_b):
            return failure(
                "degenerate_homography",
                "The fitted alignment is degenerate (matches may lie on a "
                "line or the scene is unsuitable) — try another frame.",
                num_inliers=num_inliers,
                inlier_ratio=round(inlier_ratio, 4))

        inliers = _spread_inliers(
            native_a, native_b, conf, mask, size_a, int(opts["top_k"]))

        return {
            "success": True,
            "homography": H_native.tolist(),
            "inliers": inliers,
            "num_matches": num_matches,
            "num_inliers": num_inliers,
            "inlier_ratio": round(inlier_ratio, 4),
            "image_size_a": list(size_a),
            "image_size_b": list(size_b),
            "model": "minima_loftr",
            "elapsed_ms": int((time.time() - start) * 1000),
        }

    # -------------------------------------------------------------- routing
    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        command = request.get("command")
        if command == "auto_align":
            for key in ("image_path_a", "image_path_b"):
                if not request.get(key):
                    raise ValueError(f"auto_align requires '{key}'")
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
        available = bool(
            self._weights_path and Path(self._weights_path).exists())
        return {
            "success": True,
            "available": available,
            "weights_path": self._weights_path,
            "loaded": self._model is not None,
            "device": self._device,
        }
