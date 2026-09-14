#!/usr/bin/env python
# This file is part of VIAME, and is distributed under an OSI-approved
# BSD 3-Clause License. See either the root top-level LICENSE file or
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.

"""
Cross-spectral camera alignment core (auto-align).

The matching and fitting logic behind the ``align_cameras`` pipeline
process, which registers many image pairs per job. Computes
homographies between two camera images of the same scene -- typically
different modalities (EO/RGB vs thermal IR) -- using the vendored
MINIMA-LoFTR matcher (viame.pytorch.minima_loftr), a LoFTR fine-tuned on
multimodal data that is robust to cross-spectral appearance changes.

Contents:
  LoftrMatcher          owns the model: lazy load, unload(), match()
  register_image_pair   the full single-pair pipeline: load/normalize,
                        match at MATCH_SIZE, MAGSAC at matcher resolution,
                        native-pixel refit, sanity check, spread inliers
  find_alignment_weights  locate minima_loftr.ckpt in the install

Failures the caller can act on are returned as ``success: False`` with a
machine-readable ``code``:
  insufficient_matches   scene has too little structure for the matcher
  low_confidence         matches found but RANSAC consensus is too weak
  degenerate_homography  a consensus exists but the fit is not a sane warp
"""

import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Matcher input size: long side after resize, dims floored to /8 (LoFTR df).
MATCH_SIZE = 640

# Grid used to pick spatially-spread inliers (and to measure coverage).
SPREAD_GRID = 6

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
    # Percentile stretch window for high-bit-depth (e.g. 16-bit IR) inputs
    # so low-contrast frames still spread over 8 bits. Consulted ONLY for
    # high-bit-depth single-channel inputs; 8-bit imagery is never
    # stretched, by any configuration.
    "lower_percentile": 2.0,
    "upper_percentile": 98.0,
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

    if device is not None and device.startswith("cuda") \
            and not torch.cuda.is_available():
        # Callers may default to cuda; degrade to the best available
        # backend rather than silently landing on CPU.
        device = "auto"
    if device in (None, "", "auto"):
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None \
                and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device


def _load_gray_norm(
    path: str,
    lower_percentile: float = DEFAULT_OPTIONS["lower_percentile"],
    upper_percentile: float = DEFAULT_OPTIONS["upper_percentile"],
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Load any 8/16-bit gray/color image as normalized 8-bit grayscale.

    The percentile stretch applies if and only if the input is a
    high-bit-depth single-channel image; 8-bit imagery passes through
    untouched regardless of the percentile arguments.

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
        lo, hi = np.percentile(arr, (lower_percentile, upper_percentile))
        arr = np.clip((arr - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        gray = (arr * 255.0).astype(np.uint8)
    return gray, (w, h)


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


def _spread_inliers(
    kpts_a: np.ndarray,
    kpts_b: np.ndarray,
    conf: np.ndarray,
    mask: np.ndarray,
    size_a: Tuple[int, int],
    top_k: int,
    grid: int = SPREAD_GRID,
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


def _grid_coverage(
    kpts_a: np.ndarray,
    mask: np.ndarray,
    size_a: Tuple[int, int],
    grid: int = SPREAD_GRID,
) -> float:
    """Fraction of grid cells over image A occupied by at least one inlier.

    The honest "dense features" signal: 24 inliers clustered in one corner
    score lower than 15 spread across the frame."""
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return 0.0
    w, h = size_a
    cells = set()
    for i in idx:
        cells.add((
            min(int(kpts_a[i, 0] / w * grid), grid - 1),
            min(int(kpts_a[i, 1] / h * grid), grid - 1),
        ))
    return len(cells) / float(grid * grid)


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


def fit_pooled_homography(
    observations: List[Dict[str, Any]],
    opts: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """MAGSAC over correspondences pooled from every image pair.

    Normalizes each observation's points by its own image size before RANSAC
    so the pixel threshold means the same thing across frames, fits one
    homography, then reports per-observation inlier counts and reprojection
    RMS so callers can identify (and the GUI can surface) a frame that
    disagrees with the consensus.

    Each observation is a dict with:
      points  [[ax, ay, bx, by], ...] in native pixels
      size_a  (width, height) of image A for that observation
      size_b  (width, height) of image B

    Returns on success::

      {success: True, homography (3x3 A native px -> B native px),
       num_points, num_inliers, rms_px,
       observations: [{num_points, num_inliers, inlier_ratio, rms_px}, ...]}

    Per-observation ``rms_px`` is computed over *all* of that observation's
    points against the pooled fit -- a frame whose points are outliers to
    the consensus shows a large RMS rather than disappearing from the
    statistics.
    """
    merged = dict(DEFAULT_OPTIONS)
    merged.update(opts or {})

    norm_a: List[np.ndarray] = []
    norm_b: List[np.ndarray] = []
    native_a: List[np.ndarray] = []
    native_b: List[np.ndarray] = []
    obs_slices: List[Tuple[int, int]] = []
    first_sizes: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None
    cursor = 0
    for obs in observations:
        pts = np.asarray(
            obs.get("points") if obs.get("points") is not None else [],
            dtype=np.float64,
        ).reshape(-1, 4)
        n = len(pts)
        obs_slices.append((cursor, cursor + n))
        cursor += n
        if n == 0:
            continue
        wa, ha = obs["size_a"]
        wb, hb = obs["size_b"]
        if first_sizes is None:
            first_sizes = ((wa, ha), (wb, hb))
        sa = MATCH_SIZE / float(max(wa, ha))
        sb = MATCH_SIZE / float(max(wb, hb))
        native_a.append(pts[:, 0:2])
        native_b.append(pts[:, 2:4])
        norm_a.append(pts[:, 0:2] * sa)
        norm_b.append(pts[:, 2:4] * sb)

    num_points = cursor

    def failure(code: str, message: str) -> Dict[str, Any]:
        return {
            "success": False,
            "code": code,
            "error": message,
            "num_points": num_points,
        }

    if num_points < 4:
        return failure(
            "insufficient_points",
            f"Only {num_points} pooled correspondences (need >=4).")

    a_norm = np.concatenate(norm_a)
    b_norm = np.concatenate(norm_b)
    a_native = np.concatenate(native_a)
    b_native = np.concatenate(native_b)

    cv2.setRNGSeed(0)
    H_norm, mask = cv2.findHomography(
        a_norm, b_norm,
        cv2.USAC_MAGSAC,
        ransacReprojThreshold=float(merged["ransac_threshold"]),
        confidence=0.999999,
        maxIters=10000,
    )
    if H_norm is None or mask is None:
        return failure(
            "low_confidence",
            "Could not find a consistent pooled alignment; the frames may "
            "disagree with each other.")
    mask = mask.ravel().astype(bool)
    num_inliers = int(mask.sum())
    if num_inliers < 4:
        return failure(
            "low_confidence",
            f"Pooled consensus too weak ({num_inliers} inliers).")

    # Refit in native pixels over the inliers (plain least squares),
    # mirroring the single-pair path.
    H_native, _ = cv2.findHomography(
        a_native[mask], b_native[mask], 0)
    if H_native is None or first_sizes is None \
            or not _homography_is_sane(H_native, first_sizes[0],
                                       first_sizes[1]):
        return failure(
            "degenerate_homography",
            "The pooled alignment fit is degenerate.")

    def reprojection_errors(a_pts: np.ndarray, b_pts: np.ndarray):
        projected = cv2.perspectiveTransform(
            a_pts.reshape(-1, 1, 2), H_native).reshape(-1, 2)
        return np.linalg.norm(projected - b_pts, axis=1)

    all_errors = reprojection_errors(a_native, b_native)
    overall_rms = float(np.sqrt(np.mean(all_errors[mask] ** 2)))

    per_obs: List[Dict[str, Any]] = []
    for start, end in obs_slices:
        n = end - start
        if n == 0:
            per_obs.append({
                "num_points": 0,
                "num_inliers": 0,
                "inlier_ratio": 0.0,
                "rms_px": None,
            })
            continue
        obs_mask = mask[start:end]
        obs_errors = all_errors[start:end]
        obs_inliers = int(obs_mask.sum())
        per_obs.append({
            "num_points": n,
            "num_inliers": obs_inliers,
            "inlier_ratio": round(obs_inliers / float(n), 4),
            "rms_px": round(float(np.sqrt(np.mean(obs_errors ** 2))), 4),
        })

    return {
        "success": True,
        "homography": H_native.tolist(),
        "num_points": num_points,
        "num_inliers": num_inliers,
        "rms_px": round(overall_rms, 4),
        "observations": per_obs,
    }


def loop_closure_residual(
    H_12: np.ndarray,
    H_23: np.ndarray,
    H_13: np.ndarray,
    size_1: Tuple[int, int],
    grid: int = 10,
) -> Dict[str, float]:
    """Mutual-consistency check for a solved camera triplet.

    Pushes a grid of points from camera 1 into camera 3 via the direct
    homography (H_13) and via the composed route (H_23 . H_12), and reports
    the mean/max disagreement in camera-3 pixels. Near-zero means the three
    independently fitted pairs describe one consistent rig; a large value
    means at least one pair disagrees with the route through the third
    camera.
    """
    w, h = size_1
    xs = np.linspace(0, w, grid)
    ys = np.linspace(0, h, grid)
    points = np.array(
        [[x, y] for y in ys for x in xs], dtype=np.float64
    ).reshape(-1, 1, 2)
    direct = cv2.perspectiveTransform(points, np.asarray(H_13, np.float64))
    composed = cv2.perspectiveTransform(
        points, np.asarray(H_23, np.float64) @ np.asarray(H_12, np.float64))
    dists = np.linalg.norm(
        direct.reshape(-1, 2) - composed.reshape(-1, 2), axis=1)
    return {
        "mean_px": round(float(np.mean(dists)), 4),
        "max_px": round(float(np.max(dists)), 4),
    }


class MatchResult:
    """Raw matcher output for one image pair, in matcher coordinates.

    ``scale_a``/``scale_b`` map matcher coords -> native pixels;
    ``size_a``/``size_b`` are native (width, height)."""

    __slots__ = ("kpts_a", "kpts_b", "conf", "scale_a", "scale_b",
                 "size_a", "size_b")

    def __init__(self, kpts_a, kpts_b, conf, scale_a, scale_b,
                 size_a, size_b):
        self.kpts_a = kpts_a
        self.kpts_b = kpts_b
        self.conf = conf
        self.scale_a = scale_a
        self.scale_b = scale_b
        self.size_a = size_a
        self.size_b = size_b


class LoftrMatcher:
    """Owns the MINIMA-LoFTR model: lazy load, unload, match one pair."""

    def __init__(
        self,
        weights_path: Optional[str] = None,
        device: Optional[str] = None,
        log: Optional[Callable[[str], None]] = None,
    ):
        self._weights_path = weights_path or find_alignment_weights()
        self._requested_device = device
        self._model = None
        self._device: Optional[str] = None
        self._log = log or (lambda message: None)

    @property
    def weights_path(self) -> Optional[str]:
        return self._weights_path

    @property
    def device(self) -> Optional[str]:
        return self._device

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def ensure_model(self):
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

    def unload(self) -> bool:
        """Drop the model and release its device memory.

        Returns whether a model was actually loaded. The freed allocations
        are returned to the driver so other consumers (segmentation/stereo)
        can actually reuse the VRAM, not just this process's caching
        allocator. The model reloads lazily on the next match()."""
        was_loaded = self._model is not None
        self._model = None
        if was_loaded:
            import gc
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                mps = getattr(torch.backends, "mps", None)
                if mps is not None and mps.is_available():
                    torch.mps.empty_cache()
            except Exception:
                pass
            self._log("Alignment model unloaded")
        return was_loaded

    def match(
        self,
        image_path_a: str,
        image_path_b: str,
        opts: Dict[str, Any],
    ) -> MatchResult:
        """Match one image pair at MATCH_SIZE resolution."""
        import torch

        model = self.ensure_model()
        # CoarseMatching reads thr at forward time from its own attribute.
        model.coarse_matching.thr = float(opts["match_threshold"])

        device = self._device or "cpu"
        lo = float(opts.get(
            "lower_percentile", DEFAULT_OPTIONS["lower_percentile"]))
        hi = float(opts.get(
            "upper_percentile", DEFAULT_OPTIONS["upper_percentile"]))
        lo_b = float(opts.get("cam2_lower_percentile", lo))
        hi_b = float(opts.get("cam2_upper_percentile", hi))
        gray_a, size_a = _load_gray_norm(image_path_a, lo, hi)
        gray_b, size_b = _load_gray_norm(image_path_b, lo_b, hi_b)
        img_a, scale_a = _to_match_tensor(gray_a, device)
        img_b, scale_b = _to_match_tensor(gray_b, device)

        batch = {"image0": img_a, "image1": img_b}
        with torch.no_grad():
            model(batch)
        return MatchResult(
            kpts_a=batch["mkpts0_f"].cpu().numpy(),
            kpts_b=batch["mkpts1_f"].cpu().numpy(),
            conf=batch["mconf"].cpu().numpy(),
            scale_a=scale_a,
            scale_b=scale_b,
            size_a=size_a,
            size_b=size_b,
        )


def register_image_pair(
    matcher: LoftrMatcher,
    image_path_a: str,
    image_path_b: str,
    options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Full single-pair registration: match, MAGSAC, native refit, spread.

    Returns, on success, a dict with
    ``homography`` (3x3, A native px -> B native px), ``inliers``
    (spatially spread ``[[ax, ay, bx, by], ...]`` in native px),
    ``num_matches``, ``num_inliers``, ``inlier_ratio``, ``coverage``,
    ``image_size_a``, ``image_size_b``, ``model``, ``elapsed_ms``; on
    failure ``success: False`` with a machine-readable ``code``.
    """
    opts = dict(DEFAULT_OPTIONS)
    opts.update(options or {})
    start = time.time()

    result = matcher.match(image_path_a, image_path_b, opts)
    kpts_a, kpts_b, conf = result.kpts_a, result.kpts_b, result.conf
    size_a, size_b = result.size_a, result.size_b
    num_matches = len(kpts_a)

    def failure(code: str, message: str, **extra) -> Dict[str, Any]:
        out = {
            "success": False,
            "code": code,
            "error": message,
            "num_matches": num_matches,
            "model": "minima_loftr",
            "elapsed_ms": int((time.time() - start) * 1000),
        }
        out.update(extra)
        return out

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
    native_a = kpts_a * np.asarray(result.scale_a)
    native_b = kpts_b * np.asarray(result.scale_b)
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
    coverage = _grid_coverage(native_a, mask, size_a)

    return {
        "success": True,
        "homography": H_native.tolist(),
        "inliers": inliers,
        "num_matches": num_matches,
        "num_inliers": num_inliers,
        "inlier_ratio": round(inlier_ratio, 4),
        "coverage": round(coverage, 4),
        "image_size_a": list(size_a),
        "image_size_b": list(size_b),
        "model": "minima_loftr",
        "elapsed_ms": int((time.time() - start) * 1000),
    }
