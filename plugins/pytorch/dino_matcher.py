# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""DINO feature-based stereo matching along epipolar curves.

Uses DINOv3 or DINOv2 dense ViT features to score epipolar curve candidates
by semantic similarity. Designed to work in two modes:

1. **Feature-only**: Return the best match by cosine similarity alone.
2. **Top-K filtering for NCC refinement**: Return the top-K candidates so that
   NCC template matching can select the precise match from a semantically
   filtered candidate set. This avoids NCC's failure mode on repetitive textures
   while preserving its sub-pixel precision.

Usage from C++ pipeline:
    The C++ code computes epipolar points, then calls get_top_k_indices()
    to filter candidates, and runs NCC on the filtered set.

Usage from Python:
    matcher = DINOMatcher(model_name='dinov2_vitb14')
    matcher.set_images(left_bgr, right_bgr)
    top_k = matcher.get_top_k_indices(source_xy, epipolar_points, k=100)
"""

import sys
import numpy as np

try:
    import torch
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


def _log(msg):
    print(f"[DINOMatcher] {msg}", file=sys.stderr, flush=True)


class DINOMatcher:
    """Stereo point matcher using DINO dense features along epipolar curves.

    Supports DINOv3 (from dinov3 package) and DINOv2 (via torch.hub)
    with automatic fallback.
    """

    _cached_model = None
    _cached_model_name = None

    def __init__(
        self,
        model_name="dinov2_vitb14",
        device="cuda",
        threshold=0.0,
        weights_path="",
    ):
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for DINOMatcher")

        self._model_name = model_name
        self._device = device if torch.cuda.is_available() else "cpu"
        self._threshold = threshold
        self._weights_path = weights_path

        self._model = None
        self._patch_size = 14

        self._left_features = None
        self._right_features = None
        self._left_feat_h = 0
        self._left_feat_w = 0
        self._right_feat_h = 0
        self._right_feat_w = 0
        self._left_img_h = 0
        self._left_img_w = 0
        self._right_img_h = 0
        self._right_img_w = 0

    def _load_model(self):
        """Load DINO backbone (cached across instances).

        Tries DINOv3 local package first, then DINOv3 torch.hub,
        then falls back to DINOv2 torch.hub.
        """
        if (DINOMatcher._cached_model is not None
                and DINOMatcher._cached_model_name == self._model_name):
            self._model = DINOMatcher._cached_model
            _log(f"Using cached model: {self._model_name}")
            return

        _log(f"Loading model: {self._model_name} on {self._device}")
        model = None

        # Try 1: DINOv3 local package
        if self._model_name.startswith("dinov3_"):
            try:
                from dinov3.hub import backbones
                model_fn = getattr(backbones, self._model_name)
                if self._weights_path:
                    model = model_fn(pretrained=True, weights=self._weights_path)
                else:
                    model = model_fn(pretrained=True)
                _log(f"Loaded via dinov3 package")
            except Exception as e:
                _log(f"DINOv3 package failed: {e}")

        # Try 2: DINOv3 torch.hub
        if model is None and self._model_name.startswith("dinov3_"):
            try:
                if self._weights_path:
                    model = torch.hub.load(
                        "facebookresearch/dinov3", self._model_name,
                        pretrained=True, weights=self._weights_path)
                else:
                    model = torch.hub.load(
                        "facebookresearch/dinov3", self._model_name,
                        pretrained=True)
                _log(f"Loaded via facebookresearch/dinov3 torch.hub")
            except Exception as e:
                _log(f"DINOv3 torch.hub failed: {e}")

        # Try 3: DINOv2 (either requested directly or as fallback)
        if model is None:
            dinov2_name = self._model_name
            if dinov2_name.startswith("dinov3_"):
                # Map DINOv3 names to DINOv2 equivalents
                dinov2_name = dinov2_name.replace("dinov3_", "dinov2_")
                # DINOv3 uses patch 16, DINOv2 uses patch 14
                dinov2_name = dinov2_name.replace("16", "14")
                _log(f"Falling back to DINOv2: {dinov2_name}")
            try:
                if self._weights_path:
                    model = torch.hub.load(
                        "facebookresearch/dinov2", dinov2_name,
                        pretrained=True, weights=self._weights_path)
                else:
                    model = torch.hub.load(
                        "facebookresearch/dinov2", dinov2_name,
                        pretrained=True)
                self._model_name = dinov2_name
                _log(f"Loaded via facebookresearch/dinov2 torch.hub")
            except Exception as e:
                _log(f"DINOv2 torch.hub also failed: {e}")
                raise RuntimeError(
                    f"Failed to load any DINO model. Last error: {e}")

        model = model.to(self._device)
        model.eval()

        if hasattr(model, 'patch_size'):
            self._patch_size = model.patch_size
        elif hasattr(model, 'patch_embed') and hasattr(model.patch_embed, 'patch_size'):
            ps = model.patch_embed.patch_size
            self._patch_size = ps[0] if isinstance(ps, (tuple, list)) else ps

        self._model = model
        DINOMatcher._cached_model = model
        DINOMatcher._cached_model_name = self._model_name

        _log(f"Model loaded: patch_size={self._patch_size}")

    def _preprocess(self, img_bgr):
        """Convert BGR uint8 image to normalized tensor."""
        img_rgb = img_bgr[:, :, ::-1].copy()
        tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std
        _, h, w = tensor.shape
        ps = self._patch_size
        pad_h = (ps - h % ps) % ps
        pad_w = (ps - w % ps) % ps
        if pad_h > 0 or pad_w > 0:
            tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode='reflect')
        return tensor.unsqueeze(0).to(self._device)

    @torch.no_grad()
    def _extract_features(self, img_bgr):
        """Extract dense patch features -> (1, C, H_feat, W_feat)."""
        if self._model is None:
            self._load_model()
        tensor = self._preprocess(img_bgr)
        _, _, h_pad, w_pad = tensor.shape
        h_feat = h_pad // self._patch_size
        w_feat = w_pad // self._patch_size
        features = self._model.forward_features(tensor)
        patch_tokens = features["x_norm_patchtokens"]
        feat_map = patch_tokens.reshape(1, h_feat, w_feat, -1)
        feat_map = feat_map.permute(0, 3, 1, 2)
        feat_map = F.normalize(feat_map, dim=1)
        return feat_map

    def set_images(self, left_bgr, right_bgr):
        """Extract and cache features for a stereo image pair."""
        self._left_img_h, self._left_img_w = left_bgr.shape[:2]
        self._right_img_h, self._right_img_w = right_bgr.shape[:2]
        self._left_features = self._extract_features(left_bgr)
        self._right_features = self._extract_features(right_bgr)
        _, _, self._left_feat_h, self._left_feat_w = self._left_features.shape
        _, _, self._right_feat_h, self._right_feat_w = self._right_features.shape

    def _build_grid_and_sample(self, source_xy, epipolar_points):
        """Compute cosine similarity scores for all epipolar candidates.

        Returns (scores_np, valid_indices) where scores_np has scores for
        valid candidates and valid_indices maps back to original indices.
        """
        src_feat = self._sample_feature(
            self._left_features, source_xy[0], source_xy[1],
            self._left_img_w, self._left_img_h,
            self._left_feat_w, self._left_feat_h)

        grid_coords = []
        valid_indices = []
        for i, (ex, ey) in enumerate(epipolar_points):
            if ex < 0 or ex >= self._right_img_w or ey < 0 or ey >= self._right_img_h:
                continue
            fx = ex / self._patch_size - 0.5
            fy = ey / self._patch_size - 0.5
            gx = 2.0 * fx / (self._right_feat_w - 1) - 1.0
            gy = 2.0 * fy / (self._right_feat_h - 1) - 1.0
            grid_coords.append([gx, gy])
            valid_indices.append(i)

        if not grid_coords:
            return np.array([]), []

        grid_t = torch.tensor(grid_coords, dtype=torch.float32,
                              device=self._right_features.device)
        grid_t = grid_t.view(1, 1, -1, 2)
        sampled = F.grid_sample(
            self._right_features, grid_t, mode='bilinear', align_corners=True)
        candidate_feats = sampled[0, :, 0, :]
        candidate_feats = F.normalize(candidate_feats, dim=0)
        scores = (src_feat.unsqueeze(1) * candidate_feats).sum(dim=0)

        return scores.cpu().numpy(), valid_indices

    def match_point(self, source_xy, epipolar_points):
        """Find best matching point by cosine similarity.

        Returns (best_x, best_y, score) or None.
        """
        if self._left_features is None or self._right_features is None:
            return None
        if not epipolar_points:
            return None

        scores_np, valid_indices = self._build_grid_and_sample(
            source_xy, epipolar_points)
        if len(scores_np) == 0:
            return None

        best_idx = int(np.argmax(scores_np))
        best_score = float(scores_np[best_idx])
        best_orig = valid_indices[best_idx]
        best_x, best_y = epipolar_points[best_orig]

        if best_score < self._threshold:
            return None
        return (best_x, best_y, best_score)

    def get_top_k_indices(self, source_xy, epipolar_points, k=100):
        """Return indices of top-K epipolar candidates by cosine similarity.

        This is the primary method for the two-stage approach: DINO selects
        the top-K semantically similar candidates, then NCC refines among them.

        Args:
            source_xy: (x, y) pixel coordinates of keypoint in left image.
            epipolar_points: List of (x, y) candidate pixel coordinates.
            k: Number of top candidates to return.

        Returns:
            List of (original_index, score) tuples, sorted descending by score.
        """
        if self._left_features is None or self._right_features is None:
            return []
        if not epipolar_points:
            return []

        scores_np, valid_indices = self._build_grid_and_sample(
            source_xy, epipolar_points)
        if len(scores_np) == 0:
            return []

        k = min(k, len(scores_np))
        topk_batch_indices = np.argsort(scores_np)[-k:][::-1]

        results = []
        for bi in topk_batch_indices:
            results.append((valid_indices[bi], float(scores_np[bi])))
        return results

    def _sample_feature(self, feat_map, x, y, img_w, img_h, feat_w, feat_h):
        """Bilinearly sample a feature vector at pixel (x, y)."""
        fx = x / self._patch_size - 0.5
        fy = y / self._patch_size - 0.5
        gx = 2.0 * fx / (feat_w - 1) - 1.0
        gy = 2.0 * fy / (feat_h - 1) - 1.0
        grid = torch.tensor([[[[gx, gy]]]], dtype=torch.float32,
                             device=feat_map.device)
        sampled = F.grid_sample(feat_map, grid, mode='bilinear', align_corners=True)
        vec = sampled[0, :, 0, 0]
        vec = F.normalize(vec, dim=0)
        return vec


# ---------------------------------------------------------------------------
# Module-level singleton for C++ interop
# ---------------------------------------------------------------------------

_global_matcher = None


def init_matcher(model_name="dinov2_vitb14", device="cuda", threshold=0.0,
                 weights_path=""):
    """Initialize the global DINO matcher (called from C++)."""
    global _global_matcher
    _global_matcher = DINOMatcher(
        model_name=model_name,
        device=device,
        threshold=threshold,
        weights_path=weights_path,
    )
    return True


def set_images(left_bgr, right_bgr):
    """Set stereo images and extract features (called from C++)."""
    global _global_matcher
    if _global_matcher is None:
        init_matcher()
    _global_matcher.set_images(left_bgr, right_bgr)
    return True


def set_images_from_bytes(left_bytes, left_h, left_w, left_c,
                          right_bytes, right_h, right_w, right_c):
    """Set stereo images from raw byte buffers (called from C++ via Python C API)."""
    left_shape = (left_h, left_w, left_c) if left_c > 1 else (left_h, left_w)
    right_shape = (right_h, right_w, right_c) if right_c > 1 else (right_h, right_w)
    left_bgr = np.frombuffer(left_bytes, dtype=np.uint8).reshape(left_shape).copy()
    right_bgr = np.frombuffer(right_bytes, dtype=np.uint8).reshape(right_shape).copy()
    if left_bgr.ndim == 2:
        left_bgr = np.stack([left_bgr] * 3, axis=-1)
    if right_bgr.ndim == 2:
        right_bgr = np.stack([right_bgr] * 3, axis=-1)
    return set_images(left_bgr, right_bgr)


def match_point(source_x, source_y, epipolar_xs, epipolar_ys, threshold=-1.0):
    """Match a single point (called from C++).

    Returns (success, matched_x, matched_y, score).
    """
    global _global_matcher
    if _global_matcher is None:
        return (False, 0.0, 0.0, 0.0)

    epi_pts = list(zip(epipolar_xs, epipolar_ys))

    old_thresh = _global_matcher._threshold
    if threshold >= 0:
        _global_matcher._threshold = threshold
    result = _global_matcher.match_point((source_x, source_y), epi_pts)
    _global_matcher._threshold = old_thresh

    if result is None:
        return (False, 0.0, 0.0, 0.0)
    return (True, result[0], result[1], result[2])


def get_top_k_indices(source_x, source_y, epipolar_xs, epipolar_ys, k=100):
    """Get top-K candidate indices by DINO similarity (called from C++).

    Returns a list of integer indices into the epipolar arrays, sorted by
    descending cosine similarity. The C++ side uses these indices to filter
    the candidate set for NCC template matching.

    Args:
        source_x, source_y: Source keypoint pixel coords in left image.
        epipolar_xs, epipolar_ys: Arrays of candidate coords in right image.
        k: Number of top candidates to return.

    Returns:
        List of int indices (into the epipolar arrays).
    """
    global _global_matcher
    if _global_matcher is None:
        return []

    epi_pts = list(zip(epipolar_xs, epipolar_ys))
    top_k = _global_matcher.get_top_k_indices(
        (source_x, source_y), epi_pts, k=k)

    return [idx for idx, score in top_k]
