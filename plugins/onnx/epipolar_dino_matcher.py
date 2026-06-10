# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
DINO + NCC fused epipolar stereo matching as a single ONNX graph.

This is stereo measurement method (2): the two-stage matcher selected by
``epipolar_descriptor_type=dino`` in
``configs/add-ons/dino/measurement_from_annotations_ncc_dino.pipe``. Per source
keypoint it does:

  1. Generate epipolar candidates from calibration (identical to method 1).
  2. DINO stage: extract dense DINOv2 ViT features for both images, bilinearly
     sample the source feature and every candidate feature, score by cosine
     similarity, and keep the top-K candidates. (Mirrors
     ``plugins/pytorch/dino_matcher.py``: forward_features -> x_norm_patchtokens
     -> per-channel L2-normalized feature map, grid_sample with align_corners.)
  3. NCC stage: run the exact method-1 NCC template match on just those K
     candidates and take the best. DINO removes repetitive-texture false matches;
     NCC keeps sub-pixel precision.

The DINOv2 backbone is baked into the graph (so the .onnx is self-contained but
large, ~90 MB for vitb14). Calibration is supplied as runtime tensor inputs, as
in method 1. Unlike method 1, the image size is FIXED at export time: ViT
positional-embedding interpolation makes dynamic-resolution ONNX export fragile,
and a stereo rig's resolution is constant. Export at your camera resolution.

Two graphs (see export_stereo_mapping.py): EpipolarDinoMatcher (-> matched points +
scores) and EpipolarDinoMeasurer (+ in-graph triangulation).

Image inputs are color, float32 [3, H, W], RGB, range [0, 255]. Grayscale for
NCC is derived in-graph with the BT.601 luma weights OpenCV's BGR2GRAY uses, and
the DINO ImageNet normalization is applied in-graph.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from epipolar_matcher import (
    EpipolarMatcher, compute_epipolar_points, unmap,
    _gather_patches, _ncc, _round_half_up, _INVALID_SCORE,
)

# ImageNet normalization (matches dino_matcher._preprocess).
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]
# BT.601 luma weights for RGB, matching cv2.COLOR_BGR2GRAY.
_LUMA = [0.299, 0.587, 0.114]


def load_dino_backbone(model_name="dinov2_vitb14", weights_path=""):
    """Load a DINOv2 backbone via torch.hub, mirroring dino_matcher._load_model.

    Returns (model, patch_size). The model is put in eval mode on CPU (export
    happens on CPU). xformers is intentionally not required -- its absence makes
    DINOv2 fall back to standard attention, which is what exports to ONNX.
    """
    if weights_path:
        model = torch.hub.load("facebookresearch/dinov2", model_name,
                               pretrained=True, weights=weights_path,
                               verbose=False)
    else:
        model = torch.hub.load("facebookresearch/dinov2", model_name,
                               pretrained=True, verbose=False)
    model.eval()
    ps = getattr(model, "patch_size", 14)
    if isinstance(ps, (tuple, list)):
        ps = ps[0]
    return model, ps


class _DinoFeatures(nn.Module):
    """Wrap a DINOv2 backbone to produce a per-channel L2-normalized dense
    feature map [1, C, Hf, Wf] from a color image [1, 3, H, W] in [0, 255] RGB.
    Reproduces dino_matcher._preprocess + _extract_features for a fixed size."""

    def __init__(self, model, patch_size, height, width):
        super().__init__()
        self.model = model
        self.ps = int(patch_size)
        # Pad (reflect) to a multiple of the patch size, as dino_matcher does.
        self.pad_h = (self.ps - height % self.ps) % self.ps
        self.pad_w = (self.ps - width % self.ps) % self.ps
        self.feat_h = (height + self.pad_h) // self.ps
        self.feat_w = (width + self.pad_w) // self.ps
        self.register_buffer("mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, img_rgb):                 # img_rgb: [1, 3, H, W] in [0,255]
        x = (img_rgb / 255.0 - self.mean) / self.std
        if self.pad_h > 0 or self.pad_w > 0:
            x = F.pad(x, (0, self.pad_w, 0, self.pad_h), mode="reflect")
        tokens = self.model.forward_features(x)["x_norm_patchtokens"]
        fmap = tokens.reshape(1, self.feat_h, self.feat_w, -1).permute(0, 3, 1, 2)
        return F.normalize(fmap, dim=1)         # [1, C, Hf, Wf]


def _pixel_to_grid(px, py, feat_w, feat_h, ps):
    """Pixel coords -> grid_sample coords (align_corners=True), per
    dino_matcher._sample_feature. px, py any shape; returns stacked [..., 2]."""
    gx = 2.0 * (px / ps - 0.5) / (feat_w - 1) - 1.0
    gy = 2.0 * (py / ps - 0.5) / (feat_h - 1) - 1.0
    return torch.stack([gx, gy], dim=-1)


def _rgb_to_gray(img_rgb):
    """[1,3,H,W] RGB -> [H,W] grayscale via BT.601 (matches cv2 BGR2GRAY)."""
    w = torch.tensor(_LUMA, dtype=img_rgb.dtype, device=img_rgb.device).view(3, 1, 1)
    return (img_rgb[0] * w).sum(0)


class EpipolarDinoMatcher(EpipolarMatcher):
    """DINO top-K + NCC fused matcher (method 2) as one graph.

    forward inputs (calibration block identical to EpipolarMatcher):
      left_rgb, right_rgb : [3, H, W] float32 RGB in [0, 255]
      points_left         : [P, 2]
      K_left, dist_left, R_left, t_left, K_right, dist_right, R_right, t_right
      min_depth, max_depth

    forward outputs:
      right_points : [P, 2] matched pixel in the right image
      best_score   : [P]    NCC score of the chosen (DINO-filtered) candidate
      second_score : [P]    best NCC score outside a template_size neighborhood,
                            among the top-K (for the host uniqueness test)

    template_size, num_samples and dino_top_k are graph constants.
    """

    def __init__(self, dino_model, patch_size, height, width,
                 template_size=25, num_samples=5000, dino_top_k=25):
        super().__init__(template_size, num_samples)
        self.dino = _DinoFeatures(dino_model, patch_size, height, width)
        self.top_k = int(dino_top_k)

    def _match_dino(self, left_rgb, right_rgb, points_left,
                    K_left, dist_left, R_left, t_left,
                    K_right, dist_right, R_right, t_right,
                    min_depth, max_depth):
        half = self.half
        ps = self.dino.ps
        fh, fw = self.dino.feat_h, self.dino.feat_w

        left_gray = _rgb_to_gray(left_rgb)
        right_gray = _rgb_to_gray(right_rgb)
        Hr = right_gray.shape[0]
        Wr = right_gray.shape[1]

        px = points_left[:, 0]
        py = points_left[:, 1]

        # --- epipolar candidates (identical to method 1) ---
        proj, valid, nx_l, ny_l = compute_epipolar_points(
            px, py, K_left, dist_left, R_left, t_left,
            K_right, dist_right, R_right, t_right,
            min_depth, max_depth, self.num_samples)             # proj [P, N, 2]

        # --- DINO stage: cosine similarity for every candidate ---
        left_feat = self.dino(left_rgb.unsqueeze(0))            # [1, C, fh, fw]
        right_feat = self.dino(right_rgb.unsqueeze(0))

        src_grid = _pixel_to_grid(px, py, fw, fh, ps).view(1, 1, -1, 2)
        src_feat = F.grid_sample(left_feat, src_grid, mode="bilinear",
                                 align_corners=True)[0, :, 0, :]  # [C, P]
        src_feat = F.normalize(src_feat, dim=0)

        cand_grid = _pixel_to_grid(proj[..., 0], proj[..., 1],
                                   fw, fh, ps).unsqueeze(0)      # [1, P, N, 2]
        cand_feat = F.grid_sample(right_feat, cand_grid, mode="bilinear",
                                  align_corners=True)[0]          # [C, P, N]
        cand_feat = F.normalize(cand_feat, dim=0)
        dino_scores = (src_feat.unsqueeze(-1) * cand_feat).sum(0)  # [P, N]

        # Candidates outside the right image (or behind the camera) are excluded
        # from the DINO ranking, matching dino_matcher's valid_indices filter.
        in_img = (proj[..., 0] >= 0) & (proj[..., 0] < Wr) & \
                 (proj[..., 1] >= 0) & (proj[..., 1] < Hr)
        dino_ok = valid & in_img
        dino_scores = torch.where(
            dino_ok, dino_scores, torch.full_like(dino_scores, -2.0))

        # --- top-K filter ---
        k = min(self.top_k, self.num_samples)
        topk_idx = torch.topk(dino_scores, k, dim=1).indices       # [P, k]
        topk_proj = torch.gather(
            proj, 1, topk_idx.unsqueeze(-1).expand(-1, -1, 2))     # [P, k, 2]

        # --- NCC stage on the K candidates (method-1 NCC) ---
        sx = _round_half_up(px).to(torch.int64)
        sy = _round_half_up(py).to(torch.int64)
        template = _gather_patches(left_gray, sx, sy, half)        # [P, T, T]

        cx = _round_half_up(topk_proj[..., 0]).to(torch.int64)     # [P, k]
        cy = _round_half_up(topk_proj[..., 1]).to(torch.int64)
        patches = _gather_patches(right_gray, cx, cy, half)        # [P, k, T, T]
        scores = _ncc(template, patches)                           # [P, k]

        in_x = (cx >= half) & (cx <= Wr - 1 - half)
        in_y = (cy >= half) & (cy <= Hr - 1 - half)
        ok = in_x & in_y
        scores = torch.where(ok, scores, torch.full_like(scores, _INVALID_SCORE))

        best, idx = scores.max(dim=1)
        right_points = torch.gather(
            topk_proj, 1, idx.view(-1, 1, 1).expand(-1, 1, 2)).squeeze(1)

        dx = topk_proj[..., 0] - right_points[:, 0:1]
        dy = topk_proj[..., 1] - right_points[:, 1:2]
        far = (dx * dx + dy * dy) >= float(self.template_size * self.template_size)
        second = torch.where(far, scores,
                             torch.full_like(scores, _INVALID_SCORE)).max(dim=1)[0]
        return right_points, best, second, nx_l, ny_l

    def forward(self, left_rgb, right_rgb, points_left,
                K_left, dist_left, R_left, t_left,
                K_right, dist_right, R_right, t_right,
                min_depth, max_depth):
        right_points, best, second, _, _ = self._match_dino(
            left_rgb, right_rgb, points_left,
            K_left, dist_left, R_left, t_left,
            K_right, dist_right, R_right, t_right, min_depth, max_depth)
        return right_points, best, second


class EpipolarDinoMeasurer(EpipolarDinoMatcher):
    """EpipolarDinoMatcher plus in-graph triangulation (see triangulate.py)."""

    def forward(self, left_rgb, right_rgb, points_left,
                K_left, dist_left, R_left, t_left,
                K_right, dist_right, R_right, t_right,
                min_depth, max_depth):
        from triangulate import triangulate_fast_torch

        right_points, best, second, nx_l, ny_l = self._match_dino(
            left_rgb, right_rgb, points_left,
            K_left, dist_left, R_left, t_left,
            K_right, dist_right, R_right, t_right, min_depth, max_depth)
        nx_r, ny_r = unmap(right_points[:, 0], right_points[:, 1],
                           K_right, dist_right)
        points_3d = triangulate_fast_torch(
            nx_l, ny_l, nx_r, ny_r, R_left, t_left, R_right, t_right)
        return right_points, best, second, points_3d
