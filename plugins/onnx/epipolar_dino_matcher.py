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

import os

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
    # DINOv2 routes attention through xformers when it is importable, and that
    # path has no CPU float32 kernel -- so an export box that happens to have
    # xformers installed fails to trace. Opt out explicitly rather than relying
    # on it being absent.
    os.environ.setdefault("XFORMERS_DISABLED", "1")
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

    def forward(self, img_rgb):                 # img_rgb: [B, 3, H, W] in [0,255]
        x = (img_rgb / 255.0 - self.mean) / self.std
        if self.pad_h > 0 or self.pad_w > 0:
            x = F.pad(x, (0, self.pad_w, 0, self.pad_h), mode="reflect")
        tokens = self.model.forward_features(x)["x_norm_patchtokens"]
        # [B, Hf*Wf, C] -> [B, C, Hf, Wf]. Reshaping with -1 for the batch keeps
        # it dynamic, so the same module serves one whole image or P crops.
        c = tokens.shape[-1]
        fmap = tokens.transpose(1, 2).reshape(-1, c, self.feat_h, self.feat_w)
        return F.normalize(fmap, dim=1)         # [B, C, Hf, Wf]


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

    def _dino_similarity(self, left_rgb, right_rgb, px, py, proj):
        """Cosine similarity between the source feature and every candidate
        feature, from dense whole-image DINO features.

        Returns (scores [P, N], sampled_ok [P, N]); sampled_ok is all-True here
        because every candidate is covered by the dense map.
        """
        ps = self.dino.ps
        fh, fw = self.dino.feat_h, self.dino.feat_w

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
        scores = (src_feat.unsqueeze(-1) * cand_feat).sum(0)      # [P, N]
        return scores, torch.ones_like(scores, dtype=torch.bool)

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
        dino_scores, sampled_ok = self._dino_similarity(
            left_rgb, right_rgb, px, py, proj)                     # [P, N]

        # Candidates outside the right image (or behind the camera) are excluded
        # from the DINO ranking, matching dino_matcher's valid_indices filter.
        in_img = (proj[..., 0] >= 0) & (proj[..., 0] < Wr) & \
                 (proj[..., 1] >= 0) & (proj[..., 1] < Hr)
        dino_ok = valid & in_img & sampled_ok
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


def _crop_batch(img_rgb, ox, oy, crop_h, crop_w):
    """Cut one fixed-size crop per keypoint out of a [3, H, W] image.

    ox, oy are int64 [P] top-left origins, already clamped in range. Gathering
    with broadcast index grids (rather than slicing) is what lets every
    keypoint have its own origin in a single traced op.
    Returns [P, 3, crop_h, crop_w].
    """
    ys = oy.view(-1, 1) + torch.arange(crop_h, dtype=torch.int64,
                                       device=img_rgb.device).view(1, -1)   # [P, ch]
    xs = ox.view(-1, 1) + torch.arange(crop_w, dtype=torch.int64,
                                       device=img_rgb.device).view(1, -1)   # [P, cw]
    # An image smaller than the crop, or a crop pushed against an edge, would
    # index past the end; clamping replicates the edge pixel instead, which is
    # the same kind of padding _DinoFeatures already applies for the patch grid.
    ys = torch.clamp(ys, min=0, max=img_rgb.shape[-2] - 1)
    xs = torch.clamp(xs, min=0, max=img_rgb.shape[-1] - 1)
    crops = img_rgb[:, ys.unsqueeze(-1), xs.unsqueeze(1)]                   # [3, P, ch, cw]
    return crops.permute(1, 0, 2, 3)


def _clamped_origin(center, extent, limit):
    """Top-left origin of a window of size `extent` centred on `center`, kept
    inside [0, limit - extent]. `limit` may be a traced (dynamic) dimension."""
    origin = torch.clamp(_round_half_up(center - extent / 2.0).to(torch.int64),
                         min=0)
    # Build the upper bound as a tensor before clamping: `limit` is a traced
    # image dimension, so keeping it in tensor arithmetic is what stops the
    # export from baking in the resolution it was traced at.
    hi = (limit - extent) * torch.ones_like(origin)
    return torch.minimum(origin, torch.clamp(hi, min=0))


class EpipolarDinoCropMatcher(EpipolarDinoMatcher):
    """Method 2 with the ViT run on fixed-size crops instead of whole frames,
    so the exported graph accepts ANY image resolution at runtime.

    The DINO stage only ever samples features at the source point and along the
    epipolar candidate curve, so there is no need to featurize the whole frame.
    Per keypoint this cuts one `crop_size`-square window around the source point
    from the left image, and one around the centre of that keypoint's candidate
    span from the right image. The ViT therefore always sees a constant
    [2P, 3, crop, crop] batch — its positional-embedding interpolation is fixed
    at export time as before — while `left_rgb` / `right_rgb` stay dynamic.

    Two consequences worth knowing:

      * The crop must cover the candidate span, which is set by the disparity
        search range. Candidates falling outside it are dropped from the DINO
        ranking (they can never be selected), so size the crop for the widest
        disparity you expect. There is no runtime warning when this bites.
      * Features are computed from a crop, not the whole frame, so a token's
        context differs from the dense reference in dino_matcher.py. Validate
        against it before trusting the two to agree.

    In exchange the ViT cost stops scaling with frame size: a 1920x1080 frame is
    ~10.5k patch tokens with quadratic attention, versus 576 per crop at the
    default 336.
    """

    def __init__(self, dino_model, patch_size, crop_size,
                 template_size=25, num_samples=5000, dino_top_k=25):
        # The parent's _DinoFeatures is built for the crop, not a frame.
        super().__init__(dino_model, patch_size, crop_size, crop_size,
                         template_size, num_samples, dino_top_k)
        if crop_size % int(patch_size):
            raise ValueError(
                "crop_size %d must be a multiple of the patch size %d"
                % (crop_size, patch_size))
        self.crop = int(crop_size)

    def _dino_similarity(self, left_rgb, right_rgb, px, py, proj):
        ps = self.dino.ps
        fh, fw = self.dino.feat_h, self.dino.feat_w
        crop = self.crop
        Hl, Wl = left_rgb.shape[-2], left_rgb.shape[-1]
        Hr, Wr = right_rgb.shape[-2], right_rgb.shape[-1]

        # Left crop: centred on the source point.
        lox = _clamped_origin(px, crop, Wl)
        loy = _clamped_origin(py, crop, Hl)

        # Right crop: centred on the midpoint of this keypoint's candidate span.
        # Invalid candidates carry junk coordinates, so bound only over valid ones.
        big = torch.full_like(proj[..., 0], 1e9)
        cand_valid = torch.isfinite(proj[..., 0]) & torch.isfinite(proj[..., 1])
        xs = torch.where(cand_valid, proj[..., 0], big)
        ys = torch.where(cand_valid, proj[..., 1], big)
        x_lo = xs.min(dim=1).values
        y_lo = ys.min(dim=1).values
        x_hi = torch.where(cand_valid, proj[..., 0], -big).max(dim=1).values
        y_hi = torch.where(cand_valid, proj[..., 1], -big).max(dim=1).values
        rox = _clamped_origin((x_lo + x_hi) / 2.0, crop, Wr)
        roy = _clamped_origin((y_lo + y_hi) / 2.0, crop, Hr)

        feats = self.dino(torch.cat([
            _crop_batch(left_rgb, lox, loy, crop, crop),
            _crop_batch(right_rgb, rox, roy, crop, crop),
        ], dim=0))                                              # [2P, C, fh, fw]
        p = px.shape[0]
        left_feat = feats[:p]
        right_feat = feats[p:]

        # Sample in crop-local pixel coordinates.
        src_grid = _pixel_to_grid(px - lox.to(px.dtype), py - loy.to(py.dtype),
                                  fw, fh, ps).view(-1, 1, 1, 2)  # [P, 1, 1, 2]
        src_feat = F.grid_sample(left_feat, src_grid, mode="bilinear",
                                 align_corners=True)[:, :, 0, 0]  # [P, C]
        src_feat = F.normalize(src_feat, dim=1)

        cx = proj[..., 0] - rox.to(proj.dtype).unsqueeze(1)
        cy = proj[..., 1] - roy.to(proj.dtype).unsqueeze(1)
        cand_grid = _pixel_to_grid(cx, cy, fw, fh, ps).unsqueeze(1)  # [P, 1, N, 2]
        cand_feat = F.grid_sample(right_feat, cand_grid, mode="bilinear",
                                  align_corners=True)[:, :, 0, :]     # [P, C, N]
        cand_feat = F.normalize(cand_feat, dim=1)
        scores = (src_feat.unsqueeze(-1) * cand_feat).sum(1)          # [P, N]

        # A candidate outside its crop was never featurized; exclude it rather
        # than let grid_sample's zero padding invent a similarity for it.
        sampled_ok = (cx >= 0) & (cx <= crop - 1) & (cy >= 0) & (cy <= crop - 1)
        return scores, sampled_ok


class EpipolarDinoCropMeasurer(EpipolarDinoCropMatcher):
    """EpipolarDinoCropMatcher plus in-graph triangulation (see triangulate.py)."""

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
