# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

from kwiver.vital.algo import RefineDetections
from kwiver.vital.types import DetectedObjectSet, Image, ImageContainer, Point2d

import numpy as np
import scriptconfig as scfg

from viame.pytorch.utilities import (
    report_cuda_errors,
    vital_config_update,
    register_vital_algorithm,
    parse_bool,
)
from viame.pytorch.rf_detr_detector import RFDETRDetector


class RFDETRRefinerConfig(scfg.DataConfig):
    """
    The configuration for :class:`RFDETRRefiner`.
    """
    weight = scfg.Value(None, help='Path to a trained RF-DETR checkpoint (.pt file)')
    model_size = scfg.Value('base', help='Model size: nano, small, medium, base, or large')
    num_channels = scfg.Value(3, help='Number of input channels; recovered from the checkpoint when present')
    resolution = scfg.Value(0, help=(
        'Input resolution ("1280" or "960x1728"); 0 = model-size default. '
        'Recovered from the checkpoint when present.'))
    device = scfg.Value('auto', help='Device to run on: auto, cpu, cuda, or cuda:N')
    segmentation = scfg.Value(False, help=(
        'Load a segmentation (mask) RF-DETR variant. Recovered from the '
        'checkpoint when present.'))
    keypoint_names = scfg.Value('', help=(
        'Comma-separated names for the keypoint head outputs. Empty = recover '
        'from the checkpoint, falling back to head,tail.'))
    add_masks = scfg.Value(True, help='Attach the segmentation head output to each box')
    add_keypoints = scfg.Value(True, help='Attach the keypoint head output to each box')
    overwrite_existing = scfg.Value(False, help=(
        'Recompute masks/keypoints for detections that already have them. When '
        'false, frames where every detection is already complete skip the model.'))
    keypoint_vis_thresh = scfg.Value(0.5, help=(
        'Minimum keypoint visibility (sigmoid) to attach a keypoint'))
    apply_query_delta = scfg.Value(False, help=(
        'Apply the learned per-slot reference-point delta on top of the input '
        'boxes. The model applies it to first-stage proposals; input boxes are '
        'already final, so it is off by default.'))
    fill_with_proposals = scfg.Value(True, help=(
        'Fill the query slots not used by input boxes with the model\'s own '
        'top-scoring proposals so the decoder self-attention sees the same '
        'query population it was trained with'))

    def __post_init__(self):
        super().__post_init__()


class RFDETRRefiner(RefineDetections):
    """
    Runs the segmentation and keypoint heads of an RF-DETR checkpoint on
    externally supplied boxes.

    The backbone is run once per frame; the input boxes are then injected as
    the decoder's reference points in place of the model's own two-stage
    proposals, so each box gets a mask and keypoints computed for exactly that
    box (the default config pins reference points across decoder layers).
    """

    def __init__(self):
        RefineDetections.__init__(self)
        self._kwiver_config = RFDETRRefinerConfig()
        self._detector = None

    def get_configuration(self):
        cfg = super(RefineDetections, self).get_configuration()
        for key, value in self._kwiver_config.items():
            cfg.set_value(key, str(value))
        return cfg

    @report_cuda_errors("RFDETRRefiner initialization")
    def set_configuration(self, cfg_in):
        cfg = self.get_configuration()
        vital_config_update(cfg, cfg_in)

        for key in self._kwiver_config.keys():
            self._kwiver_config[key] = str(cfg.get_value(key))

        self._add_masks = parse_bool(self._kwiver_config['add_masks'])
        self._add_keypoints = parse_bool(self._kwiver_config['add_keypoints'])
        self._overwrite = parse_bool(self._kwiver_config['overwrite_existing'])
        self._vis_thresh = float(self._kwiver_config['keypoint_vis_thresh'])
        self._apply_delta = parse_bool(self._kwiver_config['apply_query_delta'])
        self._fill = parse_bool(self._kwiver_config['fill_with_proposals'])

        # Reuse the detector's checkpoint/architecture recovery; the optimized
        # export path replaces the forward we need, so it stays off.
        self._detector = RFDETRDetector()
        for key in ('weight', 'model_size', 'num_channels', 'resolution',
                    'device', 'segmentation', 'keypoint_names'):
            self._detector._kwiver_config[key] = self._kwiver_config[key]
        self._detector._kwiver_config['optimize_inference'] = 'false'
        self._detector._build_model()

        net = self._detector._model.model.model
        self._keypoint_names = getattr(self._detector, '_keypoint_names', None)
        if self._add_masks and net.segmentation_head is None:
            print("[RFDETRRefiner] Checkpoint has no segmentation head; masks disabled")
            self._add_masks = False
        if self._add_keypoints and net.keypoint_head is None:
            print("[RFDETRRefiner] Checkpoint has no keypoint head; keypoints disabled")
            self._add_keypoints = False
        return True

    def check_configuration(self, cfg):
        return True

    def _needs_work(self, det):
        if self._overwrite:
            return True
        if self._add_masks and det.mask is None:
            return True
        if self._add_keypoints and not det.keypoints:
            return True
        return False

    @report_cuda_errors("RFDETRRefiner refinement")
    def refine(self, image_data, detections):
        import torch

        if detections is None or len(detections) == 0:
            return DetectedObjectSet()

        det_list = list(detections)
        # Query slots were trained in proposal-score order, so hand the
        # highest-confidence boxes the lowest slots.
        todo = sorted((i for i, d in enumerate(det_list) if self._needs_work(d)),
                      key=lambda i: -det_list[i].confidence)
        if not todo or not (self._add_masks or self._add_keypoints):
            return detections

        img = image_data.asarray()
        if img.ndim == 2:
            img = np.repeat(img[:, :, None], 3, axis=2)
        elif img.shape[2] == 4 and self._detector._num_channels == 3:
            img = img[:, :, :3]
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        img = np.ascontiguousarray(img)
        img_h, img_w = img.shape[:2]

        boxes = np.array([
            [det_list[i].bounding_box.min_x(), det_list[i].bounding_box.min_y(),
             det_list[i].bounding_box.max_x(), det_list[i].bounding_box.max_y()]
            for i in todo], dtype=np.float32)

        with torch.no_grad():
            masks, kps = self._run_heads(img, boxes)

        output = DetectedObjectSet()
        for det in det_list:
            output.add(det)

        for j, i in enumerate(todo):
            det = det_list[i]
            box = boxes[j]
            if masks is not None and (self._overwrite or det.mask is None):
                x1 = min(max(int(np.floor(box[0])), 0), max(img_w - 1, 0))
                y1 = min(max(int(np.floor(box[1])), 0), max(img_h - 1, 0))
                x2 = min(max(int(np.ceil(box[2])) + 1, x1 + 1), img_w)
                y2 = min(max(int(np.ceil(box[3])) + 1, y1 + 1), img_h)
                crop = np.ascontiguousarray(masks[j][y1:y2, x1:x2].astype(np.uint8))
                if crop.size and crop.any():
                    det.mask = ImageContainer(Image(crop))
            if kps is not None and (self._overwrite or not det.keypoints):
                if self._overwrite:
                    det.clear_keypoints()
                for k, name in enumerate(self._keypoint_names):
                    x, y, v = kps[j][k]
                    if v >= self._vis_thresh:
                        pt = Point2d()
                        pt.value = [float(x), float(y)]
                        det.add_keypoint(name, pt)
        return output

    def _run_heads(self, img, boxes_xyxy):
        """
        Backbone once, then decoder + heads with the given boxes as reference
        points. Returns (masks [N,H,W] bool or None, keypoints [N,K,3] or None)
        in original image coordinates.
        """
        import torch
        import torch.nn.functional as F
        import torchvision.transforms.functional as TF
        from rfdetr.utilities.tensors import nested_tensor_from_tensor_list
        from rfdetr.utilities.shapes import as_pair
        from rfdetr.models.math import inverse_sigmoid
        from rfdetr.models.transformer import gen_encoder_output_proposals
        from rfdetr.detr import _ensure_model_on_device

        rf = self._detector._model
        ctx = rf.model
        _ensure_model_on_device(ctx)
        net = ctx.model
        net.eval()
        tr = net.transformer
        device = ctx.device
        img_h, img_w = img.shape[:2]
        res_h, res_w = as_pair(ctx.resolution)

        tensor = TF.to_tensor(img).to(device)
        tensor = TF.resize(tensor, [res_h, res_w])
        tensor = TF.normalize(tensor, rf.means, rf.stds)
        samples = nested_tensor_from_tensor_list([tensor])

        features, poss = net.backbone(samples)
        srcs, masks = [], []
        for feat in features:
            src, mask = feat.decompose()
            srcs.append(src)
            masks.append(mask)

        # Flatten multi-level features exactly as Transformer.forward does.
        src_flatten, mask_flatten, pos_flatten = [], [], []
        spatial_shapes = torch.empty((len(srcs), 2), device=device, dtype=torch.long)
        spatial_shapes_hw = []
        for lvl, (src, pos, mask) in enumerate(zip(srcs, poss, masks)):
            _, _, h, w = src.shape
            spatial_shapes[lvl, 0] = h
            spatial_shapes[lvl, 1] = w
            spatial_shapes_hw.append((h, w))
            src_flatten.append(src.flatten(2).transpose(1, 2))
            pos_flatten.append(pos.flatten(2).transpose(1, 2))
            mask_flatten.append(mask.flatten(1))
        memory = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        pos_flatten = torch.cat(pos_flatten, 1)
        valid_ratios = torch.stack([tr.get_valid_ratio(m) for m in masks], 1)
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        # Input boxes -> normalized cxcywh in the model's frame. In reparam
        # mode reference points are plain cxcywh; otherwise they live in logit
        # space.
        b = torch.as_tensor(boxes_xyxy, device=device, dtype=memory.dtype)
        cx = (b[:, 0] + b[:, 2]) / 2 / img_w
        cy = (b[:, 1] + b[:, 3]) / 2 / img_h
        bw = (b[:, 2] - b[:, 0]) / img_w
        bh = (b[:, 3] - b[:, 1]) / img_h
        ref_in = torch.stack([cx, cy, bw, bh], -1).clamp(1e-4, 1 - 1e-4)

        proposals = None
        if self._fill and tr.two_stage:
            proposals = self._native_proposals(
                tr, memory, mask_flatten, spatial_shapes_hw, gen_encoder_output_proposals)

        nq = net.num_queries
        n_in = ref_in.shape[0]
        all_masks, all_kps = [], []
        for start in range(0, n_in, nq):
            ref = ref_in[start:start + nq]
            n = ref.shape[0]
            if proposals is not None and n < nq:
                # Keep our boxes in the top slots, model proposals behind them
                extra = proposals[:nq - n]
                ref = torch.cat([ref, extra], 0)
            ref = ref.unsqueeze(0)
            n_slots = ref.shape[1]

            tgt = net.query_feat.weight[:n_slots].unsqueeze(0)
            delta = net.refpoint_embed.weight[:n_slots].unsqueeze(0)
            if net.bbox_reparam:
                if self._apply_delta:
                    cxcy = delta[..., :2] * ref[..., 2:] + ref[..., :2]
                    wh = delta[..., 2:].exp() * ref[..., 2:]
                    ref = torch.cat([cxcy, wh], -1)
            else:
                ref = inverse_sigmoid(ref)
                if self._apply_delta:
                    ref = ref + delta

            hs, references = tr.decoder(
                tgt, memory,
                memory_key_padding_mask=mask_flatten,
                pos=pos_flatten,
                refpoints_unsigmoid=ref,
                level_start_index=level_start_index,
                spatial_shapes=spatial_shapes,
                valid_ratios=valid_ratios.to(memory.dtype),
                spatial_shapes_hw=spatial_shapes_hw,
            )
            if references.shape[0] != hs.shape[0]:
                references = references.expand(hs.shape[0], -1, -1, -1)

            if self._add_masks:
                logits = net.segmentation_head(
                    features[0].tensors, hs, samples.tensors.shape[-2:])[-1]
                logits = logits[0, :n].unsqueeze(1)
                m = F.interpolate(logits, size=(img_h, img_w), mode='bilinear',
                                  align_corners=False)[:, 0] > 0.0
                all_masks.append(m.cpu().numpy())

            if self._add_keypoints:
                kp = net._compute_keypoints(hs, references)[-1][0, :n]
                xy = kp[..., :2] * torch.tensor([img_w, img_h], device=device, dtype=kp.dtype)
                vis = kp[..., 2:3].sigmoid()
                all_kps.append(torch.cat([xy, vis], -1).cpu().numpy())

        masks_out = np.concatenate(all_masks, 0) if all_masks else None
        kps_out = np.concatenate(all_kps, 0) if all_kps else None
        return masks_out, kps_out

    @staticmethod
    def _native_proposals(tr, memory, mask_flatten, spatial_shapes_hw, gen_proposals):
        """Top-scoring first-stage boxes (cxcywh, reparam or logit space to match
        the transformer), ordered as the model would order its own queries."""
        import torch
        output_memory, output_proposals = gen_proposals(
            memory, mask_flatten, spatial_shapes_hw, unsigmoid=not tr.bbox_reparam)
        out = tr.enc_output_norm[0](tr.enc_output[0](output_memory))
        cls = tr.enc_out_class_embed[0](out)
        if tr.bbox_reparam:
            d = tr.enc_out_bbox_embed[0](out)
            cxcy = d[..., :2] * output_proposals[..., 2:] + output_proposals[..., :2]
            wh = d[..., 2:].exp() * output_proposals[..., 2:]
            coords = torch.cat([cxcy, wh], -1)
        else:
            coords = (tr.enc_out_bbox_embed[0](out) + output_proposals).sigmoid()
        topk = min(tr.num_queries, cls.shape[1])
        idx = torch.topk(cls.max(-1)[0], topk, dim=1)[1]
        return torch.gather(coords, 1, idx.unsqueeze(-1).repeat(1, 1, 4))[0]


def __vital_algorithm_register__():
    register_vital_algorithm(
        RFDETRRefiner, "rf_detr",
        "Run RF-DETR segmentation/keypoint heads on existing boxes"
    )
