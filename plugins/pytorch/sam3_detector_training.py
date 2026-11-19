# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
SAM 3.1 detector fine-tuning engine, built on sam3.train.

Used by SAM3Trainer and by SAM3TrackerTrainer (when it also trains the
detector), either in-process on one GPU or as the per-rank entry point of a
multi-GPU run:

    python -m torch.distributed.run --nproc_per_node N \\
        sam3_detector_training.py params.json

Deliberately free of kwiver imports so DDP ranks can run it as a plain
script. The data arrives as a JSON list of images:

    [{"image": path, "group": str,
      "objects": [{"bbox": [x1, y1, x2, y2],
                   "polys": [[x, y, x, y, ...], ...]}]}]

in original image pixels. ``group`` names the video or folder an image came
from, so validation can hold out whole groups. An image with no objects is
a negative: SAM3 learns to report nothing there.

What is trained
---------------
The SAM 3.1 detector (``detector.*`` in the checkpoint) is the open-vocabulary
grounding model that finds every instance of a text query: a DETR-style
encoder/decoder over the image and prompt tokens, a dot-product scorer with a
presence head, a box head, and a mask head. It shares its vision backbone
with the tracker. We rebuild it as Sam3MultiplexDetector exactly as
build_sam3_multiplex_video_predictor does, keep the vision backbone and text
encoder frozen (so a tracker fine-tuned against this backbone stays valid, and
so it can be trained with or without one), and train the transformer,
scorer, geometry encoder and mask head with SAM 3's own training objective
from sam3.train, as in its roboflow fine-tuning recipe:

* one-to-one Hungarian matching (BinaryHungarianMatcherV2) plus the
  one-to-many auxiliary queries (BinaryOneToManyMatcher, weight 2);
* box L1 (x5) + GIoU (x2), and the presence-aware focal classification loss
  IABCEMdetr (x20, presence x20), on the final and every auxiliary decoder
  layer;
* the mask loss (focal x200 + dice x10, point-sampled) where the object has
  a polygon. Box-only annotations still train detection.

Every query is the single training text query ("fish" by default), so all
labels are merged into one class, matching how the SAM3 tracking pipelines
query the model.

The result is written into a copy of the source checkpoint (detector keys
outside the backbone only), so it loads anywhere ``sam3.1_weights.pt`` does.

Numerics
--------
The frozen backbone runs in fp16 without grad. The trained heads run under
fp16 autocast with a GradScaler (``amp``; Turing has no bf16), losses in fp32.
Multi-GPU gradient sync and the out-of-memory skip are shared with the
tracker engine (sam3_training_common).
"""

import json
import os
import random
import sys
import time

import numpy as np

if __name__ == "__main__":
    # Run as a script (a DDP rank), Python puts this file's directory first
    # on sys.path. That is viame/pytorch, whose torchvision subpackage then
    # shadows the real torchvision. Search it last instead.
    _here = os.path.dirname(os.path.abspath(__file__))
    sys.path[:] = [q for q in sys.path
                   if os.path.abspath(q or os.curdir) != _here] + [_here]

try:
    from viame.pytorch import sam3_training_common as common
except ImportError:  # run as a script without viame on the path
    import sam3_training_common as common

MODEL_SIZE = common.MODEL_SIZE
DETECTOR_PREFIX = "detector."
MAX_TARGETS = 200                 # the decoder has 200 queries
NUM_MASK_POINTS = 112 * 112       # point-sampled mask loss


def log(msg):
    print(f"[SAM3DetectorTraining] {msg}", flush=True)


# =============================================================================
# Model
# =============================================================================

def _matcher():
    from sam3.train.matcher import BinaryHungarianMatcherV2
    return BinaryHungarianMatcherV2(focal=True, cost_class=2.0, cost_bbox=5.0,
                                    cost_giou=2.0, alpha=0.25, gamma=2,
                                    stable=False)


def build_detector(ckpt_path, device, train=True):
    """
    Sam3MultiplexDetector as SAM 3.1 inference builds it, with the
    checkpoint's ``detector.*`` weights. With ``train`` the backbone is
    frozen and a matcher is attached so forward_grounding computes the
    one-to-one assignment the loss needs.
    """
    import pkg_resources
    from sam3 import model_builder as mb
    from sam3.model.sam3_multiplex_detector import Sam3MultiplexDetector
    from sam3.model.vl_combiner import SAM3VLBackboneTri
    common.patch_sdpa_for_pre_ampere()

    bpe = pkg_resources.resource_filename(
        "sam3", "assets/bpe_simple_vocab_16e6.txt.gz")
    tri_neck = mb._create_multiplex_tri_backbone(
        compile_mode=None, use_fa3=False, use_rope_real=False)
    backbone = SAM3VLBackboneTri(scalp=0, visual=tri_neck,
                                 text=mb._create_text_encoder(bpe))
    model = Sam3MultiplexDetector(
        num_feature_levels=1,
        backbone=backbone,
        transformer=mb._create_sam3_transformer(use_fa3=False),
        segmentation_head=mb._create_segmentation_head(use_fa3=False),
        semantic_segmentation_head=None,
        input_geometry_encoder=mb._create_geometry_encoder(),
        use_early_fusion=True,
        use_dot_prod_scoring=True,
        dot_prod_scoring=mb._create_dot_product_scoring(),
        supervise_joint_box_scores=True,
        is_multiplex=True,
        matcher=_matcher() if train else None,
    )

    full = common.load_full_checkpoint(ckpt_path)
    common.require_sam31(full, ckpt_path, "detector")
    state = {k[len(DETECTOR_PREFIX):]: v for k, v in full.items()
             if k.startswith(DETECTOR_PREFIX)}
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if not k.endswith("num_batches_tracked")]
    if missing or unexpected:
        raise RuntimeError(
            f"Detector weight load mismatch: {len(missing)} missing "
            f"(e.g. {missing[:5]}), {len(unexpected)} unexpected "
            f"(e.g. {unexpected[:5]})")

    for q in model.backbone.parameters():
        q.requires_grad_(False)
    model.to(device)
    model.train(train)
    model.backbone.eval()
    return model


def detector_state_dict(model):
    """Trained detector weights (the frozen backbone is left out)."""
    return {k: v.detach().cpu() for k, v in model.state_dict().items()
            if not k.startswith("backbone.")}


def export_checkpoint(state, source_ckpt, out_path):
    """Save a full SAM 3.1 checkpoint carrying the fine-tuned detector."""
    return common.write_into_checkpoint(source_ckpt, out_path,
                                        DETECTOR_PREFIX, state,
                                        skip=("backbone.",))


def build_loss(distributed):
    """SAM 3's detection objective (sam3.train), per the fine-tune recipe."""
    from sam3.train.loss.loss_fns import Boxes, IABCEMdetr, Masks
    from sam3.train.loss.sam3_loss import Sam3LossWrapper
    from sam3.train.matcher import BinaryOneToManyMatcher
    return Sam3LossWrapper(
        loss_fns_find=[
            Boxes(weight_dict={"loss_bbox": 5.0, "loss_giou": 2.0}),
            IABCEMdetr(weak_loss=False,
                       weight_dict={"loss_ce": 20.0, "presence_loss": 20.0},
                       pos_weight=10.0, alpha=0.25, gamma=2,
                       use_presence=True, pos_focal=False,
                       pad_n_queries=MAX_TARGETS, pad_scale_pos=1.0),
            Masks(weight_dict={"loss_mask": 200.0, "loss_dice": 10.0},
                  focal_alpha=0.25, focal_gamma=2.0, compute_aux=False,
                  num_sample_points=NUM_MASK_POINTS, oversample_ratio=3.0,
                  importance_sample_ratio=0.75),
        ],
        # "global" averages the box count across ranks, which needs a
        # process group.
        normalization="global" if distributed else "local",
        matcher=_matcher(),
        o2m_matcher=BinaryOneToManyMatcher(alpha=0.3, threshold=0.4, topk=4),
        o2m_weight=2.0,
        use_o2m_matcher_on_o2m_aux=False,
        scale_by_find_batch_size=True,
    )


# =============================================================================
# Data
# =============================================================================

def prepare_images(spec, label="train"):
    """Drop degenerate boxes and over-full images; report what is left."""
    out, n_obj, n_poly, n_neg, n_cap = [], 0, 0, 0, 0
    for item in spec:
        objs = []
        for o in item.get("objects", []):
            x1, y1, x2, y2 = (float(v) for v in o["bbox"])
            if x2 - x1 < 1 or y2 - y1 < 1:
                continue
            polys = [p for p in o.get("polys") or [] if len(p) >= 6]
            objs.append({"bbox": [x1, y1, x2, y2], "polys": polys})
        if len(objs) > MAX_TARGETS:
            n_cap += 1
            objs = sorted(objs, key=lambda o: -(o["bbox"][2] - o["bbox"][0])
                          * (o["bbox"][3] - o["bbox"][1]))[:MAX_TARGETS]
        n_obj += len(objs)
        n_poly += sum(1 for o in objs if o["polys"])
        n_neg += not objs
        out.append({"image": item["image"],
                    "group": item.get("group") or os.path.dirname(
                        item["image"]),
                    "objects": objs})
    log(f"{label}: {len(out)} images ({n_neg} without objects), {n_obj} "
        f"objects, {n_poly} with polygons")
    if n_cap:
        log(f"{label}: {n_cap} images had more than {MAX_TARGETS} objects; "
            "kept the largest")
    return out


class ImageDataset:
    """
    Images as SAM 3.1 inference sees them -- squashed to MODEL_SIZE,
    mean/std 0.5 -- turned into sam3.train Datapoints with one find query.

    Training augmentation: horizontal flip, brightness/contrast, and a random
    zoom (a crop scaled up, or the whole frame scaled down onto a padded
    canvas) so the model sees fish at more sizes than the data has.
    """

    def __init__(self, items, text_query, train=True):
        self.items = items
        self.text_query = text_query
        self.train = train

    def __len__(self):
        return len(self.items)

    @staticmethod
    def _zoom(rng, w, h):
        """Source crop (x0, y0, cw, ch) and its placement in the canvas."""
        S = MODEL_SIZE
        r = rng.random()
        if r < 0.25:                                 # zoom in
            s = rng.uniform(0.55, 1.0)
            cw, ch = w * s, h * s
            x0, y0 = rng.uniform(0, w - cw), rng.uniform(0, h - ch)
            return (x0, y0, cw, ch), (0.0, 0.0, float(S), float(S))
        if r < 0.45:                                 # zoom out
            s = rng.uniform(0.5, 1.0)
            ow, oh = S * s, S * s
            ox, oy = rng.uniform(0, S - ow), rng.uniform(0, S - oh)
            return (0.0, 0.0, float(w), float(h)), (ox, oy, ow, oh)
        return (0.0, 0.0, float(w), float(h)), (0.0, 0.0, float(S), float(S))

    def __getitem__(self, idx):
        import cv2
        import torch
        from PIL import Image
        from sam3.train.data.sam3_image_dataset import (
            Datapoint, FindQueryLoaded, Image as SamImage, InferenceMetadata,
            Object)

        item = self.items[idx]
        rng = random.Random() if self.train else random.Random(idx)
        S = MODEL_SIZE
        img = Image.open(item["image"]).convert("RGB")
        w, h = img.size

        (x0, y0, cw, ch), (ox, oy, ow, oh) = (
            self._zoom(rng, w, h) if self.train else
            ((0.0, 0.0, float(w), float(h)), (0.0, 0.0, float(S), float(S))))
        sx, sy = ow / cw, oh / ch
        flip = self.train and rng.random() < 0.5

        canvas = np.zeros((S, S, 3), dtype=np.uint8)
        crop = img.crop((int(round(x0)), int(round(y0)),
                         int(round(x0 + cw)), int(round(y0 + ch))))
        iw, ih = max(1, int(round(ow))), max(1, int(round(oh)))
        ix, iy = int(round(ox)), int(round(oy))
        canvas[iy:iy + ih, ix:ix + iw] = np.asarray(crop.resize((iw, ih)))[
            :S - iy, :S - ix]
        if flip:
            canvas = canvas[:, ::-1]

        def tx(x):
            x = ox + (x - x0) * sx
            return S - x if flip else x

        def ty(y):
            return oy + (y - y0) * sy

        lo_x, hi_x = (S - (ox + ow), S - ox) if flip else (ox, ox + ow)
        objects = []
        for o in item["objects"]:
            x1, y1, x2, y2 = o["bbox"]
            bx = sorted((tx(x1), tx(x2)))
            by = (ty(y1), ty(y2))
            full_area = (bx[1] - bx[0]) * (by[1] - by[0])
            cx1, cx2 = max(bx[0], lo_x), min(bx[1], hi_x)
            cy1, cy2 = max(by[0], oy), min(by[1], oy + oh)
            if cx2 - cx1 < 2 or cy2 - cy1 < 2:
                continue
            if (cx2 - cx1) * (cy2 - cy1) < 0.4 * full_area:
                continue  # mostly cropped away; neither positive nor negative
            seg = None
            if o["polys"]:
                m = np.zeros((S, S), dtype=np.uint8)
                for p in o["polys"]:
                    pts = np.array(p, dtype=np.float64).reshape(-1, 2)
                    pts = np.stack([[tx(x) for x in pts[:, 0]],
                                    [ty(y) for y in pts[:, 1]]], 1)
                    cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 1)
                seg = torch.from_numpy(m).bool()
            box = torch.tensor([(cx1 + cx2) / 2 / S, (cy1 + cy2) / 2 / S,
                                (cx2 - cx1) / S, (cy2 - cy1) / S],
                               dtype=torch.float32)
            objects.append(Object(bbox=box, area=float(box[2] * box[3]),
                                  object_id=len(objects), segment=seg))

        x = torch.from_numpy(np.ascontiguousarray(canvas)).permute(
            2, 0, 1).float() / 255.0
        if self.train:
            b, c = rng.uniform(0.8, 1.2), rng.uniform(0.8, 1.2)
            mean = x.mean()
            x = (((x - mean) * c + mean) * b).clamp(0, 1)
        x = (x - 0.5) / 0.5

        query = FindQueryLoaded(
            query_text=self.text_query, image_id=0,
            object_ids_output=list(range(len(objects))), is_exhaustive=True,
            inference_metadata=InferenceMetadata(
                coco_image_id=idx, original_image_id=idx,
                original_category_id=1, original_size=(h, w), object_id=0,
                frame_index=0))
        return Datapoint(find_queries=[query],
                         images=[SamImage(data=x, objects=objects,
                                          size=(S, S))])


def collate(batch):
    import torch
    from sam3.train.data.collator import collate_fn_api
    out = collate_fn_api(batch, dict_key="all", with_seg_masks=True)["all"]
    # A batch with no objects (all negatives) collates its masks to a 1-D
    # empty tensor, which the mask loss rejects; give it the [0, H, W] shape.
    h, w = out.img_batch.shape[-2:]
    for t in out.find_targets:
        if t.segments is not None and t.segments.dim() != 3:
            t.segments = torch.zeros((0, h, w), dtype=torch.bool)
            t.is_valid_segment = torch.zeros((0,), dtype=torch.bool)
    return out


# =============================================================================
# Forward + loss
# =============================================================================

def _to_device(obj, device, fields):
    import torch
    for f in fields:
        v = getattr(obj, f, None)
        if torch.is_tensor(v):
            setattr(obj, f, v.to(device, non_blocking=True))


def _float_features(x):
    import torch
    from sam3.model.data_misc import NestedTensor
    if isinstance(x, NestedTensor):
        return NestedTensor(_float_features(x.tensors), x.mask)
    if torch.is_tensor(x):
        return x.float() if x.is_floating_point() else x
    if isinstance(x, (list, tuple)):
        return type(x)(_float_features(v) for v in x)
    if isinstance(x, dict):
        return {k: _float_features(v) for k, v in x.items()}
    return x


def batch_forward(model, batch, device, amp=True):
    """
    The frozen backbone (fp16, no grad) and text encoder, then the trained
    grounding heads -- the training path of Sam3Image.forward with the
    backbone split out. Returns the stage outputs and the targets.
    """
    import torch
    from sam3.model.geometry_encoders import Prompt
    from sam3.model.model_misc import SAM3Output

    img = batch.img_batch.to(device, non_blocking=True)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16,
                                         enabled=device.type == "cuda"):
        backbone_out = model.backbone.forward_image(
            img, need_sam3_out=True, need_interactive_out=False,
            need_propagation_out=False)
        backbone_out.update(model.backbone.forward_text(
            batch.find_text_batch, device=device))
    # fp32 features in; autocast casts them down again where it applies.
    backbone_out = _float_features(backbone_out)
    backbone_out["img_batch_all_stages"] = img

    find_input = batch.find_inputs[0]
    _to_device(find_input, device, ("img_ids", "text_ids", "input_boxes",
                                    "input_boxes_mask", "input_boxes_label",
                                    "input_points", "input_points_mask"))
    find_target = batch.find_targets[0]
    _to_device(find_target, device, ("num_boxes", "boxes", "boxes_padded",
                                     "is_exhaustive", "segments",
                                     "is_valid_segment", "object_ids",
                                     "object_ids_padded", "repeated_boxes"))
    prompt = Prompt(box_embeddings=find_input.input_boxes,
                    box_mask=find_input.input_boxes_mask,
                    box_labels=find_input.input_boxes_label)
    with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        out = model.forward_grounding(backbone_out=backbone_out,
                                      find_input=find_input,
                                      find_target=find_target,
                                      geometric_prompt=prompt)
    for k in ("prev_encoder_out", "encoder_hidden_states"):
        out.pop(k, None)
    stages = SAM3Output(iter_mode=SAM3Output.IterMode.LAST_STEP_PER_STAGE)
    stages.append([common.to_float(out)])
    return stages, out, find_target


def train_loss(model, criterion, batch, device, amp=True):
    from sam3.train.loss.loss_fns import CORE_LOSS_KEY
    stages, _, target = batch_forward(model, batch, device, amp)
    losses = criterion(stages, [model.back_convert(target)])
    core = losses.pop(CORE_LOSS_KEY)
    parts = {k: float(v) for k, v in losses.items()
             if "_aux" not in k and not k.endswith("_o2m")}
    return core, parts, int(target.num_boxes.sum())


# =============================================================================
# Evaluation
# =============================================================================

def _iou_matrix(a, b):
    """IoU between [N,4] and [M,4] xyxy arrays."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    area_a = (a[:, 2:] - a[:, :2]).prod(-1)
    area_b = (b[:, 2:] - b[:, :2]).prod(-1)
    return inter / np.maximum(area_a[:, None] + area_b[None] - inter, 1e-9)


def average_precision(dets, n_gt, iou_thr):
    """
    Single-class AP (101-point, COCO style). ``dets`` holds one
    (scores, ious_with_gt) pair per image.
    """
    scores, tp = [], []
    for sc, ious in dets:
        order = np.argsort(-sc)
        taken = np.zeros(ious.shape[1], dtype=bool)
        for i in order:
            hit = False
            if ious.shape[1]:
                cand = np.where(~taken & (ious[i] >= iou_thr))[0]
                if len(cand):
                    j = cand[np.argmax(ious[i, cand])]
                    taken[j] = hit = True
            scores.append(sc[i])
            tp.append(hit)
    if n_gt == 0:
        return float("nan")
    if not scores:
        return 0.0
    order = np.argsort(-np.array(scores), kind="stable")
    tp = np.array(tp)[order]
    ctp = np.cumsum(tp)
    recall = ctp / n_gt
    precision = ctp / np.arange(1, len(tp) + 1)
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    rs = np.linspace(0, 1, 101)
    idx = np.searchsorted(recall, rs, side="left")
    return float(np.mean([precision[i] if i < len(precision) else 0.0
                          for i in idx]))


def best_f1(dets, n_gt, iou_thr=0.5):
    """Best F1 over score thresholds, and the threshold that gives it."""
    scores, tp = [], []
    for sc, ious in dets:
        order = np.argsort(-sc)
        taken = np.zeros(ious.shape[1], dtype=bool)
        for i in order:
            hit = False
            if ious.shape[1]:
                cand = np.where(~taken & (ious[i] >= iou_thr))[0]
                if len(cand):
                    taken[cand[np.argmax(ious[i, cand])]] = hit = True
            scores.append(sc[i])
            tp.append(hit)
    if not scores or n_gt == 0:
        return 0.0, 0.5
    order = np.argsort(-np.array(scores), kind="stable")
    s = np.array(scores)[order]
    ctp = np.cumsum(np.array(tp)[order])
    f1 = 2 * ctp / (np.arange(1, len(s) + 1) + n_gt)
    i = int(np.argmax(f1))
    return float(f1[i]), float(s[i])


def evaluate(model, loader, device, amp=True, min_score=0.05):
    """
    Detection quality on held-out images: AP at IoU 0.5 and averaged over
    0.5:0.95 (single class), best F1, and the validation loss. Returns the
    per-rank raw detections too, so a multi-GPU run can pool them.
    """
    import torch
    from sam3.model.box_ops import box_cxcywh_to_xyxy

    model.eval()
    dets, n_gt = [], 0
    with torch.no_grad():
        for batch in loader:
            _, out, target = batch_forward(model, batch, device, amp)
            prob = out["pred_logits"].float().sigmoid().squeeze(-1)
            boxes = out["pred_boxes_xyxy"].float()
            gt = box_cxcywh_to_xyxy(target.boxes.view(-1, 4).float())
            counts = target.num_boxes.tolist()
            start = 0
            for b in range(prob.shape[0]):
                g = gt[start:start + counts[b]].cpu().numpy()
                start += counts[b]
                keep = prob[b] > min_score
                sc = prob[b][keep].cpu().numpy()
                bx = boxes[b][keep].cpu().numpy()
                dets.append((sc, _iou_matrix(bx, g)))
                n_gt += len(g)
    model.train()
    model.backbone.eval()
    return dets, n_gt


def summarize(dets, n_gt):
    aps = [average_precision(dets, n_gt, t)
           for t in np.arange(0.5, 0.96, 0.05)]
    f1, thr = best_f1(dets, n_gt)
    return dict(ap50=aps[0], ap75=aps[5], map=float(np.mean(aps)), f1=f1,
                f1_threshold=thr, n_images=len(dets), n_gt=n_gt)


def _gather(obj, distributed):
    if not distributed:
        return [obj]
    import torch.distributed as dist
    out = [None] * dist.get_world_size()
    dist.all_gather_object(out, obj)
    return out


# =============================================================================
# Inference
# =============================================================================

class DetectorPredictor:
    """
    Per-image SAM 3.1 detection with the model's own grounding head (no
    tracker, no Grounding DINO): every instance of each text query, with a
    box, score and mask. Preprocessing matches SAM 3.1 video inference.
    """

    def __init__(self, ckpt_path, device="cuda", amp=True):
        import torch
        self.device = torch.device(device)
        self.amp = bool(amp) and self.device.type == "cuda"
        self.model = build_detector(ckpt_path, self.device, train=False)
        self.model.eval()

    def predict(self, image_rgb, queries, threshold=0.3, max_detections=100,
                masks=True):
        """
        ``image_rgb`` is an HxWx3 uint8 array. Returns (boxes [N,4] xyxy
        pixels, scores [N], query index [N], masks [N,H,W] bool or None),
        sorted by score.
        """
        import torch
        import torch.nn.functional as F
        from PIL import Image
        from sam3.train.data.sam3_image_dataset import (
            Datapoint, FindQueryLoaded, Image as SamImage, InferenceMetadata)

        h, w = image_rgb.shape[:2]
        S = MODEL_SIZE
        x = torch.from_numpy(np.asarray(
            Image.fromarray(image_rgb).resize((S, S)))).permute(
            2, 0, 1).float() / 255.0
        x = (x - 0.5) / 0.5
        meta = InferenceMetadata(coco_image_id=0, original_image_id=0,
                                 original_category_id=0,
                                 original_size=(h, w), object_id=0,
                                 frame_index=0)
        dp = Datapoint(
            find_queries=[FindQueryLoaded(query_text=q, image_id=0,
                                          object_ids_output=[],
                                          is_exhaustive=True,
                                          inference_metadata=meta)
                          for q in queries],
            images=[SamImage(data=x, objects=[], size=(S, S))])
        with torch.no_grad():
            _, out, _ = batch_forward(self.model, collate([dp]), self.device,
                                      self.amp)
            prob = out["pred_logits"].float().sigmoid().squeeze(-1)
            boxes = out["pred_boxes_xyxy"].float()
            scale = torch.tensor([w, h, w, h], device=boxes.device,
                                 dtype=boxes.dtype)
            all_boxes, all_scores, all_q, all_masks = [], [], [], []
            for qi in range(len(queries)):
                keep = torch.nonzero(prob[qi] > threshold).flatten()
                keep = keep[prob[qi][keep].argsort(descending=True)]
                keep = keep[:max_detections]
                all_boxes.append((boxes[qi][keep] * scale).clamp(min=0))
                all_scores.append(prob[qi][keep])
                all_q.append(torch.full_like(keep, qi))
                if masks and "pred_masks" in out and len(keep):
                    m = out["pred_masks"][qi][keep].float()[:, None]
                    all_masks += [F.interpolate(mi[None], size=(h, w),
                                                mode="bilinear",
                                                align_corners=False)[0, 0] > 0
                                  for mi in m]
            sc = torch.cat(all_scores)
            order = sc.argsort(descending=True)
            bx = torch.cat(all_boxes)[order]
            bx[:, 0::2] = bx[:, 0::2].clamp(max=w)
            bx[:, 1::2] = bx[:, 1::2].clamp(max=h)
            mk = None
            if masks and all_masks:
                mk = torch.stack(all_masks)[order].cpu().numpy()
            return (bx.cpu().numpy(), sc[order].cpu().numpy(),
                    torch.cat(all_q)[order].cpu().numpy(), mk)


# =============================================================================
# Training loop
# =============================================================================

DEFAULTS = dict(
    max_iterations=3000, learning_rate=8e-5, weight_decay=0.1,
    warmup_iterations=200, batch_size=4, eval_every=500, eval_images=500,
    num_workers=4, seed=0, text_query="fish", amp=True, timeout=604800,
)


def run(params):
    """
    Train. ``params`` holds: train_spec / val_spec (paths to JSON image
    lists), sam_model_id (source 3.1 checkpoint), work_dir, output_model
    (path to write), plus the DEFAULTS keys. Works as a single process or as
    a torch.distributed.run rank.
    """
    import torch

    p = dict(DEFAULTS)
    p.update(params)
    seed = int(p["seed"])
    distributed, rank, world, device = common.setup_distributed(seed)
    is_main = rank == 0
    work = p["work_dir"]
    os.makedirs(work, exist_ok=True)
    t_start = time.time()
    amp = bool(p["amp"])

    with open(p["train_spec"]) as f:
        train_items = prepare_images(json.load(f), "train")
    val_items = []
    if p.get("val_spec") and os.path.exists(p["val_spec"]):
        with open(p["val_spec"]) as f:
            val_items = prepare_images(json.load(f), "val")
    if not any(it["objects"] for it in train_items):
        raise RuntimeError("No annotated training images for the SAM3 "
                           "detector.")

    model = build_detector(p["sam_model_id"], device, train=True)
    criterion = build_loss(distributed)
    params_ = [q for q in model.parameters() if q.requires_grad]
    if is_main:
        log(f"trainable parameters {sum(q.numel() for q in params_):,} / "
            f"{sum(q.numel() for q in model.parameters()):,}")
    opt = common.adamw(params_, float(p["learning_rate"]),
                       float(p["weight_decay"]))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    iters = int(p["max_iterations"])
    warm = int(p["warmup_iterations"])
    lr0 = float(p["learning_rate"])

    start_iter = 0
    last_path = os.path.join(work, "last.pt")
    if os.path.exists(last_path):
        st = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"], strict=False)
        opt.load_state_dict(st["opt"])
        scaler.load_state_dict(st["scaler"])
        start_iter = st["iter"]
        if is_main:
            log(f"resuming from iteration {start_iter}")

    text = str(p["text_query"])
    train_ds = ImageDataset(train_items, text, train=True)
    if distributed:
        sampler = torch.utils.data.DistributedSampler(
            train_ds, num_replicas=world, rank=rank, shuffle=True, seed=seed)
    else:
        sampler = torch.utils.data.RandomSampler(train_ds)
    workers = int(p["num_workers"])
    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=int(p["batch_size"]), sampler=sampler,
        num_workers=workers, collate_fn=collate, pin_memory=True,
        drop_last=True, persistent_workers=workers > 0)
    if is_main:
        log(f"{len(train_items)} training images; {len(val_items)} "
            f"validation images; batch {int(p['batch_size'])} x {world} GPUs")

    val_loader = None
    if val_items:
        n_eval = int(p["eval_images"])
        if n_eval and len(val_items) > n_eval:
            val_items = random.Random(123).sample(val_items, n_eval)
        val_ds = ImageDataset(val_items, text, train=False)
        sub = torch.utils.data.Subset(val_ds,
                                      list(range(rank, len(val_ds), world)))
        val_loader = torch.utils.data.DataLoader(
            sub, batch_size=1, shuffle=False, num_workers=min(2, workers),
            collate_fn=collate)

    def run_eval(it):
        dets, n_gt = evaluate(model, val_loader, device, amp)
        parts = _gather((dets, n_gt), distributed)
        r = None
        if is_main:
            all_dets = [d for part in parts for d in part[0]]
            r = summarize(all_dets, sum(part[1] for part in parts))
            log(f"eval iter {it}: " + json.dumps(r))
            with open(os.path.join(work, "eval.jsonl"), "a") as f:
                f.write(json.dumps(dict(iter=it, **r)) + "\n")
        return r

    best_f = os.path.join(work, "best.json")
    best_path = os.path.join(work, "best.pt")
    select_key = "map"

    def maybe_save_best(it, r):
        if not is_main:
            return
        best = json.load(open(best_f)) if os.path.exists(best_f) else None
        if best is None or r[select_key] > best[select_key]:
            common.atomic_save(dict(model=detector_state_dict(model), iter=it,
                                    eval=r), best_path)
            json.dump(dict(iter=it, **r), open(best_f, "w"))
            log(f"new best {select_key} {r[select_key]:.4f} at iteration {it}")

    eval_every = int(p["eval_every"])
    if val_loader is not None and start_iter == 0 and eval_every > 0:
        maybe_save_best(0, run_eval(0))  # the source model, as a floor

    it, epoch, n_oom, hist, gn = start_iter, 0, 0, [], float("nan")
    t_last = time.time()
    timeout = float(p["timeout"]) if p.get("timeout") else None
    opt.zero_grad(set_to_none=True)
    done = it >= iters
    while not done:
        if distributed:
            sampler.set_epoch(epoch)
        for batch in loader:
            for g in opt.param_groups:
                g["lr"] = common.cosine_lr(it, lr0, warm, iters)
            skip, parts = False, {}
            try:
                loss, parts, n_obj = train_loss(model, criterion, batch,
                                                device, amp)
                scaler.scale(loss).backward()
            except torch.OutOfMemoryError:
                skip = True
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                n_oom += 1
                log(f"rank {rank}: out of GPU memory; batch skipped "
                    f"({n_oom} so far)")
            stepped, step_gn = common.sync_and_step(
                params_, opt, scaler, skip, distributed, world, 0.1, device)
            it += 1
            if stepped:
                gn = step_gn
                hist.append(dict(loss=float(loss), **parts))

            if is_main and it % 20 == 0 and hist:
                keys = ("loss", "loss_ce", "presence_loss", "loss_bbox",
                        "loss_giou", "loss_mask", "loss_dice")
                h = {k: np.mean([x.get(k, np.nan) for x in hist])
                     for k in keys}
                hist = []
                dt = (time.time() - t_last) / 20
                t_last = time.time()
                log(f"iter {it}/{iters} " + " ".join(
                    f"{k.replace('loss_', '')} {h[k]:.4f}" for k in keys)
                    + f" gnorm {gn:.3f} lr "
                    f"{common.cosine_lr(it, lr0, warm, iters):.2e} "
                    f"scale {scaler.get_scale():.0f} {dt:.2f}s/it")
            if is_main and it % 500 == 0:
                common.atomic_save(dict(model=detector_state_dict(model),
                                        opt=opt.state_dict(),
                                        scaler=scaler.state_dict(), iter=it),
                                   last_path)
            late = common.out_of_time(t_start, timeout, distributed, device)
            if val_loader is not None and eval_every > 0 and (
                    it % eval_every == 0 or it >= iters or late):
                maybe_save_best(it, run_eval(it))
            if it >= iters or late:
                if late and is_main:
                    log("training timeout reached; stopping")
                done = True
                break
        epoch += 1

    if is_main:
        if val_loader is None or not os.path.exists(best_path):
            common.atomic_save(dict(model=detector_state_dict(model), iter=it),
                               best_path)
        st = torch.load(best_path, map_location="cpu", weights_only=False)
        export_checkpoint(st["model"], p["sam_model_id"], p["output_model"])
        log(f"exported iteration {st['iter']} weights to {p['output_model']}")
    if distributed:
        import torch.distributed as dist
        dist.barrier()
        dist.destroy_process_group()


def main():
    if len(sys.argv) != 2:
        print("usage: sam3_detector_training.py params.json", file=sys.stderr)
        sys.exit(2)
    with open(sys.argv[1]) as f:
        run(json.load(f))


if __name__ == "__main__":
    main()
