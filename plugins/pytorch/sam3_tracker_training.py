# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
SAM 3.1 tracker fine-tuning engine.

Used by SAM3TrackerTrainer (sam3_trainer.py), either in-process on one GPU
or as the per-rank entry point of a multi-GPU run:

    python -m torch.distributed.run --nproc_per_node N \\
        sam3_tracker_training.py params.json

Deliberately free of kwiver imports so DDP ranks can run it as a plain
script. The data arrives as a JSON-serializable spec built by the trainer:

    {"name": str,
     "frames": {frame_id: image_path},
     "objects": {frame_id: [{"tid": int, "bbox": [x1, y1, x2, y2],
                             "polys": [[x, y, x, y, ...], ...]}]}}

with coordinates in original image pixels.

What is trained
---------------
In SAM 3.1 the tracker (``tracker.model.*`` in the checkpoint) has no
backbone of its own; it shares the detector's vision backbone
(``detector.backbone.vision_backbone.*``). We build the multiplex tracker
standalone with a copy of that backbone, keep the backbone frozen (so the
text-query detector is unaffected), and train every tracker weight --
memory attention, memory encoder, the propagation and interactive mask
decoders, prompt encoder and object-pointer heads -- with the SAM2 video
objective via the model's own training forward
(``prepare_prompt_inputs`` + ``forward_tracking``): the tracker is prompted
with a GT mask, box or points on a conditioning frame, must propagate
through the clip, and objects that appear later are introduced at
transition frames exactly as new detections are at inference. Losses are
SAM2's MultiStepMultiMasksAndIous: focal (x20) + dice + L1 IoU-prediction
+ object-score BCE.

The result is written back into a copy of the source checkpoint (tracker
keys only), so it loads anywhere ``sam3.1_weights.pt`` does.

Numerics
--------
* The frozen backbone runs in fp16 autocast; it is most of the compute.
* The tracker heads overflow in fp16 (NaN IoU/object scores), so they run
  in fp32, except the memory-attention layers, which are fp16 inside an
  activation-checkpointed wrapper. That gives fp16 speed at fp32 stability.
* Multi-GPU sync all-reduces the *scaled* gradients before ``unscale_`` so
  every rank makes the same skip-on-inf decision; unscaling first lets a
  rank step on another rank's inf gradients and poison the weights.
* A clip that runs out of GPU memory is skipped on every rank together.
"""

import contextlib
import functools
import json
import math
import os
import random
import sys
import time
from types import SimpleNamespace

import numpy as np

MODEL_SIZE = 1008                 # SAM 3.1 input resolution
LOSS_RES = 504                    # loss resolution (memory bound)
LOSS_W = {"mask": 20.0, "dice": 1.0, "iou": 1.0, "class": 1.0}

TRACKER_PREFIX = "tracker.model."
BACKBONE_SRC_PREFIX = "detector.backbone.vision_backbone."
BACKBONE_DST_PREFIX = "backbone.vision_backbone."


def log(msg):
    print(f"[SAM3TrackerTraining] {msg}", flush=True)


# =============================================================================
# Checkpoint conversion
# =============================================================================

def load_full_checkpoint(path):
    import torch
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]
    return ckpt


def is_sam31_checkpoint(state):
    return any(k.startswith("tracker.model.") for k in state)


def full_to_tracker_state(full):
    out = {}
    for k, v in full.items():
        if k.startswith(TRACKER_PREFIX):
            out[k[len(TRACKER_PREFIX):]] = v
        elif k.startswith(BACKBONE_SRC_PREFIX):
            out[BACKBONE_DST_PREFIX + k[len(BACKBONE_SRC_PREFIX):]] = v
    return out


def tracker_to_full_state(tracker_state, full_template):
    """Write tracker weights (not the backbone) into a full checkpoint."""
    out = dict(full_template)
    for k, v in tracker_state.items():
        if k.startswith("backbone."):
            continue
        fk = TRACKER_PREFIX + k
        if fk not in out or out[fk].shape != v.shape:
            raise ValueError(f"tracker key {k} does not match the source "
                             "checkpoint")
        out[fk] = v.detach().to(out[fk].dtype).cpu().contiguous()
    return out


def export_checkpoint(tracker_state, source_ckpt, out_path):
    """Save a full SAM 3.1 checkpoint carrying the fine-tuned tracker."""
    import torch
    base = load_full_checkpoint(source_ckpt)
    full = tracker_to_full_state(tracker_state, base)
    for k in base:
        if not k.startswith("tracker.") and not torch.equal(full[k], base[k]):
            raise RuntimeError(f"non-tracker weight {k} changed on export")
    tmp = out_path + ".tmp"
    torch.save(full, tmp)
    os.replace(tmp, out_path)
    return out_path


# =============================================================================
# Model
# =============================================================================

def patch_sdpa_for_pre_ampere():
    """
    sam3/model/decoder.py forces sdpa_kernel(FLASH_ATTENTION), which has no
    kernel before Ampere. Same workaround as
    sam3_utilities._patch_sam3_sdpa_for_pre_ampere, kept local so ranks need
    no kwiver imports.
    """
    import torch
    try:
        if torch.cuda.get_device_capability(0)[0] >= 8:
            return
    except Exception:
        return
    from sam3.model import decoder as _dec

    @contextlib.contextmanager
    def _noop(*args, **kwargs):
        yield
    _dec.sdpa_kernel = _noop


def enable_memory_attention_checkpointing(model, fp16=True):
    """
    Activation-checkpoint each memory-attention layer, optionally in fp16
    with fp32 outputs. (The builder's own ``use_act_checkpoint`` path fails
    on these layers' *args signature.)
    """
    import torch
    from torch.utils.checkpoint import checkpoint

    def to_float(x):
        if torch.is_tensor(x) and x.dtype == torch.float16:
            return x.float()
        if isinstance(x, (tuple, list)):
            return type(x)(to_float(v) for v in x)
        if isinstance(x, dict):
            return {k: to_float(v) for k, v in x.items()}
        return x

    for layer in model.transformer.encoder.layers:
        orig = layer.forward

        def run(*args, __orig=orig, **kwargs):
            with torch.autocast("cuda", dtype=torch.float16, enabled=fp16):
                return to_float(__orig(*args, **kwargs))

        @functools.wraps(orig)
        def fwd(*args, __run=run, __layer=layer, **kwargs):
            if not (__layer.training and torch.is_grad_enabled()):
                return __run(*args, **kwargs)
            return checkpoint(__run, *args, use_reentrant=False, **kwargs)
        layer.forward = fwd


def build_trainable_tracker(ckpt_path, device, rank=0, seed=0):
    from sam3.model_builder import build_sam3_multiplex_video_model
    patch_sdpa_for_pre_ampere()

    model = build_sam3_multiplex_video_model(
        checkpoint_path=None, load_from_HF=False, multiplex_count=16,
        use_fa3=False, use_rope_real=False, strict_state_dict_loading=False,
        device="cpu", compile=False)

    full = load_full_checkpoint(ckpt_path)
    if not is_sam31_checkpoint(full):
        raise RuntimeError(
            f"{ckpt_path} is not a SAM 3.1 checkpoint (no tracker.model.* "
            "keys). The SAM3 tracker trainer fine-tunes the 3.1 multiplex "
            "tracker; point sam_model_id at sam3.1_weights.pt.")
    missing, unexpected = model.load_state_dict(full_to_tracker_state(full),
                                                strict=False)
    missing = [k for k in missing if not k.endswith("num_batches_tracked")]
    if missing or unexpected:
        raise RuntimeError(
            f"Tracker weight load mismatch: {len(missing)} missing "
            f"(e.g. {missing[:5]}), {len(unexpected)} unexpected "
            f"(e.g. {unexpected[:5]})")

    # Training-time prompt sampling (inference leaves these at 0, meaning
    # "condition on a mask"). Mostly masks, which is how the pipeline seeds
    # tracks from detections, plus boxes and clicks for interactive use.
    model.prob_to_use_pt_input_for_train = 0.3
    model.prob_to_use_box_input_for_train = 0.5
    model.num_frames_to_correct_for_train = 2
    model.rand_frames_to_correct_for_train = True
    model.num_correction_pt_per_frame = 3
    model.num_init_cond_frames_for_train = 1
    model.rand_init_cond_frames_for_train = True
    # The model seeds its prompt RNGs with 42; vary them per rank.
    model.rng = np.random.default_rng(1000 + seed * 10 + rank)
    model.rng2 = np.random.default_rng(2000 + seed * 10 + rank)

    for p in model.backbone.parameters():
        p.requires_grad_(False)
    enable_memory_attention_checkpointing(model, fp16=True)
    model.to(device)
    return model


def tracker_state_dict(model):
    return {k: v.detach().cpu() for k, v in model.state_dict().items()
            if not k.startswith("backbone.")}


# =============================================================================
# Data
# =============================================================================

def _runs(ids):
    runs, cur = [], [ids[0]]
    for i in ids[1:]:
        if i == cur[-1] + 1:
            cur.append(i)
        else:
            runs.append(cur)
            cur = [i]
    runs.append(cur)
    return runs


def prepare_sequences(spec, allow_box_targets=False, label="train"):
    """
    Normalize a spec into sequences with integer frame ids and decide, per
    sequence, whether masks come from polygons or (if allowed) boxes.

    Frames where some object lacks a polygon are dropped from polygon
    sequences: rasterizing only the others would teach "that fish is gone".
    """
    seqs = []
    n_box, n_skip = 0, 0
    for s in spec:
        frames = {int(k): v for k, v in s["frames"].items()}
        objs = {int(k): v for k, v in s["objects"].items() if v}
        objs = {k: v for k, v in objs.items() if k in frames}
        if not objs:
            continue
        has_poly = any(o["polys"] for v in objs.values() for o in v)
        if has_poly:
            target = "poly"
            objs = {k: v for k, v in objs.items()
                    if all(o["polys"] for o in v)}
        elif allow_box_targets:
            target = "box"
            n_box += 1
        else:
            n_skip += 1
            continue
        if objs:
            seqs.append(dict(name=s["name"], frames=frames, objects=objs,
                             target=target))
    if n_skip:
        log(f"{label}: skipped {n_skip} sequences with no polygons (set "
            "box_targets = true to train on boxes, which teaches box-shaped "
            "masks)")
    if n_box:
        log(f"{label}: {n_box} sequences use box-shaped mask targets")
    return seqs


class ClipDataset:
    """
    Multi-object clips of consecutive annotated frames. Each item holds the
    frames normalized like SAM3 inference (squashed to MODEL_SIZE, mean/std
    0.5), [T, N, S, S] masks, and the visible objects per frame.
    """

    def __init__(self, seqs, clip_len=8, max_objects=8, train=True,
                 first_frame_objects_only=False, stride=None,
                 max_clips=None, seed=0):
        self.seqs = seqs
        self.clip_len = clip_len
        self.max_objects = max_objects
        self.train = train
        self.first_only = first_frame_objects_only
        self.clips = []
        stride = stride or max(1, clip_len // 2)
        for si, s in enumerate(seqs):
            for run in _runs(sorted(s["objects"])):
                if len(run) < 2:
                    continue
                last = max(1, len(run) - clip_len + 1)
                for st in range(0, last, stride):
                    self.clips.append((si, run[st:st + clip_len * 2]))
        if max_clips and len(self.clips) > max_clips:
            self.clips = random.Random(seed).sample(self.clips, max_clips)

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        import cv2
        import torch
        from PIL import Image

        si, run = self.clips[idx]
        seq = self.seqs[si]
        rng = random.Random() if self.train else random.Random(idx)

        step = 1
        if (self.train and len(run) >= 2 * self.clip_len - 1
                and rng.random() < 0.25):
            step = 2  # larger motion
        fids = run[: (self.clip_len - 1) * step + 1: step]
        if self.train and rng.random() < 0.5:
            fids = fids[::-1]  # temporal reversal
        T = len(fids)

        first = [o["tid"] for o in seq["objects"][fids[0]]]
        rng.shuffle(first)
        order = list(first)
        if not self.first_only:
            later = []
            for f in fids[1:]:
                for o in seq["objects"][f]:
                    if o["tid"] not in order and o["tid"] not in later:
                        later.append(o["tid"])
            rng.shuffle(later)
            order += later
        order = order[: self.max_objects]
        index = {t: i for i, t in enumerate(order)}
        N = max(1, len(order))

        S = MODEL_SIZE
        imgs = np.empty((T, S, S, 3), dtype=np.uint8)
        masks = np.zeros((T, N, S, S), dtype=np.uint8)
        for t, f in enumerate(fids):
            img = Image.open(seq["frames"][f]).convert("RGB")
            w, h = img.size
            imgs[t] = np.asarray(img.resize((S, S)))
            sx, sy = S / w, S / h
            for o in seq["objects"][f]:
                if o["tid"] not in index:
                    continue
                m = masks[t, index[o["tid"]]]
                if seq["target"] == "box":
                    x1, y1, x2, y2 = o["bbox"]
                    m[max(int(round(y1 * sy)), 0):max(int(round(y2 * sy)), 0),
                      max(int(round(x1 * sx)), 0):max(int(round(x2 * sx)), 0)] = 1
                else:
                    for p in o["polys"]:
                        pts = np.array(p, dtype=np.float64).reshape(-1, 2)
                        pts[:, 0] *= sx
                        pts[:, 1] *= sy
                        cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 1)
        if self.train and rng.random() < 0.5:
            imgs = imgs[:, :, ::-1]
            masks = masks[:, :, :, ::-1]

        x = torch.from_numpy(np.ascontiguousarray(imgs)).permute(
            0, 3, 1, 2).float() / 255.0
        if self.train:
            b, c = rng.uniform(0.8, 1.2), rng.uniform(0.8, 1.2)
            mean = x.mean()
            x = (((x - mean) * c + mean) * b).clamp(0, 1)
        x = (x - 0.5) / 0.5
        masks = torch.from_numpy(np.ascontiguousarray(masks)).bool()
        visible = {t: set(torch.nonzero(masks[t].flatten(1).any(1))
                          .flatten().tolist()) for t in range(T)}
        return dict(images=x, masks=masks, visible=visible,
                    name=seq["name"], fids=fids)


def _collate_one(batch):
    return batch[0]


# =============================================================================
# Forward + loss
# =============================================================================

def make_input(sample, device):
    """The duck-typed BatchedDatapoint the tracker's training forward needs."""
    import torch
    from sam3.model.data_misc import NestedTensor
    T = sample["images"].shape[0]
    N = sample["masks"].shape[1]
    return SimpleNamespace(
        img_batch=NestedTensor(sample["images"].to(device, non_blocking=True),
                               None),
        find_inputs=[SimpleNamespace(img_ids=torch.tensor([t], device=device))
                     for t in range(T)],
        find_targets=[SimpleNamespace(
            segments=sample["masks"][t].to(device, non_blocking=True),
            num_boxes=torch.ones(N, device=device)) for t in range(T)],
        visible_objects_per_frame={k: set(v)
                                   for k, v in sample["visible"].items()},
    )


def compute_backbone(model, imgs, chunk=2):
    """
    Frozen backbone in fp16 without grad, then the trainable high-res
    projections (conv_s0/conv_s1 belong to the mask decoders) with grad,
    mirroring VideoTrackingMultiplex.forward_image.
    """
    import torch
    from sam3.model.data_misc import NestedTensor
    outs = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for i in range(0, imgs.shape[0], chunk):
            outs.append(model.backbone.forward_image(
                imgs[i:i + chunk], need_sam3_out=False,
                need_interactive_out=True, need_propagation_out=True))
    backbone_out = {}
    for key in ("interactive", "sam2_backbone_out"):
        n_lvl = len(outs[0][key]["backbone_fpn"])
        fpn = [NestedTensor(torch.cat([o[key]["backbone_fpn"][l].tensors
                                       for o in outs]).float(), None)
               for l in range(n_lvl)]
        pos = [torch.cat([o[key]["vision_pos_enc"][l] for o in outs]).float()
               for l in range(len(outs[0][key]["vision_pos_enc"]))]
        backbone_out[key] = {"backbone_fpn": fpn, "vision_pos_enc": pos,
                             "vision_features": fpn[-1].tensors}
    if model.use_high_res_features_in_sam:
        for key, dec in (("interactive", model.interactive_sam_mask_decoder),
                         ("sam2_backbone_out", model.sam_mask_decoder)):
            fpn = backbone_out[key]["backbone_fpn"]
            fpn[0].tensors = dec.conv_s0(fpn[0].tensors)
            fpn[1].tensors = dec.conv_s1(fpn[1].tensors)
    return backbone_out


def _focal(inputs, targets, alpha=0.25, gamma=2.0):
    import torch.nn.functional as F
    prob = inputs.sigmoid()
    ce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce * ((1 - p_t) ** gamma)
    return (alpha * targets + (1 - alpha) * (1 - targets)) * loss


def frame_loss(out, target):
    """SAM2 MultiStepMultiMasksAndIous for one frame; target [N,H,W] bool."""
    import torch
    import torch.nn.functional as F
    N = target.shape[0]
    tgt = F.interpolate(target.unsqueeze(1).float(),
                        size=(LOSS_RES, LOSS_RES), mode="area")
    tgt = (tgt >= 0.5).float()
    losses = {k: 0.0 for k in LOSS_W}
    for src, ious, obj in zip(out["multistep_pred_multimasks"],
                              out["multistep_pred_ious"],
                              out["multistep_object_score_logits"]):
        src, ious, obj = src[:N].float(), ious[:N].float(), obj[:N].float()
        if src.shape[-2:] != tgt.shape[-2:]:
            src = F.interpolate(src, size=tgt.shape[-2:], mode="bilinear",
                                align_corners=False)
        t = tgt.expand_as(src)
        focal = _focal(src, t).flatten(2).mean(-1)
        p, tf = src.sigmoid().flatten(2), t.flatten(2)
        dice = 1 - (2 * (p * tf).sum(-1) + 1) / (p.sum(-1) + tf.sum(-1) + 1)
        pm, gm = src.flatten(2) > 0, tf > 0
        actual = ((pm & gm).sum(-1).float()
                  / (pm | gm).sum(-1).float().clamp(min=1))
        iou_l = F.l1_loss(ious, actual, reduction="none")
        present = gm[:, 0].any(-1, keepdim=True).float()
        cls = F.binary_cross_entropy_with_logits(obj, present, reduction="none")
        if focal.shape[1] > 1:
            best = (focal * LOSS_W["mask"] + dice * LOSS_W["dice"]).argmin(-1)
            ar = torch.arange(N, device=src.device)
            focal_b = focal[ar, best].unsqueeze(1)
            dice_b = dice[ar, best].unsqueeze(1)
            iou_b = iou_l.mean(-1, keepdim=True)
        else:
            focal_b, dice_b, iou_b = focal, dice, iou_l
        losses["mask"] = losses["mask"] + (focal_b * present).sum()
        losses["dice"] = losses["dice"] + (dice_b * present).sum()
        losses["iou"] = losses["iou"] + (iou_b * present).sum()
        losses["class"] = losses["class"] + cls.sum()
    return losses, N


def clip_forward(model, sample, device, interact_all=False):
    inp = make_input(sample, device)
    backbone_out = compute_backbone(model, inp.img_batch.tensors)
    backbone_out = model.prepare_prompt_inputs(backbone_out, inp)
    # Eval-mode point/box prompts need the prompted objects named.
    objs = list(range(sample["masks"].shape[1])) if interact_all else None
    outs = model.forward_tracking(backbone_out, inp, objects_to_interact=objs)
    total = {k: 0.0 for k in LOSS_W}
    n_obj = 0
    for t, out in enumerate(outs):
        l, n = frame_loss(out, inp.find_targets[t].segments)
        for k in total:
            total[k] = total[k] + l[k]
        n_obj += n
    parts = {k: v / max(n_obj, 1) for k, v in total.items()}
    loss = sum(LOSS_W[k] * parts[k] for k in parts)
    return loss, parts, outs, inp


def _mask_boxes(m):
    import torch
    out = torch.zeros(m.shape[0], 4, device=m.device)
    rows, cols = m.any(2), m.any(1)
    for i in range(m.shape[0]):
        r = torch.nonzero(rows[i]).flatten()
        c = torch.nonzero(cols[i]).flatten()
        if len(r) and len(c):
            out[i] = torch.tensor([c[0], r[0], c[-1] + 1, r[-1] + 1],
                                  device=m.device, dtype=out.dtype)
    return out


def _box_iou(a, b):
    lt = a[:, :2].maximum(b[:, :2])
    rb = a[:, 2:].minimum(b[:, 2:])
    inter = (rb - lt).clamp(min=0).prod(1)
    area_a = (a[:, 2:] - a[:, :2]).clamp(min=0).prod(1)
    area_b = (b[:, 2:] - b[:, :2]).clamp(min=0).prod(1)
    return inter / (area_a + area_b - inter).clamp(min=1)


def evaluate(model, loader, device, prompt="mask"):
    """
    Eval-mode propagation from a GT prompt on frame 0 ("mask", as the
    pipeline seeds tracks from detections, or "box"). Reports mean mask IoU
    and box IoU (box of the predicted mask vs GT) over (frame > 0, visible
    object) pairs, and object-present accuracy.
    """
    import torch
    model.eval()
    saved = (model.prob_to_use_pt_input_for_eval,
             model.prob_to_use_box_input_for_eval,
             model.num_correction_pt_per_frame)
    if prompt == "box":
        model.prob_to_use_pt_input_for_eval = 1.0
        model.prob_to_use_box_input_for_eval = 1.0
        model.num_correction_pt_per_frame = 0  # no GT-derived clicks
    else:
        model.prob_to_use_pt_input_for_eval = 0.0
    ious, bious, ok, n_pres, losses = [], [], 0, 0, []
    with torch.no_grad():
        for sample in loader:
            loss, _, outs, inp = clip_forward(model, sample, device,
                                              interact_all=(prompt == "box"))
            losses.append(float(loss))
            for t in range(1, len(outs)):
                pred = outs[t]["pred_masks_high_res"][:, 0] > 0
                gt = inp.find_targets[t].segments
                n = min(pred.shape[0], gt.shape[0])
                pred, gt = pred[:n], gt[:n]
                if pred.shape[-2:] != gt.shape[-2:]:
                    continue
                vis = gt.flatten(1).any(1)
                inter = (pred & gt).flatten(1).sum(1).float()
                uni = (pred | gt).flatten(1).sum(1).float().clamp(min=1)
                ious += (inter / uni)[vis].tolist()
                bious += _box_iou(_mask_boxes(pred), _mask_boxes(gt))[vis].tolist()
                ok += int((pred.flatten(1).any(1) == vis).sum())
                n_pres += n
    (model.prob_to_use_pt_input_for_eval,
     model.prob_to_use_box_input_for_eval,
     model.num_correction_pt_per_frame) = saved
    model.train()
    model.backbone.eval()
    return dict(loss=float(np.mean(losses)) if losses else float("nan"),
                miou=float(np.mean(ious)) if ious else float("nan"),
                box_iou=float(np.mean(bious)) if bious else float("nan"),
                presence_acc=ok / max(n_pres, 1), n_pairs=len(ious))


# =============================================================================
# Training loop
# =============================================================================

DEFAULTS = dict(
    max_iterations=4000, learning_rate=2e-5, weight_decay=0.05,
    warmup_iterations=300, clip_length=8, max_objects=8, eval_every=500,
    eval_clips=300, num_workers=4, seed=0, box_targets=False,
    timeout=604800,
)


def run(params):
    """
    Train. ``params`` holds: train_spec / val_spec (paths to JSON specs),
    sam_model_id (source 3.1 checkpoint), work_dir, output_model (path to
    write), plus the DEFAULTS keys. Works as a single process or as a
    torch.distributed.run rank.
    """
    import torch
    import torch.distributed as dist

    p = dict(DEFAULTS)
    p.update(params)
    distributed = "RANK" in os.environ and int(os.environ.get(
        "WORLD_SIZE", "1")) > 1
    if distributed:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        local = int(os.environ.get("LOCAL_RANK", rank))
    else:
        rank, world, local = 0, 1, 0
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    is_main = rank == 0
    seed = int(p["seed"])
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    work = p["work_dir"]
    os.makedirs(work, exist_ok=True)
    t_start = time.time()

    with open(p["train_spec"]) as f:
        train_seqs = prepare_sequences(json.load(f), p["box_targets"], "train")
    val_seqs = []
    if p.get("val_spec") and os.path.exists(p["val_spec"]):
        with open(p["val_spec"]) as f:
            val_seqs = prepare_sequences(json.load(f), p["box_targets"], "val")
    if not train_seqs:
        raise RuntimeError(
            "No usable training sequences: SAM3 tracker training needs tracks "
            "with polygon annotations (or box_targets = true).")

    model = build_trainable_tracker(p["sam_model_id"], device, rank, seed)
    model.train()
    model.backbone.eval()
    params_ = [q for q in model.parameters() if q.requires_grad]
    if is_main:
        log(f"trainable parameters {sum(q.numel() for q in params_):,} / "
            f"{sum(q.numel() for q in model.parameters()):,}")
    opt = torch.optim.AdamW(
        [{"params": [q for q in params_ if q.ndim > 1],
          "weight_decay": float(p["weight_decay"])},
         {"params": [q for q in params_ if q.ndim <= 1], "weight_decay": 0.0}],
        lr=float(p["learning_rate"]))
    scaler = torch.amp.GradScaler("cuda")

    iters = int(p["max_iterations"])
    warm = int(p["warmup_iterations"])
    lr0 = float(p["learning_rate"])

    def lr_at(i):
        if i < warm:
            return lr0 * (i + 1) / warm
        frac = (i - warm) / max(1, iters - warm)
        return lr0 * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * frac)))

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

    train_ds = ClipDataset(train_seqs, int(p["clip_length"]),
                           int(p["max_objects"]), train=True)
    if is_main:
        log(f"{len(train_seqs)} training sequences, {len(train_ds)} clips; "
            f"{len(val_seqs)} validation sequences")
    if distributed:
        sampler = torch.utils.data.DistributedSampler(
            train_ds, num_replicas=world, rank=rank, shuffle=True, seed=seed)
    else:
        sampler = torch.utils.data.RandomSampler(train_ds)
    workers = int(p["num_workers"])
    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=1, sampler=sampler, num_workers=workers,
        collate_fn=_collate_one, pin_memory=True,
        persistent_workers=workers > 0)

    val_loader = None
    if val_seqs:
        val_ds = ClipDataset(val_seqs, int(p["clip_length"]),
                             int(p["max_objects"]), train=False,
                             first_frame_objects_only=True,
                             stride=int(p["clip_length"]),
                             max_clips=int(p["eval_clips"]), seed=123)
        sub = torch.utils.data.Subset(val_ds,
                                      list(range(rank, len(val_ds), world)))
        val_loader = torch.utils.data.DataLoader(
            sub, batch_size=1, shuffle=False, num_workers=min(2, workers),
            collate_fn=_collate_one)
    # Mask-init val on polygon data; box-init (and box IoU) otherwise.
    val_prompt = ("mask" if val_seqs and all(s["target"] == "poly"
                                             for s in val_seqs) else "box")
    select_key = "miou" if val_prompt == "mask" else "box_iou"

    def run_eval(it):
        r = evaluate(model, val_loader, device, prompt=val_prompt)
        if distributed:
            n = r["n_pairs"]
            v = torch.tensor([r["loss"], r["miou"] * n, r["box_iou"] * n, n,
                              r["presence_acc"]], device=device,
                             dtype=torch.float64)
            v = torch.nan_to_num(v)
            dist.all_reduce(v)
            n = max(v[3].item(), 1)
            r = dict(loss=v[0].item() / world, miou=v[1].item() / n,
                     box_iou=v[2].item() / n, n_pairs=int(v[3].item()),
                     presence_acc=v[4].item() / world)
        if is_main:
            log(f"eval iter {it}: " + json.dumps(r))
            with open(os.path.join(work, "eval.jsonl"), "a") as f:
                f.write(json.dumps(dict(iter=it, **r)) + "\n")
        return r

    best_f = os.path.join(work, "best.json")
    best_path = os.path.join(work, "best.pt")

    def maybe_save_best(it, r):
        if not is_main:
            return
        best = json.load(open(best_f)) if os.path.exists(best_f) else None
        if best is None or r[select_key] > best[select_key]:
            torch.save(dict(model=tracker_state_dict(model), iter=it, eval=r),
                       best_path)
            json.dump(dict(iter=it, **r), open(best_f, "w"))
            log(f"new best {select_key} {r[select_key]:.4f} at iteration {it}")

    eval_every = int(p["eval_every"])
    if val_loader is not None and start_iter == 0 and eval_every > 0:
        maybe_save_best(0, run_eval(0))  # the source model, as a floor

    it, epoch, n_oom, hist = start_iter, 0, 0, []
    t_last = time.time()
    timeout = float(p["timeout"]) if p.get("timeout") else None
    opt.zero_grad(set_to_none=True)
    done = it >= iters
    while not done:
        if distributed:
            sampler.set_epoch(epoch)
        for sample in loader:
            for g in opt.param_groups:
                g["lr"] = lr_at(it)
            skip = 0.0
            try:
                loss, parts, _, _ = clip_forward(model, sample, device)
                scaler.scale(loss).backward()
            except torch.OutOfMemoryError:
                skip = 1.0
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                n_oom += 1
                log(f"rank {rank}: out of GPU memory on a "
                    f"{sample['masks'].shape[1]}-object clip of "
                    f"{sample['name']}; skipped ({n_oom} so far)")
            for q in params_:
                if q.grad is None:
                    q.grad = torch.zeros_like(q)
            if distributed:
                flat = torch.cat([q.grad.flatten() for q in params_] +
                                 [torch.tensor([skip], device=device)])
                dist.all_reduce(flat)
                skip = float(flat[-1])
                flat /= world
                off = 0
                for q in params_:
                    n = q.numel()
                    q.grad.copy_(flat[off:off + n].view_as(q))
                    off += n
            it += 1
            if skip > 0:
                opt.zero_grad(set_to_none=True)
            else:
                scaler.unscale_(opt)
                gn = torch.nn.utils.clip_grad_norm_(params_, 0.1)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                hist.append([float(loss)] + [float(parts[k]) for k in LOSS_W])

            if is_main and it % 20 == 0 and hist:
                h = np.mean(hist, 0)
                hist = []
                dt = (time.time() - t_last) / 20
                t_last = time.time()
                log(f"iter {it}/{iters} loss {h[0]:.4f} (mask {h[1]:.4f} "
                    f"dice {h[2]:.4f} iou {h[3]:.4f} cls {h[4]:.4f}) "
                    f"gnorm {float(gn):.3f} lr {lr_at(it):.2e} "
                    f"{dt:.2f}s/it")
            if is_main and it % 500 == 0:
                torch.save(dict(model=tracker_state_dict(model),
                                opt=opt.state_dict(),
                                scaler=scaler.state_dict(), iter=it),
                           last_path + ".tmp")
                os.replace(last_path + ".tmp", last_path)
            out_of_time = (timeout is not None
                           and time.time() - t_start > timeout)
            if distributed:
                flag = torch.tensor([float(out_of_time)], device=device)
                dist.all_reduce(flag)
                out_of_time = flag.item() > 0
            if val_loader is not None and eval_every > 0 and (
                    it % eval_every == 0 or it >= iters or out_of_time):
                maybe_save_best(it, run_eval(it))
            if it >= iters or out_of_time:
                if out_of_time and is_main:
                    log("training timeout reached; stopping")
                done = True
                break
        epoch += 1

    if is_main:
        if val_loader is None or not os.path.exists(best_path):
            # No validation: the final weights are the model.
            torch.save(dict(model=tracker_state_dict(model), iter=it),
                       best_path)
        st = torch.load(best_path, map_location="cpu", weights_only=False)
        export_checkpoint(st["model"], p["sam_model_id"], p["output_model"])
        log(f"exported iteration {st['iter']} weights to {p['output_model']}")
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


def main():
    if len(sys.argv) != 2:
        print("usage: sam3_tracker_training.py params.json", file=sys.stderr)
        sys.exit(2)
    with open(sys.argv[1]) as f:
        run(json.load(f))


if __name__ == "__main__":
    main()
