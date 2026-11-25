# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #
"""Model construction and online inference shared by the YOLOV++ trainer and detector.

YOLOV++ is a YOLOX detector whose per-frame proposals are re-scored by an
attention head over the proposals of other frames of the same video. The
upstream library (packages/pytorch-libs/yolov, package name ``yolox``) only
ships a batch forward for it; the per-frame feature memory used online lives
here and calls the upstream head's own aggregation modules."""

import random
import re

# (depth, width) of the YOLOX CSPDarknet variants.
MODEL_SIZES = {
    's': (0.33, 0.50), 'm': (0.67, 0.75), 'l': (1.00, 1.00), 'x': (1.33, 1.25),
}

# COCO-pretrained YOLOX weights each variant is fine-tuned from.
SEED_URLS = {
    size: 'https://github.com/Megvii-BaseDetection/YOLOX/releases/download/'
          '0.1.1rc0/yolox_%s.pth' % size
    for size in MODEL_SIZES
}


def natural_sort_key(name):
    """Sort key ordering frame2 before frame10."""
    return [int(tok) if tok.isdigit() else tok for tok in re.split(r'(\d+)', name)]


def quiet_upstream_logging():
    """Silence the per-batch warnings upstream emits through loguru."""
    from loguru import logger
    logger.disable('yolox')


def model_scale(model_size):
    key = str(model_size).strip().lower()
    if key not in MODEL_SIZES:
        raise ValueError(f"Unknown YOLOV model size: {model_size}. "
                         f"Expected one of: {', '.join(MODEL_SIZES)}")
    return MODEL_SIZES[key]


def init_yolo(model):
    """YOLOX's BatchNorm settings, as upstream applies to every model it builds."""
    import torch.nn as nn
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eps = 1e-3
            module.momentum = 0.03


def build_base_detector(model_size, num_classes):
    """The still-image YOLOX detector trained in stage 1."""
    from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead

    depth, width = model_scale(model_size)
    in_channels = [256, 512, 1024]
    backbone = YOLOPAFPN(depth, width, in_channels=in_channels, act='silu')
    head = YOLOXHead(num_classes, width, in_channels=in_channels, act='silu')
    model = YOLOX(backbone, head)
    init_yolo(model)
    model.head.initialize_biases(1e-2)
    return model


def video_head_config(model_size, min_proposals=-1, max_proposals=0):
    """Head arguments of upstream exps/yolov++/v++_*_decoupleReg.py.

    min_proposals < 0 keeps the upstream value (50 for the large variant, 0
    for the small one); max_proposals = 0 leaves the count unbounded.
    """
    key = str(model_size).strip().lower()
    if int(min_proposals) < 0:
        min_proposals = 50 if key in ('l', 'x') else 0
    return dict(
        heads=4, drop=0.0, use_score=True, defualt_p=30, sim_thresh=0.75,
        pre_nms=0.75, ave=True, defulat_pre=750, test_conf=0.001,
        use_mask=False, gmode=True, lmode=False, both_mode=False,
        localBlocks=1, use_ffn=True, use_time_emd=False, use_loc_emd=False,
        loc_fuse_type='identity', use_qkv=True, local_mask=False,
        local_mask_branch='', pure_pos_emb=False, loc_conf=False,
        iou_base=False, reconf=True, ota_mode=True, ota_cls=False,
        traj_linking=False, iou_window=0, globalBlocks=1, use_pre_nms=False,
        cat_ota_fg=False, agg_type='msa', decouple_reg=True,
        minimal_limit=int(min_proposals), maximal_limit=int(max_proposals),
    )


def build_video_detector(model_size, num_classes, head_config):
    """The YOLOV++ model trained in stage 2: YOLOX plus the aggregation head."""
    from yolox.models import YOLOPAFPN
    from yolox.models.v_plus_head import YOLOVHead
    from yolox.models.yolov_plus import YOLOV

    class ReferenceMaskedHead(YOLOVHead):
        """YOLOVHead whose losses can skip frames that carry no groundtruth.

        Upstream supervises every frame of a batch. A reference frame read from
        disk without labels would be trained as empty, so its objectness logits
        are pinned to a confident negative, which the zero target then ignores.
        """
        supervised = None

        def postprocess_widx(self, *args, **kwargs):
            out = super().postprocess_widx(*args, **kwargs)
            self._proposal_counts = [0 if idx is None else len(idx) for idx in out[1]]
            return out

        def get_losses(self, outputs, *args):
            import torch
            supervised = self.supervised
            if supervised is not None and not bool(supervised.all()):
                args = list(args)
                outputs = outputs.clone()
                outputs[~supervised, :, 4] = -30.0
                counts = torch.as_tensor(self._proposal_counts, device=outputs.device)
                frame_of = torch.repeat_interleave(
                    torch.arange(len(counts), device=outputs.device), counts)
                # args[9] is the refined objectness of every proposal.
                refined_obj = args[9].clone()
                refined_obj[~supervised[frame_of]] = -30.0
                args[9] = refined_obj
            return super().get_losses(outputs, *args)

    depth, width = model_scale(model_size)
    in_channels = [256, 512, 1024]
    backbone = YOLOPAFPN(depth, width, in_channels=in_channels)
    head = ReferenceMaskedHead(num_classes, width, in_channels=in_channels,
                               **dict(head_config))
    model = YOLOV(backbone, head)
    init_yolo(model)
    model.head.initialize_biases(1e-2)
    return model


def frozen_video_modules(model):
    """Modules upstream keeps fixed while the aggregation head trains."""
    head = model.head
    return [model.backbone, head.stems, head.reg_convs, head.cls_convs, head.reg_preds]


def load_matching_weights(model, state_dict):
    """Load every tensor whose name and shape match; returns (loaded, skipped) names."""
    own = model.state_dict()
    loaded, skipped = {}, []
    for key, value in own.items():
        other = state_dict.get(key)
        if other is not None and tuple(other.shape) == tuple(value.shape):
            loaded[key] = other
        else:
            skipped.append(key)
    model.load_state_dict(loaded, strict=False)
    return sorted(loaded), skipped


def read_checkpoint(path):
    """(state_dict, meta) of a checkpoint written here or by upstream YOLOX/YOLOV."""
    import torch
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"{path} is not a YOLOX/YOLOV checkpoint")
    state_dict = checkpoint.get('model', checkpoint)
    meta = checkpoint.get('meta') or {}
    return state_dict, dict(meta)


def checkpoint_num_classes(state_dict):
    weight = state_dict.get('head.cls_preds.0.weight')
    return None if weight is None else int(weight.shape[0])


def checkpoint_model_size(state_dict):
    """Model size from the stem width, or None for an unknown backbone."""
    weight = state_dict.get('backbone.backbone.stem.conv.conv.weight')
    if weight is None:
        return None
    for size, (_, width) in MODEL_SIZES.items():
        if int(weight.shape[0]) == int(64 * width):
            return size
    return None


def is_video_checkpoint(state_dict):
    return any(key.startswith('head.agg.') for key in state_dict)


def build_from_checkpoint(path, model_size=None, min_proposals=None, max_proposals=None):
    """Rebuild the model a checkpoint holds; returns (model, meta, is_video)."""
    state_dict, meta = read_checkpoint(path)
    size = checkpoint_model_size(state_dict) or meta.get('model_size') or model_size
    num_classes = checkpoint_num_classes(state_dict) or len(meta.get('class_names') or [])
    if not size or not num_classes:
        raise ValueError(f"{path}: cannot recover the YOLOX architecture from the weights")
    video = is_video_checkpoint(state_dict)
    if video:
        head_config = dict(meta.get('head_config') or video_head_config(size))
        if min_proposals is not None and int(min_proposals) >= 0:
            head_config['minimal_limit'] = int(min_proposals)
        if max_proposals is not None and int(max_proposals) >= 0:
            head_config['maximal_limit'] = int(max_proposals)
        model = build_video_detector(size, num_classes, head_config)
        meta['head_config'] = head_config
    else:
        model = build_base_detector(size, num_classes)
    model.load_state_dict(state_dict, strict=True)
    meta['model_size'] = size
    meta['num_classes'] = num_classes
    return model, meta, video


def select_cuda_device(device):
    """Make a cuda:N device the current one; upstream allocates on the current device."""
    import torch
    device = torch.device(device)
    if device.type != 'cuda':
        raise RuntimeError(
            "YOLOV++ needs a CUDA device: the upstream attention head "
            "allocates on 'cuda' explicitly.")
    if device.index is not None:
        torch.cuda.set_device(device.index)
    return torch.device('cuda', torch.cuda.current_device())


def preprocess_frame(bgr, input_size):
    """Letterbox a BGR uint8 frame to input_size; returns (CHW float32 array, ratio)."""
    from yolox.data.data_augment import preproc
    return preproc(bgr, tuple(input_size))


def base_detect(model, imgs, score_thresh, nms_thresh):
    """Still-image YOLOX detections: one [n, 7] tensor (or None) per frame."""
    import torch
    from yolox.utils.boxes import postprocess
    with torch.no_grad():
        outputs = model(imgs)
        return postprocess(outputs, model.head.num_classes,
                           conf_thre=score_thresh, nms_thre=nms_thresh)


def extract_proposals(model, imgs):
    """Per-frame proposals and their head features, as the upstream forward selects them.

    Returns one dict per frame (None when the frame has no proposal) holding
    'pred' [P, 7 + classes], 'cls' [P, C] and 'reg' [P, C].
    """
    import torch
    with torch.no_grad():
        return _extract_proposals(model, imgs)


def _extract_proposals(model, imgs):
    import torch
    head = model.head
    fpn_outs = model.backbone(imgs)
    decoded, cls_feats, reg_feats = [], [], []
    for k, x in enumerate(fpn_outs):
        x = head.stems[k](x)
        reg_feat = head.reg_convs[k](x)
        cls_feat = head.cls_convs[k](x)
        cls_feats.append(head.cls_convs2[k](x))
        reg_feats.append(reg_feat)
        decoded.append(torch.cat([
            head.reg_preds[k](reg_feat),
            head.obj_preds[k](reg_feat).sigmoid(),
            head.cls_preds[k](cls_feat).sigmoid()], 1))
    head.hw = [x.shape[-2:] for x in decoded]
    decoded = torch.cat([x.flatten(start_dim=2) for x in decoded], dim=2).permute(0, 2, 1)
    decoded = head.decode_outputs(decoded, dtype=fpn_outs[0].type())
    cls_flat = torch.cat([x.flatten(start_dim=2) for x in cls_feats], dim=2).permute(0, 2, 1)
    reg_flat = torch.cat([x.flatten(start_dim=2) for x in reg_feats], dim=2).permute(0, 2, 1)

    preds, idxs, _, _ = head.postprocess_widx(
        decoded, num_classes=head.num_classes, nms_thre=head.nms_thresh, ota_idxs=None)

    frames = []
    for i in range(imgs.shape[0]):
        if idxs[i] is None or len(idxs[i]) == 0:
            frames.append(None)
            continue
        idx = idxs[i].to(cls_flat.device)
        frames.append({'pred': preds[i], 'cls': cls_flat[i, idx], 'reg': reg_flat[i, idx]})
    return frames


def aggregate(model, frames, nms_thresh, keys=None):
    """Re-score the proposals of `frames` against each other with the YOLOV++ head.

    frames: non-empty proposal dicts from extract_proposals. Returns the final
    [n, 7] detections (x1, y1, x2, y2, obj, cls_conf, cls) of each frame index
    in keys (default: all), None where nothing survives.
    """
    import torch
    with torch.no_grad():
        return _aggregate(model, frames, nms_thresh, keys)


def _aggregate(model, frames, nms_thresh, keys):
    import torch
    from yolox.models.post_process import postprocess

    head = model.head
    cls_feats = torch.cat([f['cls'] for f in frames])
    reg_feats = torch.cat([f['reg'] for f in frames])
    cls_scores = torch.cat([f['pred'][:, 5] for f in frames]).to(cls_feats.dtype)
    fg_scores = torch.cat([f['pred'][:, 4] for f in frames]).to(cls_feats.dtype)

    kwargs = dict(head.kwargs)
    kwargs.update({'lframe': 0, 'gframe': len(frames), 'afternum': head.Afternum})
    agg_args = dict(sim_thresh=head.sim_thresh, ave=head.ave, use_mask=head.use_mask)
    features_cls, features_reg = head.agg(
        cls_feats.unsqueeze(0), reg_feats.unsqueeze(0), cls_scores, fg_scores,
        **agg_args, **kwargs)
    if kwargs.get('decouple_reg', False):
        _, features_reg = head.agg_iou(
            cls_feats.unsqueeze(0), reg_feats.unsqueeze(0), cls_scores, fg_scores,
            **agg_args, **kwargs)
    cls_preds = head.cls_pred(features_cls)
    obj_preds = head.obj_pred(features_reg) if kwargs.get('reconf', False) else None

    starts = [0]
    for f in frames:
        starts.append(starts[-1] + f['pred'].shape[0])

    results = []
    for k in (range(len(frames)) if keys is None else keys):
        lo, hi = starts[k], starts[k + 1]
        conf = None if obj_preds is None else [obj_preds[lo:hi].squeeze(-1)]
        result, _ = postprocess(
            [frames[k]['pred'].clone()], head.num_classes, [cls_preds[lo:hi]],
            conf_output=conf, nms_thre=nms_thresh)
        results.append(result[0])
    return results


class OnlineVideoDetector:
    """Causal YOLOV++ inference, one frame at a time.

    Keeps the proposals and head features of past frames and aggregates each
    new frame against a set of them: the frames at fixed offsets when
    frame_offsets is given, otherwise a random draw from the memory as
    upstream's online demo does.
    """

    def __init__(self, model, input_size, nms_thresh=0.5, reference_frames=15,
                 memory_frames=120, frame_offsets=None, seed=0):
        self.model = model
        self.input_size = tuple(int(v) for v in input_size)
        self.nms_thresh = float(nms_thresh)
        self.frame_offsets = sorted({-abs(int(o)) for o in (frame_offsets or [])})
        self.reference_frames = int(reference_frames)
        self.memory_frames = max(int(memory_frames), 1)
        if self.frame_offsets:
            self.memory_frames = -min(self.frame_offsets)
        self._seed = seed
        self.reset()

    def reset(self):
        """Forget every past frame; call at the start of a new sequence."""
        self._memory = []
        self._rng = random.Random(self._seed)

    def _references(self):
        if self.frame_offsets:
            picked = [self._memory[off] for off in self.frame_offsets
                      if -off <= len(self._memory)]
        else:
            usable = [f for f in self._memory if f is not None]
            count = min(self.reference_frames, len(usable))
            picked = self._rng.sample(usable, count) if count > 0 else []
        return [f for f in picked if f is not None]

    def remember(self, frame):
        self._memory.append(frame)
        del self._memory[:-self.memory_frames]

    def step_tensor(self, img):
        """Detect on a preprocessed [1, 3, H, W] tensor; returns [n, 7] or None."""
        frame = extract_proposals(self.model, img)[0]
        result = None
        if frame is not None:
            result = aggregate(self.model, [frame] + self._references(),
                               self.nms_thresh, keys=[0])[0]
        self.remember(frame)
        return result

    def step(self, bgr):
        """Detect on a BGR uint8 frame; returns [n, 6] (x1, y1, x2, y2, score, class)."""
        import torch
        img, ratio = preprocess_frame(bgr, self.input_size)
        device = next(self.model.parameters()).device
        result = self.step_tensor(torch.from_numpy(img).unsqueeze(0).to(device))
        return detections_to_numpy(result, ratio, bgr.shape[:2])


def detections_to_numpy(result, ratio, image_hw=None):
    """[n, 7] network-space detections -> [n, 6] numpy in image pixels."""
    import numpy as np
    if result is None or result.shape[0] == 0:
        return np.zeros((0, 6), dtype=np.float32)
    result = result.detach().float().cpu().numpy()
    boxes = result[:, :4] / float(ratio)
    if image_hw is not None:
        boxes[:, 0::2] = boxes[:, 0::2].clip(0, image_hw[1])
        boxes[:, 1::2] = boxes[:, 1::2].clip(0, image_hw[0])
    scores = result[:, 4] * result[:, 5]
    return np.concatenate([boxes, scores[:, None], result[:, 6:7]], axis=1).astype(np.float32)
