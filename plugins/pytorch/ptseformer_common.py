# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
PTSEFormer model code shared by the VIAME trainer, launcher and detector.

Imports the upstream model (packages/pytorch-libs/ptseformer) unmodified and
adapts it from here: a pure-PyTorch deformable attention in place of the
compiled CUDA op, the torch 1.5 attention semantics upstream relies on, and a
forward split into a per-frame stage and a fusion stage so a detector can
cache the frames it has already seen.

Imports nothing from kwiver or viame, so the launcher can load it by path.
"""

import argparse
import importlib
import os
import re
import sys
import types

UPSTREAM_PACKAGE = "ptseformer"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Channels of the last three ResNet stages, as upstream hard-codes them.
SUPPORTED_BACKBONES = ("resnet50", "resnet101", "resnext50_32x4d", "resnext101_32x8d")

_UPSTREAM = None


def ms_deform_attn_pytorch(value, spatial_shapes, sampling_locations, attention_weights):
    """Multi-scale deformable attention with grid_sample (no compiled op).

    Same result as upstream's ms_deform_attn_core_pytorch, but each level is
    weighted and summed as it is sampled instead of stacking all of them.
    """
    import torch.nn.functional as F

    n, _, heads, dim = value.shape
    num_queries = sampling_locations.shape[1]
    grids = 2 * sampling_locations - 1
    output = None
    start = 0
    for level, (h, w) in enumerate(spatial_shapes.tolist()):
        # n, h*w, heads, dim -> n*heads, dim, h, w
        level_value = value[:, start:start + h * w].flatten(2).transpose(1, 2)
        level_value = level_value.reshape(n * heads, dim, h, w)
        start += h * w
        # n, queries, heads, points, 2 -> n*heads, queries, points, 2
        level_grid = grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        sampled = F.grid_sample(level_value, level_grid, mode='bilinear',
                                padding_mode='zeros', align_corners=False)
        weights = attention_weights[:, :, :, level].transpose(1, 2)
        weights = weights.reshape(n * heads, 1, num_queries, -1)
        sampled = (sampled * weights).sum(-1)
        output = sampled if output is None else output + sampled
    return output.view(n, heads * dim, num_queries).transpose(1, 2).contiguous()


class _PureMSDeformAttnFunction:
    """Stands in for the compiled MSDeformAttnFunction; autograd does the backward."""

    @staticmethod
    def apply(value, spatial_shapes, level_start_index, sampling_locations,
              attention_weights, im2col_step=None):
        return ms_deform_attn_pytorch(
            value, spatial_shapes, sampling_locations, attention_weights)


def load_upstream():
    """Import the upstream model modules and return what the plugin uses."""
    global _UPSTREAM
    if _UPSTREAM is not None:
        return _UPSTREAM

    import torch
    from torch import nn
    from torch.utils.checkpoint import checkpoint

    pkg = UPSTREAM_PACKAGE
    # Upstream imports these by top-level name; alias them for the import only.
    aliases = ("util", "util.misc", "util.box_ops", "MultiScaleDeformableAttention")
    saved = {name: sys.modules.get(name) for name in aliases}
    try:
        sys.modules["MultiScaleDeformableAttention"] = types.ModuleType(
            "MultiScaleDeformableAttention")
        try:
            sys.modules["util"] = importlib.import_module(pkg + ".util")
        except ImportError as exc:
            raise ImportError(
                "The PTSEFormer library is not installed. Build VIAME with "
                "VIAME_ENABLE_PYTORCH-PTSEFORMER=ON, or put a folder named "
                "'ptseformer' holding the upstream checkout on PYTHONPATH.") from exc
        misc = importlib.import_module(pkg + ".util.misc")
        box_ops = importlib.import_module(pkg + ".util.box_ops")
        sys.modules["util.misc"] = misc
        sys.modules["util.box_ops"] = box_ops
        builder = importlib.import_module(pkg + ".src.models.model_builder")
        attn = importlib.import_module(pkg + ".src.models.ops.modules.ms_deform_attn")
        detr_util = importlib.import_module(pkg + ".src.models.detr_util.detr_util")
        matcher = importlib.import_module(pkg + ".src.models.matcher.matcher")
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    attn.MSDeformAttnFunction = _PureMSDeformAttnFunction

    class TemporalAttention(nn.MultiheadAttention):
        """Upstream's enhancement attention under torch >= 1.6.

        Upstream passes (1, tokens, C) to a sequence-first attention, so tokens
        are the batch and each one attends over the same position in every
        reference frame. torch 1.5 folded the longer key into the sequence
        axis implicitly; that reshape is explicit here.
        """

        def forward(self, query, key, value, **kwargs):
            tokens = query.shape[1]
            key = key.reshape(-1, tokens, key.shape[-1])
            value = value.reshape(-1, tokens, value.shape[-1])
            kwargs.setdefault("need_weights", False)
            return super().forward(query, key, value, **kwargs)

    class ZeroIndexedMatcher(matcher.HungarianMatcher):
        """Upstream's matcher reads the class cost at label - 1; labels here are 0-based."""

        def forward(self, outputs, targets):
            shifted = [dict(t, labels=t["labels"] + 1) for t in targets]
            return super().forward(outputs, shifted)

    class Criterion(detr_util.SetCriterion):
        """Upstream's criterion plus a single-process evaluation of the same losses."""

        def local_losses(self, outputs, targets):
            # forward() averages the box count over all ranks, which blocks
            # when only one rank is validating.
            outputs = {k: v for k, v in outputs.items() if k != "aux_outputs"}
            indices = self.matcher(outputs, targets)
            num_boxes = max(sum(len(t["labels"]) for t in targets), 1)
            losses = {}
            for loss in self.losses:
                losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))
            return losses

    class CausalPTSEFormer(builder.PTSEFormer):
        """PTSEFormer with forward() split into a per-frame and a fusion stage.

        forward() computes what upstream computes. The split lets the detector
        encode each frame once and reuse it as a reference for later frames.
        """

        def __init__(self, cfg):
            super().__init__(cfg)
            # Recompute the enhancement stages in the backward pass (same gradients).
            self.stage_checkpointing = False
            for stage in (self.s_decoder1, self.s_decoder2, self.corr, self.our_decoder):
                for module in stage.modules():
                    if type(module) is nn.MultiheadAttention:
                        module.__class__ = TemporalAttention

        def _stage(self, decoder, tgt, memory):
            """One enhancement decoder on one feature level."""
            if self.stage_checkpointing and self.training and torch.is_grad_enabled():
                # Their 2048-wide FFNs over every token dominate training memory.
                return checkpoint(
                    lambda t, m: decoder(tgt=t, memory=m), tgt, memory,
                    use_reentrant=False).squeeze(0)
            return decoder(tgt=tgt, memory=memory).squeeze(0)

        def encode_frame(self, nest):
            """Stage 1: backbone and deformable encoder for one frame."""
            if nest.tensors.shape[0] != 1:
                raise ValueError(
                    "PTSEFormer takes one key frame per forward pass (its "
                    "enhancement stages drop the batch axis)")
            srcs, masks, pos = self.get_feat(nest)
            memory, dec_utils = self.d_enc(srcs, masks, pos)
            sizes = [h * w for h, w in dec_utils[0].tolist()]
            return {
                "memory": memory,
                "levels": list(memory.split(sizes, dim=1)),
                "dec_utils": dec_utils,
            }

        def reference_queries(self, ref, dec_utils):
            """Object queries decoded from a reference frame's memory (upstream 'qln')."""
            hs, _, _ = self.pre_dec(ref["memory"], self.query_embed.weight, dec_utils)
            return self.qln(hs[-1])

        def fuse(self, cur, refs, ref_queries=None):
            """Stages 2-5: enhance the key frame with its references and decode."""
            if not refs:
                raise ValueError("PTSEFormer needs at least one reference frame")
            cur_levels = cur["levels"]
            ref_levels = [ref["levels"] for ref in refs]
            ref_cat = [torch.cat(level, dim=1) for level in zip(*ref_levels)]

            # Stage 2: references corrected by the key frame; key frame gathers references.
            ref_corr = [
                [self._stage(self.corr, r, c) for r, c in zip(levels, cur_levels)]
                for levels in ref_levels]
            ref_corr_cat = [torch.cat(level, dim=1) for level in zip(*ref_corr)]
            enhanced = [self._stage(self.s_decoder1, c, r)
                        for c, r in zip(cur_levels, ref_cat)]
            # Stage 3: gather the corrected references.
            enhanced = [self._stage(self.s_decoder2, c, r)
                        for c, r in zip(enhanced, ref_corr_cat)]
            # Stage 4: gate against the unenhanced key frame.
            enhanced = [self._stage(self.our_decoder, e, c)
                        for e, c in zip(enhanced, cur_levels)]
            memory = torch.cat(enhanced, dim=1)

            # Stage 5: key-frame queries plus the queries of every reference.
            dec_utils = cur["dec_utils"]
            if ref_queries is None:
                ref_queries = [self.reference_queries(ref, dec_utils) for ref in refs]
            queries = self.query_embed.weight.expand(memory.shape[0], -1, -1)
            query_mix = torch.cat([queries] + list(ref_queries), dim=1)
            hs, init_reference, inter_references = self.d_dec(memory, query_mix, dec_utils)
            # Heads in float32: upstream's inverse_sigmoid overflows in half precision.
            with torch.autocast(hs.device.type, enabled=False):
                return self.cal_loss(
                    hs.float(), init_reference.float(), inter_references.float())

        def forward(self, samples_dict, reference_grad=True):
            cur = self.encode_frame(samples_dict["cur"])
            with torch.set_grad_enabled(reference_grad and torch.is_grad_enabled()):
                refs = [self.encode_frame(nest) for nest in samples_dict["ref_l"]]
            return self.fuse(cur, refs)

    _UPSTREAM = types.SimpleNamespace(
        model_class=CausalPTSEFormer,
        matcher_class=ZeroIndexedMatcher,
        criterion_class=Criterion,
        NestedTensor=misc.NestedTensor,
        box_ops=box_ops,
    )
    return _UPSTREAM


def default_arch():
    """Architecture settings recorded in every checkpoint (upstream r101 yaml values)."""
    return {
        "backbone": "resnet101",
        "dilation": False,
        "num_feature_levels": 4,
        "hidden_dim": 256,
        "dim_feedforward": 1024,
        "nheads": 8,
        "enc_n_points": 4,
        "dec_n_points": 4,
        "dropout": 0.1,
        "num_queries": 100,
        "num_classes": 1,
    }


def make_upstream_cfg(arch, lr_backbone=1e-5):
    """The slice of upstream's yacs config its model builder reads."""
    ns = types.SimpleNamespace
    return ns(
        TRAIN=ns(lr_backbone=lr_backbone, pretrained="", device="cpu"),
        MODEL=ns(
            name="PTSEFormer",
            backbone=arch["backbone"],
            dilation=bool(arch["dilation"]),
            position_embedding="sine",
            num_feature_levels=int(arch["num_feature_levels"]),
            # Only sizes the shared prediction-head list; two decoder layers are built.
            enc_layers=6,
            dec_layers=6,
            dim_feedforward=int(arch["dim_feedforward"]),
            hidden_dim=int(arch["hidden_dim"]),
            dropout=float(arch["dropout"]),
            nheads=int(arch["nheads"]),
            dec_n_points=int(arch["dec_n_points"]),
            enc_n_points=int(arch["enc_n_points"]),
            masks=False,
            num_classes=int(arch["num_classes"]),
            num_queries=int(arch["num_queries"]),
        ),
    )


def build_model(arch, lr_backbone=1e-5):
    """Build the (unseeded) model for an architecture dict."""
    import warnings

    if arch["backbone"] not in SUPPORTED_BACKBONES:
        raise ValueError(
            f"backbone = {arch['backbone']!r} is not supported; use one of "
            f"{', '.join(SUPPORTED_BACKBONES)}")
    upstream = load_upstream()
    with warnings.catch_warnings():
        # Upstream builds the ResNet with torchvision's retired pretrained=False.
        warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
        return upstream.model_class(make_upstream_cfg(arch, lr_backbone))


def build_criterion(num_classes):
    """Upstream's set criterion with its yaml loss weights and a 0-based matcher."""
    upstream = load_upstream()
    coef = {"class": 2.0, "bbox": 5.0, "giou": 2.0, "focal_alpha": 0.25}
    matcher = upstream.matcher_class(
        cost_class=coef["class"], cost_bbox=coef["bbox"], cost_giou=coef["giou"])
    weight_dict = {"loss_ce": coef["class"], "loss_bbox": coef["bbox"],
                   "loss_giou": coef["giou"]}
    # Upstream weights only the final layer, although it computes the auxiliary terms.
    return upstream.criterion_class(
        num_classes, matcher, weight_dict, ["labels", "boxes", "cardinality"],
        focal_alpha=coef["focal_alpha"])


def read_checkpoint(path):
    """Load a checkpoint as plain data (no arbitrary unpickling)."""
    import torch

    with torch.serialization.safe_globals([argparse.Namespace]):
        return torch.load(path, map_location="cpu", weights_only=True)


def _deformable_detr_to_ptseformer(state):
    """Rename Deformable-DETR (fundamentalvision) weights onto PTSEFormer's modules."""
    renamed = {}
    for key, value in state.items():
        if key.startswith("transformer.encoder."):
            renamed["d_encoder." + key[len("transformer.encoder."):]] = value
        elif key.startswith("transformer.decoder.layers."):
            tail = key[len("transformer.decoder."):]
            renamed["d_decoder." + tail] = value
            renamed["pre_decoder." + tail] = value
        elif key == "transformer.level_embed":
            renamed["level_embed"] = value
        elif key.startswith("transformer.reference_points."):
            tail = key[len("transformer.reference_points."):]
            renamed["reference_points." + tail] = value
            renamed["reference_points1." + tail] = value
        else:
            renamed[key] = value
    return renamed


def seed_model(model, path):
    """Seed a model from a local file and return a one-line description.

    Accepts a PTSEFormer checkpoint, a Deformable-DETR checkpoint or a
    torchvision ResNet state dict; tensors whose shape differs are skipped.
    """
    checkpoint = read_checkpoint(path)
    state = checkpoint
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict"):
            if isinstance(checkpoint.get(key), dict):
                state = checkpoint[key]
                break
    state = {k[len("module."):] if k.startswith("module.") else k: v
             for k, v in state.items()}

    if "conv1.weight" in state and "layer4.0.conv1.weight" in state:
        kind = "torchvision ResNet backbone"
        state = {"backbone.0.body." + k: v for k, v in state.items()
                 if not k.startswith("fc.")}
    elif any(k.startswith("transformer.encoder.") for k in state):
        kind = "Deformable-DETR"
        state = _deformable_detr_to_ptseformer(state)
    elif any(k.startswith("d_encoder.") for k in state):
        kind = "PTSEFormer"
    else:
        raise ValueError(
            f"{path}: not a PTSEFormer, Deformable-DETR or torchvision ResNet "
            "checkpoint (no recognised parameter names)")

    own = model.state_dict()
    # A deeper ResNet's weights fit a shallower one's shapes; refuse rather than mix them.
    if any(k.startswith("backbone.0.body.") and k not in own for k in state):
        raise ValueError(
            f"{path} was made with a different backbone than the one being "
            "built; set the trainer's backbone to match the seed_model")
    usable, mismatched = {}, []
    for key, value in state.items():
        if key not in own:
            continue
        target = own[key]
        if tuple(value.shape) == tuple(target.shape):
            usable[key] = value
        elif (key == "query_embed.weight" and value.shape[1] == target.shape[1]
              and value.shape[0] > target.shape[0]):
            # A longer query table seeds a shorter one from its first rows.
            usable[key] = value[:target.shape[0]]
        else:
            mismatched.append(key)
    model.load_state_dict(usable, strict=False)

    def groups(keys):
        return sorted({k.split(".")[0] for k in keys})

    missing = [k for k in own if k not in usable]
    summary = (f"{kind} seed {path}: loaded {len(usable)}/{len(own)} tensors"
               f"; left at init: {', '.join(groups(missing)) or 'none'}")
    if mismatched:
        summary += f"; shape mismatch (skipped): {', '.join(groups(mismatched))}"
    return summary


def parse_reference_window(value):
    """Parse "min,max" reference offsets, in frames ("-12,0") or seconds ("-0.6s,0s")."""
    if isinstance(value, (list, tuple)):
        tokens = [str(tok) for tok in value]
    else:
        tokens = str(value).strip().strip("[]()").replace(";", ",").split(",")
    tokens = [tok.strip().strip("'\"").lower() for tok in tokens if tok.strip()]
    if len(tokens) != 2:
        raise ValueError(
            f"reference_window = {value!r}: expected two offsets, 'min,max'")
    window = []
    for tok in tokens:
        if tok.endswith("s"):
            window.append(f"{float(tok[:-1]):g}s")
        else:
            window.append(int(tok))
    return window


def reference_window_to_frames(window, frame_rate):
    """(min, max) frame offsets for a parse_reference_window result."""
    frames = []
    for bound in window:
        if isinstance(bound, str):
            frames.append(int(round(float(bound[:-1]) * float(frame_rate))))
        else:
            frames.append(int(bound))
    low, high = min(frames), max(frames)
    return low, high


def spread_reference_offsets(low, high, count):
    """Test-time offsets: count frames spread evenly over [low, high], nearest first.

    Upstream spreads its test references the same way over its window.
    """
    if count <= 0:
        return []
    if count == 1 or low == high:
        return [high] * count
    step = (high - low) / float(count - 1)
    return [int(round(high - i * step)) for i in range(count)]


def causal_reference_offsets(low, high, count):
    """Spread offsets restricted to past frames, for validation and the detector."""
    high = min(high, -1)
    low = min(low, high)
    return spread_reference_offsets(low, high, count)


def natural_sort_key(name):
    """Sort key ordering frame2 before frame10."""
    return [int(tok) if tok.isdigit() else tok for tok in re.split(r'(\d+)', name)]


def sequence_frame_rate(folder):
    """(frames/s, source) for a folder of frames, or (None, None) when unknown.

    Same sources as the RF-DETR trainer: a "<folder>_frame_rate.txt" sidecar
    from video extraction, else "fps: N" in a groundtruth csv header.
    """
    folder = folder.rstrip(os.sep)
    try:
        with open(folder + "_frame_rate.txt") as fh:
            rate = float(fh.read().split()[0])
        if rate > 0:
            return rate, "extraction"
    except (OSError, ValueError, IndexError):
        pass
    try:
        csvs = [n for n in os.listdir(folder) if n.lower().endswith('.csv')]
    except OSError:
        csvs = []
    for csv_name in sorted(csvs):
        try:
            with open(os.path.join(folder, csv_name)) as fh:
                for line in fh:
                    if not line.startswith('#'):
                        break
                    match = re.search(r'fps[:=]\s*([0-9.]+)', line)
                    if match and float(match.group(1)) > 0:
                        return float(match.group(1)), "csv metadata"
        except (OSError, ValueError):
            continue
    return None, None


def preprocess_frame(rgb, resolution, device, flip=False):
    """HxWx3 uint8 RGB array -> normalised 1x3xHxW tensor resized to resolution."""
    import torch
    import torch.nn.functional as F

    tensor = torch.from_numpy(rgb).to(device).permute(2, 0, 1)[None].float() / 255.0
    if flip:
        tensor = tensor.flip(-1)
    if tuple(tensor.shape[-2:]) != tuple(resolution):
        tensor = F.interpolate(tensor, size=tuple(resolution), mode="bilinear",
                               align_corners=False, antialias=True)
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    return (tensor - mean) / std


def to_nested(tensor):
    """Wrap a 1x3xHxW tensor as an unpadded upstream NestedTensor."""
    import torch

    mask = torch.zeros(
        (tensor.shape[0],) + tuple(tensor.shape[-2:]), dtype=torch.bool,
        device=tensor.device)
    return load_upstream().NestedTensor(tensor, mask)


def decode_outputs(outputs, max_detections=100):
    """Model outputs -> (boxes xyxy in [0, 1], per-class scores) of the best queries."""
    import torch

    scores = outputs["pred_logits"][0].float().sigmoid()
    boxes = load_upstream().box_ops.box_cxcywh_to_xyxy(outputs["pred_boxes"][0].float())
    best = scores.max(dim=1).values
    keep = torch.topk(best, min(int(max_detections), best.numel())).indices
    return boxes[keep].clamp(0, 1), scores[keep]
