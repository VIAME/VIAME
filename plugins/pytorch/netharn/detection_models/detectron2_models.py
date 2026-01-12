# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
Detectron2 Detection Models for Netharn

This module provides netharn-compatible wrappers for Detectron2 models,
following the same pattern as mm_models.py for MMDetection.

Supports Faster R-CNN, Mask R-CNN, RetinaNet, and other Detectron2 architectures.

Example:
    >>> # xdoctest: +SKIP
    >>> from viame.pytorch.netharn.detection_models import detectron2_models
    >>> model = detectron2_models.Detectron2_FasterRCNN(['person', 'car', 'bike'])
    >>> batch = model.demo_batch(bsize=2, h=800, w=1200)
    >>> outputs = model.forward(batch, return_loss=True, return_result=True)
    >>> assert 'loss_parts' in outputs
    >>> assert 'batch_results' in outputs
"""
import numpy as np
import ubelt as ub
import torch
import kwimage
import kwarray
from collections import OrderedDict
from viame.pytorch import netharn as nh
from viame.pytorch.netharn.data.channel_spec import ChannelSpec
from viame.pytorch.netharn.data import data_containers


class Detectron2_Coder:
    """
    Standardize Detectron2 network outputs to kwimage.Detections format.

    Converts the Detectron2 output format (list of dicts with 'instances')
    into kwimage.Detections objects for consistent downstream processing.

    Example:
        >>> # xdoctest: +SKIP
        >>> classes = ['person', 'car', 'bike']
        >>> coder = Detectron2_Coder(classes)
        >>> # mock_outputs simulates Detectron2 inference output
        >>> dets = coder.decode_batch(outputs)
    """

    def __init__(self, classes, score_thresh=0.0):
        """
        Args:
            classes: List of class names or kwcoco.CategoryTree
            score_thresh: Minimum score threshold for detections
        """
        import kwcoco
        self.classes = kwcoco.CategoryTree.coerce(classes)
        self.score_thresh = score_thresh

    def decode_batch(self, outputs):
        """
        Transform Detectron2 outputs into a list of kwimage.Detections objects.

        Args:
            outputs (Dict): dict containing 'batch_results' which is a list of
                dicts with key 'instances' (Detectron2 Instances object)

        Returns:
            List[kwimage.Detections]: One detection object per batch item
        """
        batch_results = outputs['batch_results']
        batch_dets = []

        for result in batch_results:
            if result is None:
                det = kwimage.Detections(
                    boxes=kwimage.Boxes(np.empty((0, 4)), 'ltrb'),
                    scores=np.array([]),
                    class_idxs=np.array([], dtype=int),
                    classes=self.classes
                )
                batch_dets.append(det)
                continue

            # Handle Detectron2 output format
            if isinstance(result, dict) and 'instances' in result:
                instances = result['instances']
            else:
                instances = result

            # Extract fields from Instances object
            if hasattr(instances, 'pred_boxes') and len(instances) > 0:
                boxes = instances.pred_boxes.tensor.detach().cpu().numpy()
                scores = instances.scores.detach().cpu().numpy()
                class_idxs = instances.pred_classes.detach().cpu().numpy().astype(int)

                # Apply score threshold
                keep = scores >= self.score_thresh
                boxes = boxes[keep]
                scores = scores[keep]
                class_idxs = class_idxs[keep]

                # Handle masks if present
                masks = None
                if hasattr(instances, 'pred_masks') and instances.pred_masks is not None:
                    masks_tensor = instances.pred_masks[keep]
                    if len(masks_tensor) > 0:
                        masks = masks_tensor.detach().cpu().numpy()

                det = kwimage.Detections(
                    boxes=kwimage.Boxes(boxes, 'ltrb'),  # Detectron2 uses XYXY format
                    scores=scores,
                    class_idxs=class_idxs,
                    classes=self.classes,
                    segmentations=masks if masks is not None else None,
                )
            else:
                det = kwimage.Detections(
                    boxes=kwimage.Boxes(np.empty((0, 4)), 'ltrb'),
                    scores=np.array([]),
                    class_idxs=np.array([], dtype=int),
                    classes=self.classes
                )

            batch_dets.append(det)

        return batch_dets


def _batch_to_detectron2_inputs(batch, device=None):
    """
    Convert netharn-style batch to Detectron2 input format.

    Detectron2 expects a list of dicts, one per image:
        - 'image': Tensor (C, H, W), values in [0, 255] or normalized
        - 'instances': Instances object with gt_boxes, gt_classes (training only)
        - 'height': int, original image height
        - 'width': int, original image width

    Args:
        batch: Netharn batch which can be:
            - A raw tensor (B, C, H, W)
            - A dict with 'inputs' key containing BatchContainer
        device: Device to place tensors on

    Returns:
        List[Dict]: Detectron2-style batched inputs
    """
    from detectron2.structures import Boxes, Instances

    # Handle raw tensor input
    if isinstance(batch, torch.Tensor):
        B, C, H, W = batch.shape
        batched_inputs = []
        for i in range(B):
            batched_inputs.append({
                'image': batch[i],
                'height': H,
                'width': W,
            })
        return batched_inputs

    # Handle BatchContainer input without labels
    if isinstance(batch, data_containers.BatchContainer):
        if batch.stack:
            images = batch.data[0] if len(batch.data) == 1 else torch.cat(batch.data, dim=0)
        else:
            images = torch.stack([d for d in ub.flatten(batch.data)])
        B, C, H, W = images.shape
        batched_inputs = []
        for i in range(B):
            batched_inputs.append({
                'image': images[i],
                'height': H,
                'width': W,
            })
        return batched_inputs

    # Extract images from dict batch
    if 'inputs' in batch:
        inputs = batch['inputs']
        if isinstance(inputs, dict):
            main_key = 'rgb' if 'rgb' in inputs else list(inputs.keys())[0]
            imgs_container = inputs[main_key]
        else:
            imgs_container = inputs

        # Unwrap BatchContainer
        if isinstance(imgs_container, data_containers.BatchContainer):
            if imgs_container.stack:
                images = imgs_container.data[0]
            else:
                images = torch.stack(list(ub.flatten(imgs_container.data)))
        elif isinstance(imgs_container, torch.Tensor):
            images = imgs_container
        else:
            raise TypeError(f"Unexpected input type: {type(imgs_container)}")
    else:
        raise ValueError("Batch must contain 'inputs' key")

    if device is not None:
        images = images.to(device)

    B, C, H, W = images.shape

    # Build batched_inputs list
    batched_inputs = []

    # Check if we have labels
    has_labels = 'label' in batch
    label = batch.get('label', {})

    # Get boxes container
    boxes_container = None
    box_format = None
    if 'tlbr' in label:
        boxes_container = label['tlbr']
        box_format = 'tlbr'
    elif 'cxywh' in label:
        boxes_container = label['cxywh']
        box_format = 'cxywh'

    class_container = label.get('class_idxs')
    weight_container = label.get('weight')

    # Unwrap label containers if present
    all_boxes_lists = []
    all_cidxs_lists = []
    all_weights_lists = []

    if boxes_container is not None:
        for device_idx, (device_data_boxes, device_data_cidxs) in enumerate(
                zip(boxes_container.data, class_container.data)):
            if isinstance(device_data_boxes, (list, tuple)):
                all_boxes_lists.extend(device_data_boxes)
                all_cidxs_lists.extend(device_data_cidxs)
                if weight_container is not None and device_idx < len(weight_container.data):
                    device_weights = weight_container.data[device_idx]
                    if isinstance(device_weights, (list, tuple)):
                        all_weights_lists.extend(device_weights)
                    else:
                        all_weights_lists.append(device_weights)
            else:
                all_boxes_lists.append(device_data_boxes)
                all_cidxs_lists.append(device_data_cidxs)
                if weight_container is not None and device_idx < len(weight_container.data):
                    all_weights_lists.append(weight_container.data[device_idx])

    # Build inputs for each image
    for bx in range(B):
        input_dict = {
            'image': images[bx],
            'height': H,
            'width': W,
        }

        # Add instances if we have labels
        if has_labels and bx < len(all_boxes_lists):
            boxes = all_boxes_lists[bx]
            cidxs = all_cidxs_lists[bx]

            if boxes is not None and len(boxes) > 0:
                # Convert to tensor
                if not isinstance(boxes, torch.Tensor):
                    boxes = torch.tensor(boxes, dtype=torch.float32, device=images.device)
                else:
                    boxes = boxes.to(device=images.device, dtype=torch.float32)

                if not isinstance(cidxs, torch.Tensor):
                    cidxs = torch.tensor(cidxs, dtype=torch.int64, device=images.device)
                else:
                    cidxs = cidxs.to(device=images.device, dtype=torch.int64)

                # Apply weight filtering
                if weight_container is not None and bx < len(all_weights_lists):
                    weights = all_weights_lists[bx]
                    if weights is not None:
                        if not isinstance(weights, torch.Tensor):
                            weights = torch.tensor(weights, device=images.device)
                        else:
                            weights = weights.to(device=images.device)
                        valid_mask = weights >= 0.1
                        boxes = boxes[valid_mask]
                        cidxs = cidxs[valid_mask]

                # Convert boxes to XYXY format if needed
                if box_format == 'cxywh' and len(boxes) > 0:
                    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
                    x1 = cx - w / 2
                    y1 = cy - h / 2
                    x2 = cx + w / 2
                    y2 = cy + h / 2
                    boxes = torch.stack([x1, y1, x2, y2], dim=1)

                # Create Instances object
                if len(boxes) > 0:
                    instances = Instances((H, W))
                    instances.gt_boxes = Boxes(boxes)
                    instances.gt_classes = cidxs
                    input_dict['instances'] = instances
                else:
                    instances = Instances((H, W))
                    instances.gt_boxes = Boxes(torch.zeros(0, 4, device=images.device))
                    instances.gt_classes = torch.zeros(0, dtype=torch.int64, device=images.device)
                    input_dict['instances'] = instances
            else:
                instances = Instances((H, W))
                instances.gt_boxes = Boxes(torch.zeros(0, 4, device=images.device))
                instances.gt_classes = torch.zeros(0, dtype=torch.int64, device=images.device)
                input_dict['instances'] = instances

        batched_inputs.append(input_dict)

    return batched_inputs


def _demo_batch_detectron2(bsize=1, channels='rgb', h=800, w=1200, classes=3, with_mask=False):
    """
    Generate a demo batch for testing Detectron2 detectors.

    Args:
        bsize: Batch size
        channels: Channel specification (e.g., 'rgb')
        h: Image height
        w: Image width
        classes: Number of classes or list of class names
        with_mask: Whether to include masks

    Returns:
        Dict: Netharn-style batch with inputs and labels
    """
    rng = kwarray.ensure_rng(0)
    if isinstance(bsize, list):
        item_sizes = bsize
        bsize = len(item_sizes)
    else:
        item_sizes = [rng.randint(1, 10) for _ in range(bsize)]

    channels = ChannelSpec.coerce(channels)
    B, H, W = bsize, h, w

    # Create input tensors
    input_shapes = {
        key: (B, c, H, W)
        for key, c in channels.sizes().items()
    }
    inputs = {
        key: torch.rand(*shape) * 255  # Detectron2 expects [0, 255]
        for key, shape in input_shapes.items()
    }

    batch_items = []
    for bx in range(B):
        num_dets = item_sizes[bx]
        dets = kwimage.Detections.random(num=num_dets, classes=classes)
        dets = dets.scale((W, H))
        dets = dets.tensor()

        label = {
            'tlbr': data_containers.ItemContainer(
                dets.boxes.to_ltrb().data.float(), stack=False),
            'class_idxs': data_containers.ItemContainer(
                dets.class_idxs, stack=False),
            'weight': data_containers.ItemContainer(
                torch.ones(len(dets), dtype=torch.float32), stack=False),
        }

        item = {
            'inputs': {
                key: data_containers.ItemContainer(vals[bx], stack=True)
                for key, vals in inputs.items()
            },
            'label': label,
        }
        batch_items.append(item)

    batch = data_containers.container_collate(batch_items, num_devices=1)
    return batch


def _create_faster_rcnn_config(num_classes, backbone='R50', fpn=True, pretrained=False):
    """
    Create a Faster R-CNN configuration.

    Args:
        num_classes: Number of foreground classes
        backbone: Backbone type ('R50', 'R101', 'R152')
        fpn: Use Feature Pyramid Network
        pretrained: Path to pretrained weights or False

    Returns:
        CfgNode: Detectron2 config
    """
    from detectron2.config import get_cfg

    cfg = get_cfg()

    # Model architecture
    cfg.MODEL.META_ARCHITECTURE = "GeneralizedRCNN"
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Backbone
    depth_map = {'R50': 50, 'R101': 101, 'R152': 152}
    depth = depth_map.get(backbone, 50)

    if fpn:
        cfg.MODEL.BACKBONE.NAME = "build_resnet_fpn_backbone"
        cfg.MODEL.RESNETS.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
        cfg.MODEL.FPN.IN_FEATURES = ["res2", "res3", "res4", "res5"]
        cfg.MODEL.FPN.OUT_CHANNELS = 256
        cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[32], [64], [128], [256], [512]]
        cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]
        cfg.MODEL.RPN.IN_FEATURES = ["p2", "p3", "p4", "p5", "p6"]
        cfg.MODEL.ROI_HEADS.NAME = "StandardROIHeads"
        cfg.MODEL.ROI_HEADS.IN_FEATURES = ["p2", "p3", "p4", "p5"]
        cfg.MODEL.ROI_BOX_HEAD.NAME = "FastRCNNConvFCHead"
        cfg.MODEL.ROI_BOX_HEAD.NUM_FC = 2
        cfg.MODEL.ROI_BOX_HEAD.FC_DIM = 1024
        cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = 7
    else:
        cfg.MODEL.BACKBONE.NAME = "build_resnet_backbone"
        cfg.MODEL.RESNETS.OUT_FEATURES = ["res4"]
        cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[32, 64, 128, 256, 512]]
        cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]
        cfg.MODEL.RPN.IN_FEATURES = ["res4"]
        cfg.MODEL.ROI_HEADS.NAME = "Res5ROIHeads"
        cfg.MODEL.ROI_HEADS.IN_FEATURES = ["res4"]

    cfg.MODEL.RESNETS.DEPTH = depth
    cfg.MODEL.RESNETS.NORM = "FrozenBN"
    cfg.MODEL.BACKBONE.FREEZE_AT = 2

    # Proposal generator
    cfg.MODEL.PROPOSAL_GENERATOR.NAME = "RPN"
    cfg.MODEL.RPN.HEAD_NAME = "StandardRPNHead"
    cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 2000 if fpn else 12000
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 1000 if fpn else 6000
    cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 1000 if fpn else 2000
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 1000

    # ROI heads
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.25
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5

    # Input
    cfg.INPUT.FORMAT = "BGR"
    cfg.INPUT.MIN_SIZE_TRAIN = (640, 672, 704, 736, 768, 800)
    cfg.INPUT.MAX_SIZE_TRAIN = 1333
    cfg.INPUT.MIN_SIZE_TEST = 800
    cfg.INPUT.MAX_SIZE_TEST = 1333

    # Pixel normalization (ImageNet BGR)
    cfg.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675]
    cfg.MODEL.PIXEL_STD = [1.0, 1.0, 1.0]

    # Disable masks and keypoints
    cfg.MODEL.MASK_ON = False
    cfg.MODEL.KEYPOINT_ON = False

    # Pretrained weights
    if pretrained:
        cfg.MODEL.WEIGHTS = pretrained
    else:
        cfg.MODEL.WEIGHTS = ""

    # Test settings
    cfg.TEST.DETECTIONS_PER_IMAGE = 100

    return cfg


def _create_mask_rcnn_config(num_classes, backbone='R50', pretrained=False):
    """
    Create a Mask R-CNN configuration.

    Args:
        num_classes: Number of foreground classes
        backbone: Backbone type ('R50', 'R101', 'R152')
        pretrained: Path to pretrained weights or False

    Returns:
        CfgNode: Detectron2 config
    """
    cfg = _create_faster_rcnn_config(num_classes, backbone, fpn=True, pretrained=pretrained)

    # Enable masks
    cfg.MODEL.MASK_ON = True
    cfg.MODEL.ROI_MASK_HEAD.NAME = "MaskRCNNConvUpsampleHead"
    cfg.MODEL.ROI_MASK_HEAD.NUM_CONV = 4
    cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION = 14

    return cfg


def _create_retinanet_config(num_classes, backbone='R50', pretrained=False):
    """
    Create a RetinaNet configuration.

    Args:
        num_classes: Number of foreground classes
        backbone: Backbone type ('R50', 'R101', 'R152')
        pretrained: Path to pretrained weights or False

    Returns:
        CfgNode: Detectron2 config
    """
    from detectron2.config import get_cfg

    cfg = get_cfg()

    # Model architecture
    cfg.MODEL.META_ARCHITECTURE = "RetinaNet"
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Backbone with FPN
    depth_map = {'R50': 50, 'R101': 101, 'R152': 152}
    depth = depth_map.get(backbone, 50)

    cfg.MODEL.BACKBONE.NAME = "build_retinanet_resnet_fpn_backbone"
    cfg.MODEL.RESNETS.DEPTH = depth
    cfg.MODEL.RESNETS.OUT_FEATURES = ["res3", "res4", "res5"]
    cfg.MODEL.RESNETS.NORM = "FrozenBN"
    cfg.MODEL.BACKBONE.FREEZE_AT = 2

    # FPN
    cfg.MODEL.FPN.IN_FEATURES = ["res3", "res4", "res5"]
    cfg.MODEL.FPN.OUT_CHANNELS = 256

    # RetinaNet head
    cfg.MODEL.RETINANET.NUM_CLASSES = num_classes
    cfg.MODEL.RETINANET.IN_FEATURES = ["p3", "p4", "p5", "p6", "p7"]
    cfg.MODEL.RETINANET.NUM_CONVS = 4
    cfg.MODEL.RETINANET.IOU_THRESHOLDS = [0.4, 0.5]
    cfg.MODEL.RETINANET.IOU_LABELS = [0, -1, 1]
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.05
    cfg.MODEL.RETINANET.TOPK_CANDIDATES_TEST = 1000
    cfg.MODEL.RETINANET.NMS_THRESH_TEST = 0.5
    cfg.MODEL.RETINANET.FOCAL_LOSS_GAMMA = 2.0
    cfg.MODEL.RETINANET.FOCAL_LOSS_ALPHA = 0.25
    cfg.MODEL.RETINANET.SMOOTH_L1_LOSS_BETA = 0.1

    # Anchors
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[32], [64], [128], [256], [512]]
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]

    # Input
    cfg.INPUT.FORMAT = "BGR"
    cfg.INPUT.MIN_SIZE_TRAIN = (640, 672, 704, 736, 768, 800)
    cfg.INPUT.MAX_SIZE_TRAIN = 1333
    cfg.INPUT.MIN_SIZE_TEST = 800
    cfg.INPUT.MAX_SIZE_TEST = 1333

    # Pixel normalization
    cfg.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675]
    cfg.MODEL.PIXEL_STD = [1.0, 1.0, 1.0]

    # Pretrained weights
    if pretrained:
        cfg.MODEL.WEIGHTS = pretrained
    else:
        cfg.MODEL.WEIGHTS = ""

    # Test settings
    cfg.TEST.DETECTIONS_PER_IMAGE = 100

    return cfg


class Detectron2_Detector(nh.layers.Module):
    """
    Netharn wrapper for Detectron2 detection models.

    Supports Faster R-CNN, Mask R-CNN, RetinaNet, and other Detectron2 architectures.
    Provides a unified interface compatible with netharn's training harness.

    Example:
        >>> # xdoctest: +SKIP
        >>> model = Detectron2_Detector(['person', 'car'], model_type='faster_rcnn')
        >>> batch = model.demo_batch(bsize=2, h=800, w=1200)
        >>> outputs = model.forward(batch, return_loss=True, return_result=True)
        >>> assert 'loss_parts' in outputs
        >>> assert 'batch_results' in outputs

    Attributes:
        model: The underlying Detectron2 model
        coder: Output decoder to kwimage.Detections
        cfg: Detectron2 CfgNode configuration
    """

    __BUILTIN_CRITERION__ = True

    def __init__(self, classes, channels='rgb', input_stats=None,
                 model_type='faster_rcnn', backbone='R50', fpn=True,
                 weight_path=None, score_thresh=0.05, cfg=None):
        """
        Args:
            classes: List of class names or kwcoco.CategoryTree
            channels: Input channel specification (default 'rgb')
            input_stats: Dict with 'mean' and 'std' for input normalization
                         (Note: Detectron2 handles normalization internally)
            model_type: Model architecture ('faster_rcnn', 'mask_rcnn', 'retinanet')
            backbone: Backbone type ('R50', 'R101', 'R152')
            fpn: Use Feature Pyramid Network (for faster_rcnn only)
            weight_path: Path to pretrained weights or False/None
            score_thresh: Score threshold for inference
            cfg: Optional pre-configured CfgNode (overrides other settings)
        """
        super().__init__()
        import kwcoco
        from detectron2.modeling import build_model

        # Store initialization kwargs for serialization
        self._initkw = {
            'classes': classes,
            'channels': channels,
            'input_stats': input_stats,
            'model_type': model_type,
            'backbone': backbone,
            'fpn': fpn,
            'weight_path': weight_path,
            'score_thresh': score_thresh,
        }

        # Setup classes
        self.classes = kwcoco.CategoryTree.coerce(classes)
        self.num_classes = len(self.classes)

        # Setup channels
        self.channels = ChannelSpec.coerce(channels)
        chann_norm = self.channels.normalize()
        assert len(chann_norm) == 1, "Detectron2 only supports single input stream"
        self.in_channels = len(ub.peek(chann_norm.values()))
        assert self.in_channels == 3, "Detectron2 requires 3-channel (RGB/BGR) input"

        # Input normalization is handled by Detectron2 internally
        # We use a no-op normalizer since images should be in [0, 255]
        self.input_norm = nh.layers.InputNorm()  # No-op

        self.model_type = model_type.lower()
        self.score_thresh = score_thresh

        # Build or use provided config
        if cfg is not None:
            self.cfg = cfg
        else:
            if self.model_type == 'faster_rcnn':
                self.cfg = _create_faster_rcnn_config(
                    self.num_classes, backbone, fpn, weight_path
                )
            elif self.model_type == 'mask_rcnn':
                self.cfg = _create_mask_rcnn_config(
                    self.num_classes, backbone, weight_path
                )
            elif self.model_type == 'retinanet':
                self.cfg = _create_retinanet_config(
                    self.num_classes, backbone, weight_path
                )
            else:
                raise ValueError(f"Unknown model_type: {model_type}")

        # Update score threshold in config
        if hasattr(self.cfg.MODEL, 'ROI_HEADS'):
            self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
        if hasattr(self.cfg.MODEL, 'RETINANET'):
            self.cfg.MODEL.RETINANET.SCORE_THRESH_TEST = score_thresh

        # Build model
        self.model = build_model(self.cfg)

        # Output decoder
        self.coder = Detectron2_Coder(self.classes, score_thresh=score_thresh)

    def demo_batch(self, bsize=3, h=800, w=1200):
        """
        Generate a demo batch for testing.

        Args:
            bsize: Batch size
            h: Image height
            w: Image width

        Returns:
            Dict: Netharn-style batch with inputs and labels
        """
        return _demo_batch_detectron2(
            bsize=bsize,
            channels=self.channels,
            h=h,
            w=w,
            classes=self.num_classes
        )

    def forward(self, batch, return_loss=True, return_result=True):
        """
        Forward pass with netharn-style interface.

        Args:
            batch: Netharn batch dict or raw tensor
            return_loss: Compute training loss (requires labels in batch)
            return_result: Compute detection results

        Returns:
            Dict containing:
                - 'loss_parts': OrderedDict of loss components (if return_loss and labels present)
                - 'batch_results': List of detection dicts (if return_result)
        """
        # Convert batch format
        batched_inputs = _batch_to_detectron2_inputs(batch)

        # Ensure images are in expected format [0, 255]
        # Detectron2 handles normalization internally
        for inp in batched_inputs:
            img = inp['image']
            # If normalized to [0, 1], scale to [0, 255]
            if img.max() <= 1.0:
                inp['image'] = img * 255.0

        outputs = {}

        # Training mode: compute losses
        if return_loss and self.training:
            # Check if instances are present
            has_instances = all('instances' in inp for inp in batched_inputs)

            if has_instances:
                self.model.train()
                loss_dict = self.model(batched_inputs)

                # Format loss_parts as OrderedDict of scalar tensors
                loss_parts = OrderedDict()
                total_loss = 0
                for name, value in loss_dict.items():
                    if isinstance(value, torch.Tensor):
                        loss_parts[name] = value.mean().unsqueeze(0)
                        total_loss = total_loss + value.mean()
                    else:
                        loss_parts[name] = torch.tensor([value], device=self.model.device)
                        total_loss = total_loss + value

                loss_parts['loss_total'] = total_loss.unsqueeze(0) if isinstance(total_loss, torch.Tensor) else torch.tensor([total_loss])
                outputs['loss_parts'] = loss_parts

        # Inference mode: compute predictions
        if return_result:
            self.model.eval()
            with torch.no_grad():
                # Remove instances for inference
                inference_inputs = []
                for inp in batched_inputs:
                    inference_inp = {k: v for k, v in inp.items() if k != 'instances'}
                    inference_inputs.append(inference_inp)

                results = self.model(inference_inputs)
                outputs['batch_results'] = results

        return outputs


class Detectron2_FasterRCNN(Detectron2_Detector):
    """Detectron2 Faster R-CNN detector."""

    def __init__(self, classes, channels='rgb', input_stats=None,
                 backbone='R50', fpn=True, weight_path=None, score_thresh=0.05):
        super().__init__(
            classes=classes,
            channels=channels,
            input_stats=input_stats,
            model_type='faster_rcnn',
            backbone=backbone,
            fpn=fpn,
            weight_path=weight_path,
            score_thresh=score_thresh,
        )


class Detectron2_FasterRCNN_R50_FPN(Detectron2_Detector):
    """Detectron2 Faster R-CNN with ResNet-50 FPN backbone."""

    def __init__(self, classes, channels='rgb', input_stats=None,
                 weight_path=None, score_thresh=0.05):
        super().__init__(
            classes=classes,
            channels=channels,
            input_stats=input_stats,
            model_type='faster_rcnn',
            backbone='R50',
            fpn=True,
            weight_path=weight_path,
            score_thresh=score_thresh,
        )


class Detectron2_FasterRCNN_R101_FPN(Detectron2_Detector):
    """Detectron2 Faster R-CNN with ResNet-101 FPN backbone."""

    def __init__(self, classes, channels='rgb', input_stats=None,
                 weight_path=None, score_thresh=0.05):
        super().__init__(
            classes=classes,
            channels=channels,
            input_stats=input_stats,
            model_type='faster_rcnn',
            backbone='R101',
            fpn=True,
            weight_path=weight_path,
            score_thresh=score_thresh,
        )


class Detectron2_MaskRCNN(Detectron2_Detector):
    """Detectron2 Mask R-CNN detector."""

    def __init__(self, classes, channels='rgb', input_stats=None,
                 backbone='R50', weight_path=None, score_thresh=0.05):
        super().__init__(
            classes=classes,
            channels=channels,
            input_stats=input_stats,
            model_type='mask_rcnn',
            backbone=backbone,
            fpn=True,  # Mask R-CNN always uses FPN
            weight_path=weight_path,
            score_thresh=score_thresh,
        )


class Detectron2_RetinaNet(Detectron2_Detector):
    """Detectron2 RetinaNet detector."""

    def __init__(self, classes, channels='rgb', input_stats=None,
                 backbone='R50', weight_path=None, score_thresh=0.05):
        super().__init__(
            classes=classes,
            channels=channels,
            input_stats=input_stats,
            model_type='retinanet',
            backbone=backbone,
            fpn=True,  # RetinaNet always uses FPN
            weight_path=weight_path,
            score_thresh=score_thresh,
        )
