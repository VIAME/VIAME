# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
MIT-YOLO Detection Models for Netharn

This module provides netharn-compatible wrappers for MIT-YOLO models,
following the same pattern as mm_models.py for MMDetection.

Supports v9-c, v9-s, v9-m, and other MIT-YOLO variants.

Example:
    >>> # xdoctest: +SKIP
    >>> from viame.pytorch.netharn.detection_models import mityolo_models
    >>> model = mityolo_models.MitYOLO_V9C(['person', 'car', 'bike'], weight_path=False)
    >>> batch = model.demo_batch(bsize=2, h=640, w=640)
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


class MitYOLO_Coder:
    """
    Standardize MIT-YOLO network outputs to kwimage.Detections format.

    Converts the NMS outputs (list of tensors with [class_id, x1, y1, x2, y2, score])
    into kwimage.Detections objects for consistent downstream processing.

    Example:
        >>> # xdoctest: +SKIP
        >>> classes = ['person', 'car', 'bike']
        >>> coder = MitYOLO_Coder(classes)
        >>> mock_outputs = {
        ...     'batch_results': [
        ...         torch.tensor([[0, 10, 20, 50, 60, 0.9], [1, 100, 100, 200, 200, 0.8]])
        ...     ]
        ... }
        >>> dets = coder.decode_batch(mock_outputs)
        >>> assert len(dets) == 1
        >>> assert len(dets[0]) == 2
    """

    def __init__(self, classes, nms_cfg=None):
        """
        Args:
            classes: List of class names or kwcoco.CategoryTree
            nms_cfg: Optional NMS configuration dict with keys:
                - min_confidence: Minimum confidence threshold
                - min_iou: IoU threshold for NMS
                - max_bbox: Maximum detections per image
        """
        import kwcoco
        self.classes = kwcoco.CategoryTree.coerce(classes)
        self.nms_cfg = nms_cfg or {
            'min_confidence': 0.05,
            'min_iou': 0.5,
            'max_bbox': 100,
        }

    def decode_batch(self, outputs):
        """
        Transform MIT-YOLO NMS outputs into a list of kwimage.Detections objects.

        Args:
            outputs (Dict): dict containing 'batch_results' which is a list of
                tensors, each of shape (num_dets, 6) with [class_id, x1, y1, x2, y2, score]

        Returns:
            List[kwimage.Detections]: One detection object per batch item
        """
        batch_results = outputs['batch_results']
        batch_dets = []

        for result in batch_results:
            # result shape: (num_dets, 6) -> [class_id, x1, y1, x2, y2, score]
            if result is None or len(result) == 0:
                det = kwimage.Detections(
                    boxes=kwimage.Boxes(np.empty((0, 4)), 'ltrb'),
                    scores=np.array([]),
                    class_idxs=np.array([], dtype=int),
                    classes=self.classes
                )
            else:
                if isinstance(result, torch.Tensor):
                    result_np = result.detach().cpu().numpy()
                else:
                    result_np = np.asarray(result)

                pred_cidxs = result_np[:, 0].astype(int)
                pred_ltrb = result_np[:, 1:5]  # x1, y1, x2, y2 = ltrb format
                pred_score = result_np[:, 5]

                det = kwimage.Detections(
                    boxes=kwimage.Boxes(pred_ltrb, 'ltrb'),
                    scores=pred_score,
                    class_idxs=pred_cidxs,
                    classes=self.classes
                )
            batch_dets.append(det)

        return batch_dets


def _batch_to_mityolo_inputs(batch, device=None):
    """
    Convert netharn-style batch to MIT-YOLO format.

    Args:
        batch: Netharn batch which can be:
            - A raw tensor (B, C, H, W)
            - A dict with:
                - inputs.rgb: BatchContainer of images
                - label.tlbr or label.cxywh: BatchContainer of boxes
                - label.class_idxs: BatchContainer of class indices
                - label.weight: BatchContainer of weights (filter boxes with weight < 0.1)
        device: Optional device to move tensors to

    Returns:
        Dict containing:
            - 'images': Tensor (B, C, H, W)
            - 'targets': Tensor (B, max_targets, 5) with [cls, x1, y1, x2, y2]
            - 'batch_size': int
    """
    # Handle raw tensor input (inference only)
    if isinstance(batch, torch.Tensor):
        B, C, H, W = batch.shape
        return {
            'images': batch,
            'targets': torch.zeros(B, 0, 5, device=batch.device),
            'batch_size': B,
        }

    # Handle BatchContainer input without labels
    if isinstance(batch, data_containers.BatchContainer):
        if batch.stack:
            images = batch.data[0] if len(batch.data) == 1 else torch.cat(batch.data, dim=0)
        else:
            images = torch.stack([d for d in ub.flatten(batch.data)])
        B = images.shape[0]
        return {
            'images': images,
            'targets': torch.zeros(B, 0, 5, device=images.device),
            'batch_size': B,
        }

    # Extract images from dict batch
    if 'inputs' in batch:
        inputs = batch['inputs']
        if isinstance(inputs, dict):
            # Find main input stream (prefer rgb)
            main_key = 'rgb' if 'rgb' in inputs else list(inputs.keys())[0]
            imgs_container = inputs[main_key]
        else:
            imgs_container = inputs

        # Unwrap BatchContainer to get image tensor
        if isinstance(imgs_container, data_containers.BatchContainer):
            if imgs_container.stack:
                # Stacked tensor: data[0] is already (B, C, H, W)
                images = imgs_container.data[0]
            else:
                # List of tensors: stack them
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

    # Process labels if present
    if 'label' not in batch:
        targets = torch.zeros(B, 0, 5, device=images.device)
    else:
        label = batch['label']

        # Get boxes container - prefer tlbr, fall back to cxywh
        box_format = None
        if 'tlbr' in label:
            boxes_container = label['tlbr']
            box_format = 'tlbr'
        elif 'cxywh' in label:
            boxes_container = label['cxywh']
            box_format = 'cxywh'
        else:
            # No boxes, return empty targets
            targets = torch.zeros(B, 0, 5, device=images.device)
            return {
                'images': images,
                'targets': targets,
                'batch_size': B,
            }

        class_container = label['class_idxs']
        weight_container = label.get('weight', None)

        # Unwrap containers and compute max boxes
        # BatchContainer.data is a list (one per device), and each element
        # is a list of tensors (one per batch item)
        all_boxes_lists = []
        all_cidxs_lists = []
        all_weights_lists = []

        # Handle nested structure: data[device_idx][batch_idx]
        for device_data_boxes, device_data_cidxs in zip(boxes_container.data, class_container.data):
            if isinstance(device_data_boxes, (list, tuple)):
                all_boxes_lists.extend(device_data_boxes)
                all_cidxs_lists.extend(device_data_cidxs)
                if weight_container is not None:
                    for device_data_weights in weight_container.data:
                        if isinstance(device_data_weights, (list, tuple)):
                            all_weights_lists.extend(device_data_weights)
                        else:
                            all_weights_lists.append(device_data_weights)
                    break  # weights already processed
            else:
                all_boxes_lists.append(device_data_boxes)
                all_cidxs_lists.append(device_data_cidxs)
                if weight_container is not None:
                    all_weights_lists.append(weight_container.data[0] if len(weight_container.data) == 1 else weight_container.data)

        # Compute max number of boxes
        max_boxes = 0
        for boxes in all_boxes_lists:
            if boxes is not None and len(boxes) > 0:
                max_boxes = max(max_boxes, len(boxes))
        max_boxes = max(max_boxes, 1)  # At least 1 to avoid empty tensor issues

        # Build targets tensor: (B, max_boxes, 5) with [class_id, x1, y1, x2, y2]
        targets = torch.zeros(B, max_boxes, 5, device=images.device)
        # Use 0 for padding (MIT-YOLO filters by checking if box area > 0)

        for bx in range(min(B, len(all_boxes_lists))):
            boxes = all_boxes_lists[bx]
            cidxs = all_cidxs_lists[bx]

            if boxes is None or len(boxes) == 0:
                continue

            # Convert to tensor if needed
            if not isinstance(boxes, torch.Tensor):
                boxes = torch.tensor(boxes, dtype=torch.float32, device=images.device)
            else:
                boxes = boxes.to(device=images.device, dtype=torch.float32)

            if not isinstance(cidxs, torch.Tensor):
                cidxs = torch.tensor(cidxs, dtype=torch.float32, device=images.device)
            else:
                cidxs = cidxs.to(device=images.device, dtype=torch.float32)

            # Convert boxes to xyxy (ltrb) format if needed
            if box_format == 'cxywh':
                # cxywh -> xyxy: x1=cx-w/2, y1=cy-h/2, x2=cx+w/2, y2=cy+h/2
                cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
                x1 = cx - w / 2
                y1 = cy - h / 2
                x2 = cx + w / 2
                y2 = cy + h / 2
                boxes = torch.stack([x1, y1, x2, y2], dim=1)
            # tlbr is same as xyxy (ltrb), no conversion needed

            # Apply weight filtering if available
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

            num_boxes = min(len(boxes), max_boxes)
            if num_boxes > 0:
                targets[bx, :num_boxes, 0] = cidxs[:num_boxes]
                targets[bx, :num_boxes, 1:5] = boxes[:num_boxes]

    return {
        'images': images,
        'targets': targets,
        'batch_size': B,
    }


def _demo_batch(bsize=1, channels='rgb', h=640, w=640, classes=3, with_mask=False):
    """
    Generate a demo batch for testing MIT-YOLO detectors.

    Args:
        bsize: Batch size
        channels: Channel specification (e.g., 'rgb')
        h: Image height
        w: Image width
        classes: Number of classes or list of class names
        with_mask: Whether to include masks (not used for YOLO but kept for API compatibility)

    Returns:
        Dict: Netharn-style batch with inputs and labels

    Example:
        >>> # xdoctest: +SKIP
        >>> batch = _demo_batch(bsize=4, h=640, w=640, classes=80)
        >>> assert batch['inputs']['rgb'].data[0].shape == (4, 3, 640, 640)
    """
    rng = kwarray.ensure_rng(0)
    if isinstance(bsize, list):
        item_sizes = bsize
        bsize = len(item_sizes)
    else:
        item_sizes = [rng.randint(1, 10) for _ in range(bsize)]

    channels = ChannelSpec.coerce(channels)
    B, H, W = bsize, h, w

    # Create input tensors for each channel stream
    input_shapes = {
        key: (B, c, H, W)
        for key, c in channels.sizes().items()
    }
    inputs = {
        key: torch.rand(*shape)
        for key, shape in input_shapes.items()
    }

    batch_items = []
    for bx in range(B):
        # Generate random detections
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

    # Collate into batch
    batch = data_containers.container_collate(batch_items, num_devices=1)
    return batch


class MitYOLO_LossWrapper:
    """
    Wrapper for MIT-YOLO loss functions compatible with netharn.

    Handles both DualLoss (with auxiliary head) and single-head loss.
    Computes BCE loss for classification, CIoU loss for boxes, and
    Distribution Focal Loss for regression.

    Example:
        >>> # xdoctest: +SKIP
        >>> from yolo.utils.bounding_box_utils import Vec2Box
        >>> loss_fn = MitYOLO_LossWrapper(loss_cfg, vec2box, class_num=80, reg_max=16)
        >>> loss, loss_dict = loss_fn(aux_preds, main_preds, targets)
    """

    def __init__(self, loss_cfg, vec2box, class_num, reg_max):
        """
        Args:
            loss_cfg: Loss configuration dict with keys:
                - objective: dict with BCELoss, BoxLoss, DFLoss weights
                - aux: auxiliary head weight (default 0.25)
                - matcher: matcher configuration
            vec2box: Vec2Box converter instance
            class_num: Number of classes
            reg_max: Maximum regression value for DFL
        """
        from yolo.tools.loss_functions import YOLOLoss
        from omegaconf import OmegaConf

        # Convert to OmegaConf if needed
        if not hasattr(loss_cfg, 'matcher'):
            loss_cfg = OmegaConf.create(loss_cfg)

        self.loss = YOLOLoss(loss_cfg, vec2box, class_num=class_num, reg_max=reg_max)

        self.aux_rate = loss_cfg.get('aux', 0.25)
        self.iou_rate = loss_cfg.objective.get('BoxLoss', 7.5)
        self.dfl_rate = loss_cfg.objective.get('DFLoss', 1.5)
        self.cls_rate = loss_cfg.objective.get('BCELoss', 0.5)

    def __call__(self, aux_predicts, main_predicts, targets):
        """
        Compute loss.

        Args:
            aux_predicts: Tuple of (pred_cls, pred_anc, pred_box) from AUX head, or None
            main_predicts: Tuple of (pred_cls, pred_anc, pred_box) from Main head
            targets: (B, max_targets, 5) tensor with [cls, x1, y1, x2, y2]

        Returns:
            Tuple[Tensor, Dict]: Total loss and loss components dict
        """
        main_iou, main_dfl, main_cls = self.loss(main_predicts, targets)

        if aux_predicts is not None:
            aux_iou, aux_dfl, aux_cls = self.loss(aux_predicts, targets)
            total_loss = [
                self.iou_rate * (aux_iou * self.aux_rate + main_iou),
                self.dfl_rate * (aux_dfl * self.aux_rate + main_dfl),
                self.cls_rate * (aux_cls * self.aux_rate + main_cls),
            ]
        else:
            total_loss = [
                self.iou_rate * main_iou,
                self.dfl_rate * main_dfl,
                self.cls_rate * main_cls,
            ]

        loss_dict = {
            'loss_box': total_loss[0],
            'loss_dfl': total_loss[1],
            'loss_bce': total_loss[2],
        }

        return sum(total_loss), loss_dict


class MitYOLO_Detector(nh.layers.Module):
    """
    Netharn wrapper for MIT-YOLO detection models.

    Supports v9-c, v9-s, v9-m, v7, and other MIT-YOLO variants.
    Provides a unified interface compatible with netharn's training harness.

    Example:
        >>> # xdoctest: +SKIP
        >>> model = MitYOLO_Detector(['person', 'car'], model_variant='v9-c')
        >>> batch = model.demo_batch(bsize=2, h=640, w=640)
        >>> outputs = model.forward(batch, return_loss=True, return_result=True)
        >>> assert 'loss_parts' in outputs
        >>> assert 'batch_results' in outputs

    Attributes:
        model: The underlying MIT-YOLO model
        vec2box: Vec2Box converter for anchor-based decoding
        loss_fn: Loss function wrapper
        coder: Output decoder to kwimage.Detections
        input_norm: Input normalization layer
    """

    __BUILTIN_CRITERION__ = True

    def __init__(self, classes, channels='rgb', input_stats=None,
                 model_variant='v9-c', weight_path=None, nms_cfg=None,
                 loss_cfg=None):
        """
        Args:
            classes: List of class names or kwcoco.CategoryTree
            channels: Input channel specification (default 'rgb')
            input_stats: Dict with 'mean' and 'std' for input normalization
            model_variant: MIT-YOLO variant name (v9-c, v9-s, v9-m, v7, etc.)
            weight_path: Path to pretrained weights, True to auto-download, False for none
            nms_cfg: NMS configuration dict with min_confidence, min_iou, max_bbox
            loss_cfg: Loss configuration dict (uses defaults if None)
        """
        super().__init__()
        import kwcoco

        # Store initialization kwargs for serialization
        self._initkw = {
            'classes': classes,
            'channels': channels,
            'input_stats': input_stats,
            'model_variant': model_variant,
            'weight_path': weight_path,
            'nms_cfg': nms_cfg,
            'loss_cfg': loss_cfg,
        }

        # Setup classes
        self.classes = kwcoco.CategoryTree.coerce(classes)
        self.num_classes = len(self.classes)

        # Setup channels
        self.channels = ChannelSpec.coerce(channels)
        chann_norm = self.channels.normalize()
        assert len(chann_norm) == 1, "MitYOLO only supports single input stream"
        self.in_channels = len(ub.peek(chann_norm.values()))

        # Setup input normalization
        if input_stats is None:
            input_stats = {}
        if len(input_stats):
            chan_keys = list(self.channels.keys())
            if chan_keys != list(input_stats.keys()):
                if 'mean' not in input_stats and 'std' not in input_stats:
                    raise AssertionError(f'input_stats mismatch: {input_stats}')
                input_stats = {chan_keys[0]: input_stats}
            main_input_stats = ub.peek(input_stats.values())
        else:
            main_input_stats = {}
        self.input_norm = nh.layers.InputNorm(**main_input_stats)

        # NMS configuration
        self.nms_cfg = nms_cfg or {
            'min_confidence': 0.05,
            'min_iou': 0.5,
            'max_bbox': 100,
        }

        # Loss configuration
        self._loss_cfg = loss_cfg or {
            'objective': {'BCELoss': 0.5, 'BoxLoss': 7.5, 'DFLoss': 1.5},
            'aux': 0.25,
            'matcher': {'iou': 'CIoU', 'topk': 10, 'factor': {'iou': 6.0, 'cls': 0.5}}
        }

        # Build the MIT-YOLO model
        self.model_variant = model_variant
        self.model_cfg = self._load_model_config(model_variant)
        self.model = self._create_model(self.num_classes, weight_path)

        # Lazy-initialized components (depend on image size)
        self.vec2box = None
        self._current_image_size = None
        self.loss_fn = None

        # Output decoder
        self.coder = MitYOLO_Coder(self.classes, self.nms_cfg)

    def _load_model_config(self, variant):
        """Load model configuration from YAML via Hydra."""
        from yolo.utils.config_utils import build_config
        # Use hydra to load the full config with model variant
        cfg = build_config(overrides=[f'model={variant}', 'task=train'])
        return cfg.model

    def _create_model(self, num_classes, weight_path):
        """Create the MIT-YOLO model."""
        from yolo.model.yolo import create_model

        # Handle weight_path options
        if weight_path is None or weight_path is False:
            actual_weight_path = False
        elif weight_path is True:
            actual_weight_path = True  # Auto-download
        else:
            actual_weight_path = weight_path

        model = create_model(
            self.model_cfg,
            weight_path=actual_weight_path,
            class_num=num_classes
        )
        return model

    def _ensure_vec2box(self, image_size, device):
        """Ensure Vec2Box is initialized for current image size."""
        if self.vec2box is None or self._current_image_size != tuple(image_size):
            from yolo.utils.bounding_box_utils import Vec2Box
            self.vec2box = Vec2Box(
                self.model,
                self.model_cfg.anchor,
                list(image_size),
                device
            )
            self._current_image_size = tuple(image_size)

    def _ensure_loss_fn(self):
        """Ensure loss function is initialized."""
        if self.loss_fn is None and self.vec2box is not None:
            from omegaconf import OmegaConf
            loss_cfg = OmegaConf.create(self._loss_cfg)
            self.loss_fn = MitYOLO_LossWrapper(
                loss_cfg,
                self.vec2box,
                class_num=self.num_classes,
                reg_max=self.model_cfg.anchor.reg_max
            )

    def demo_batch(self, bsize=3, h=640, w=640):
        """
        Generate a demo batch for testing.

        Args:
            bsize: Batch size
            h: Image height
            w: Image width

        Returns:
            Dict: Netharn-style batch with inputs and labels
        """
        return _demo_batch(
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
                - 'batch_results': List of detection tensors (if return_result)
        """
        # Convert batch format
        mityolo_inputs = _batch_to_mityolo_inputs(batch)
        images = mityolo_inputs['images']
        targets = mityolo_inputs['targets']

        # Get device and image size
        device = images.device
        B, C, H, W = images.shape
        image_size = [W, H]  # MIT-YOLO uses [W, H] convention

        # Ensure Vec2Box is ready for this image size
        self._ensure_vec2box(image_size, device)

        # Apply input normalization
        images_norm = self.input_norm(images)

        # Forward through model
        raw_outputs = self.model(images_norm)

        outputs = {}

        # Compute loss if requested and labels are available
        if return_loss and targets.shape[1] > 0 and targets.sum() > 0:
            self._ensure_loss_fn()

            # Process outputs through Vec2Box
            main_predicts = self.vec2box(raw_outputs['Main'])

            # Check for auxiliary head
            if 'AUX' in raw_outputs:
                aux_predicts = self.vec2box(raw_outputs['AUX'])
            else:
                aux_predicts = None

            # Compute loss
            loss, loss_dict = self.loss_fn(aux_predicts, main_predicts, targets)

            # Format loss_parts as OrderedDict of scalar tensors (for DataParallel)
            loss_parts = OrderedDict()
            for name, value in loss_dict.items():
                if isinstance(value, torch.Tensor):
                    loss_parts[name] = value.mean().unsqueeze(0)
                else:
                    loss_parts[name] = torch.tensor([value], device=device)

            loss_parts['loss_total'] = loss.mean().unsqueeze(0)
            outputs['loss_parts'] = loss_parts

        if return_result:
            with torch.no_grad():
                # Get main predictions
                main_predicts = self.vec2box(raw_outputs['Main'])
                pred_cls, _, pred_bbox = main_predicts

                # Apply NMS
                from yolo.utils.bounding_box_utils import bbox_nms
                from omegaconf import OmegaConf

                nms_cfg_obj = OmegaConf.create(self.nms_cfg)
                batch_results = bbox_nms(pred_cls, pred_bbox, nms_cfg_obj)

                outputs['batch_results'] = batch_results

        return outputs


class MitYOLO_V9C(MitYOLO_Detector):
    """MIT-YOLO v9-c detector (standard variant)."""

    def __init__(self, classes, channels='rgb', input_stats=None,
                 weight_path=None, nms_cfg=None, loss_cfg=None):
        super().__init__(
            classes=classes,
            channels=channels,
            input_stats=input_stats,
            model_variant='v9-c',
            weight_path=weight_path,
            nms_cfg=nms_cfg,
            loss_cfg=loss_cfg,
        )


class MitYOLO_V9S(MitYOLO_Detector):
    """MIT-YOLO v9-s detector (small variant)."""

    def __init__(self, classes, channels='rgb', input_stats=None,
                 weight_path=None, nms_cfg=None, loss_cfg=None):
        super().__init__(
            classes=classes,
            channels=channels,
            input_stats=input_stats,
            model_variant='v9-s',
            weight_path=weight_path,
            nms_cfg=nms_cfg,
            loss_cfg=loss_cfg,
        )


class MitYOLO_V9M(MitYOLO_Detector):
    """MIT-YOLO v9-m detector (medium variant)."""

    def __init__(self, classes, channels='rgb', input_stats=None,
                 weight_path=None, nms_cfg=None, loss_cfg=None):
        super().__init__(
            classes=classes,
            channels=channels,
            input_stats=input_stats,
            model_variant='v9-m',
            weight_path=weight_path,
            nms_cfg=nms_cfg,
            loss_cfg=loss_cfg,
        )
