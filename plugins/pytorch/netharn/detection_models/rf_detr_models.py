# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
RF-DETR Detection Models for Netharn

This module provides netharn-compatible wrappers for RF-DETR models,
following the same pattern as mm_models.py for MMDetection.

Supports Base, Large, Small, Medium, and Nano RF-DETR variants.

Example:
    >>> # xdoctest: +SKIP
    >>> from viame.pytorch.netharn.detection_models import rfdetr_models
    >>> model = rfdetr_models.RFDETR_Base(['person', 'car', 'bike'], weight_path=False)
    >>> batch = model.demo_batch(bsize=2, h=560, w=560)
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


class RFDETR_Coder:
    """
    Standardize RF-DETR network outputs to kwimage.Detections format.

    Converts the PostProcess outputs (list of dicts with scores, labels, boxes)
    into kwimage.Detections objects for consistent downstream processing.

    Example:
        >>> # xdoctest: +SKIP
        >>> classes = ['person', 'car', 'bike']
        >>> coder = RFDETR_Coder(classes)
        >>> mock_outputs = {
        ...     'batch_results': [
        ...         {'scores': torch.tensor([0.9, 0.8]),
        ...          'labels': torch.tensor([0, 1]),
        ...          'boxes': torch.tensor([[10, 20, 50, 60], [100, 100, 200, 200]])}
        ...     ]
        ... }
        >>> dets = coder.decode_batch(mock_outputs)
        >>> assert len(dets) == 1
        >>> assert len(dets[0]) == 2
    """

    def __init__(self, classes, score_thresh=0.0, keypoint_names=None):
        """
        Args:
            classes: List of class names or kwcoco.CategoryTree
            score_thresh: Minimum score threshold for detections
        """
        import kwcoco
        self.classes = kwcoco.CategoryTree.coerce(classes)
        self.score_thresh = score_thresh
        self.keypoint_names = keypoint_names

    def decode_batch(self, outputs):
        """
        Transform RF-DETR PostProcess outputs into a list of kwimage.Detections objects.

        Args:
            outputs (Dict): dict containing 'batch_results' which is a list of
                dicts with keys 'scores', 'labels', 'boxes' (xyxy absolute format)

        Returns:
            List[kwimage.Detections]: One detection object per batch item
        """
        batch_results = outputs['batch_results']
        # Unwrap BatchContainer if present (single-GPU path)
        if isinstance(batch_results, data_containers.BatchContainer):
            batch_results = batch_results.data
        batch_dets = []

        for result in batch_results:
            if result is None or len(result.get('scores', [])) == 0:
                det = kwimage.Detections(
                    boxes=kwimage.Boxes(np.empty((0, 4)), 'ltrb'),
                    scores=np.array([]),
                    class_idxs=np.array([], dtype=int),
                    classes=self.classes
                )
            else:
                scores = result['scores']
                labels = result['labels']
                boxes = result['boxes']

                # Convert to numpy
                if isinstance(scores, torch.Tensor):
                    scores = scores.detach().cpu().numpy()
                if isinstance(labels, torch.Tensor):
                    labels = labels.detach().cpu().numpy().astype(int)
                if isinstance(boxes, torch.Tensor):
                    boxes = boxes.detach().cpu().numpy()

                # Apply score threshold
                keep = scores >= self.score_thresh
                scores = scores[keep]
                labels = labels[keep]
                boxes = boxes[keep]

                # Clamp class indices to valid range to avoid IndexError during visualization
                num_classes = len(self.classes)
                if len(labels) > 0 and num_classes > 0:
                    labels = np.clip(labels, 0, num_classes - 1)

                det = kwimage.Detections(
                    boxes=kwimage.Boxes(boxes, 'ltrb'),  # xyxy = ltrb format
                    scores=scores,
                    class_idxs=labels,
                    classes=self.classes
                )
                if 'masks' in result:
                    masks = result['masks'].detach().cpu().numpy()[keep]
                    det.data['segmentations'] = kwimage.SegmentationList([
                        kwimage.Segmentation.coerce(
                            kwimage.Mask(mask.reshape(mask.shape[-2:]).astype(np.uint8), 'c_mask'))
                        for mask in masks])
                if 'keypoints' in result:
                    points = result['keypoints'].detach().cpu().numpy()[keep]
                    det.data['keypoints'] = kwimage.PointsList([
                        kwimage.Points(xy=p[:, :2], visible=p[:, 2],
                                       class_idxs=np.arange(len(p)),
                                       classes=self.keypoint_names)
                        for p in points])
            batch_dets.append(det)

        return batch_dets


def _batch_to_rfdetr_targets(batch, image_size, device=None):
    """Convert scattered or collated labels, keeping all annotation rows aligned."""
    label = batch.get('label', {})
    box_key = 'tlbr' if 'tlbr' in label else 'cxywh'
    if box_key not in label:
        return []
    h, w = image_size

    def rows(container):
        if isinstance(container, data_containers.BatchContainer):
            container = container.data
        result = []
        for part in container:
            result.extend(part if isinstance(part, (list, tuple)) else [part])
        return result

    fields = {key: rows(label[key]) for key in
              (box_key, 'class_idxs', 'weight', 'class_masks', 'has_mask', 'keypoints')
              if key in label}
    targets = []
    for i, boxes in enumerate(fields[box_key]):
        boxes = torch.as_tensor(boxes, dtype=torch.float32, device=device).reshape(-1, 4)
        labels = torch.as_tensor(fields['class_idxs'][i], dtype=torch.long, device=device)
        keep = torch.ones(len(boxes), dtype=torch.bool, device=device)
        if 'weight' in fields:
            keep &= torch.as_tensor(fields['weight'][i], device=device) >= 0.1
        if box_key == 'tlbr':
            boxes = torch.cat(((boxes[:, :2] + boxes[:, 2:]) / 2,
                               boxes[:, 2:] - boxes[:, :2]), dim=1)
        boxes = (boxes / boxes.new_tensor([w, h, w, h])).clamp(0, 1)
        target = {'labels': labels[keep], 'boxes': boxes[keep]}
        if 'class_masks' in fields:
            if 'has_mask' in fields:
                valid = torch.as_tensor(fields['has_mask'][i], device=device)
                if (valid[keep] <= 0).any():
                    raise ValueError('RF-DETR segmentation requires a mask for every non-ignored object')
            target['masks'] = torch.as_tensor(
                fields['class_masks'][i], device=device, dtype=torch.bool)[keep]
        if 'keypoints' in fields:
            points = torch.as_tensor(fields['keypoints'][i], device=device,
                                     dtype=torch.float32)[keep].clone()
            points[..., :2] /= points.new_tensor([w, h])
            target['keypoints'] = points
        targets.append(target)
    return targets


def _batch_to_rfdetr_inputs(batch, device=None):
    """
    Convert netharn-style batch to RF-DETR input format.

    Args:
        batch: Netharn batch which can be:
            - A raw tensor (B, C, H, W)
            - A dict with 'inputs' key containing BatchContainer

    Returns:
        Dict containing:
            - 'images': Tensor (B, C, H, W)
            - 'targets': List of target dicts (one per image)
            - 'image_size': Tuple (H, W)
    """
    # Handle raw tensor input
    if isinstance(batch, torch.Tensor):
        B, C, H, W = batch.shape
        return {
            'images': batch,
            'targets': [],
            'image_size': (H, W),
        }

    # Handle BatchContainer input without labels
    if isinstance(batch, data_containers.BatchContainer):
        if batch.stack:
            images = batch.data[0] if len(batch.data) == 1 else torch.cat(batch.data, dim=0)
        else:
            images = torch.stack([d for d in ub.flatten(batch.data)])
        B, C, H, W = images.shape
        return {
            'images': images,
            'targets': [],
            'image_size': (H, W),
        }

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

    # Convert labels to RF-DETR target format
    targets = _batch_to_rfdetr_targets(batch, (H, W), device=images.device)

    return {
        'images': images,
        'targets': targets,
        'image_size': (H, W),
    }


def _demo_batch_rfdetr(bsize=1, channels='rgb', h=560, w=560, classes=3, with_mask=False):
    """
    Generate a demo batch for testing RF-DETR detectors.

    Args:
        bsize: Batch size
        channels: Channel specification (e.g., 'rgb')
        h: Image height
        w: Image width
        classes: Number of classes or list of class names
        with_mask: Whether to include masks (not used for DETR but kept for API compatibility)

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
        key: torch.rand(*shape)
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


class RFDETR_Detector(nh.layers.Module):
    """
    Netharn wrapper for RF-DETR detection models.

    Supports Base, Large, Small, Medium, and Nano RF-DETR variants.
    Provides a unified interface compatible with netharn's training harness.

    Example:
        >>> # xdoctest: +SKIP
        >>> model = RFDETR_Detector(['person', 'car'], model_variant='base')
        >>> batch = model.demo_batch(bsize=2, h=560, w=560)
        >>> outputs = model.forward(batch, return_loss=True, return_result=True)
        >>> assert 'loss_parts' in outputs
        >>> assert 'batch_results' in outputs

    Attributes:
        model: The underlying RF-DETR model (LWDETR)
        criterion: SetCriterion for loss computation
        postprocess: PostProcess for output decoding
        coder: Output decoder to kwimage.Detections
        input_norm: Input normalization layer
    """

    __BUILTIN_CRITERION__ = True
    # SetCriterion already normalizes each replica's loss. Average across GPUs
    # and accumulated batches so neither multiplies the gradient scale.
    __LOSS_REDUCTION__ = 'mean'
    __DEPLOY_SUPPORTED__ = False  # RF-DETR doesn't support torch_liberator deployment

    # ImageNet normalization stats used by RF-DETR
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(self, classes, channels='rgb', input_stats=None,
                 model_variant='base', weight_path=None, score_thresh=0.0,
                 num_queries=None, resolution=None, segmentation_head=False,
                 keypoint_names=None):
        """
        Args:
            classes: List of class names or kwcoco.CategoryTree
            channels: Input channel specification (default 'rgb')
            input_stats: Dict with 'mean' and 'std' for input normalization
                         If None, uses ImageNet stats
            model_variant: RF-DETR variant name (base, large, small, medium, nano)
            weight_path: Path to pretrained weights, True to auto-download, False/None for none
            score_thresh: Score threshold for detections (default 0.0)
            num_queries: Number of detection queries (None uses variant default)
            resolution: Input resolution (uses variant default if None)
            segmentation_head: Use the RFDETRSeg architecture and mask losses
            keypoint_names: Ordered names enabling the optional keypoint head
        """
        super().__init__()
        import kwcoco

        # Store segmentation head setting
        self.segmentation_head = segmentation_head
        if isinstance(keypoint_names, str):
            keypoint_names = keypoint_names.split(',')
        self.keypoint_names = [n.strip().lower() for n in (keypoint_names or [])]
        if keypoint_names is not None and (not self.keypoint_names or
                not all(self.keypoint_names) or
                len(set(self.keypoint_names)) != len(self.keypoint_names)):
            raise ValueError('keypoint_names must contain unique, nonempty names')

        # Store initialization kwargs for serialization
        self._initkw = {
            'classes': classes,
            'channels': channels,
            'input_stats': input_stats,
            'model_variant': model_variant,
            'weight_path': weight_path,
            'score_thresh': score_thresh,
            'num_queries': num_queries,
            'resolution': resolution,
            'segmentation_head': segmentation_head,
            'keypoint_names': self.keypoint_names or None,
        }

        # Setup classes
        self.classes = kwcoco.CategoryTree.coerce(classes)
        self.num_classes = len(self.classes)

        # Setup channels
        self.channels = ChannelSpec.coerce(channels)
        chann_norm = self.channels.normalize()
        assert len(chann_norm) == 1, "RFDETR only supports single input stream"
        self.in_channels = len(ub.peek(chann_norm.values()))
        assert self.in_channels == 3, "RFDETR requires 3-channel (RGB) input"

        # Setup input normalization (RF-DETR uses ImageNet stats)
        if input_stats is None:
            input_stats = {
                'mean': self.IMAGENET_MEAN,
                'std': self.IMAGENET_STD,
            }
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

        # Model variant configuration
        self.model_variant = model_variant.lower()
        self.score_thresh = score_thresh

        # Get variant-specific config
        config = self._get_variant_config(self.model_variant, num_queries, resolution)
        self.resolution = config.resolution
        self.num_queries = config.num_queries

        # Build model
        self.model, self.criterion, self.postprocess = self._build_model(
            config, weight_path
        )

        # Output decoder
        self.coder = RFDETR_Coder(self.classes, score_thresh=score_thresh,
                                  keypoint_names=self.keypoint_names)

    def _get_variant_config(self, variant, num_queries, resolution):
        """Build the pydantic ModelConfig for a model variant."""
        from rfdetr import config as rfdetr_config

        variant_to_config_cls = {
            'base': rfdetr_config.RFDETRBaseConfig,
            'large': rfdetr_config.RFDETRLargeConfig,
            'small': rfdetr_config.RFDETRSmallConfig,
            'medium': rfdetr_config.RFDETRMediumConfig,
            'nano': rfdetr_config.RFDETRNanoConfig,
        }

        if self.segmentation_head:
            variant_to_config_cls = {
                'nano': rfdetr_config.RFDETRSegNanoConfig,
                'small': rfdetr_config.RFDETRSegSmallConfig,
                'medium': rfdetr_config.RFDETRSegMediumConfig,
                'large': rfdetr_config.RFDETRSegLargeConfig,
                'xlarge': rfdetr_config.RFDETRSegXLargeConfig,
                '2xlarge': rfdetr_config.RFDETRSeg2XLargeConfig,
            }

        if variant not in variant_to_config_cls:
            raise ValueError(f"Unknown variant: {variant}. "
                             f"Available: {list(variant_to_config_cls.keys())}")

        overrides = {
            'num_classes': self.num_classes,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'segmentation_head': self.segmentation_head,
            'keypoint_head': bool(self.keypoint_names),
            'num_keypoints': len(self.keypoint_names) or 2,
            # upstream trains with group_detr=13, which changes query shapes
            'group_detr': 1,
        }
        if num_queries is not None:
            overrides['num_queries'] = num_queries
            overrides['num_select'] = num_queries
        if resolution is not None:
            overrides['resolution'] = resolution

        return variant_to_config_cls[variant](**overrides)

    def _build_model(self, model_config, weight_path):
        """Build the RF-DETR model, criterion, and postprocessor."""
        import os
        from rfdetr.assets.model_weights import get_model_cache_dir
        from rfdetr.config import TrainConfig, SegmentationTrainConfig
        from rfdetr.models import (build_criterion_from_config,
                                   build_model_from_config,
                                   load_pretrain_weights)

        if weight_path is True:
            pretrain = model_config.pretrain_weights
            if pretrain and not os.path.dirname(pretrain):
                cache_dir = get_model_cache_dir()
                os.makedirs(cache_dir, exist_ok=True)
                model_config.pretrain_weights = os.path.join(cache_dir, pretrain)
        elif weight_path is False or weight_path is None:
            model_config.pretrain_weights = None
        else:
            model_config.pretrain_weights = weight_path

        config_cls = SegmentationTrainConfig if self.segmentation_head else TrainConfig
        train_config = config_cls(dataset_dir='.', output_dir='.')
        model = build_model_from_config(model_config, train_config)

        if model_config.pretrain_weights is not None:
            seed_classes = load_pretrain_weights(model, model_config)
            if seed_classes and list(seed_classes) != list(self.classes):
                import warnings
                warnings.warn('RF-DETR seed class names/order differ from the training '
                              'classes. The checkpoint loader aligns head sizes, '
                              'but does not remap class names: seed=%r, training=%r' %
                              (seed_classes, list(self.classes)))

        criterion, postprocess = build_criterion_from_config(
            model_config, train_config)

        return model, criterion, postprocess

    def demo_batch(self, bsize=3, h=None, w=None):
        """
        Generate a demo batch for testing.

        Args:
            bsize: Batch size
            h: Image height (uses model resolution if None)
            w: Image width (uses model resolution if None)

        Returns:
            Dict: Netharn-style batch with inputs and labels
        """
        default_hw = self.resolution
        if not isinstance(default_hw, (tuple, list)):
            default_hw = (default_hw, default_hw)
        if h is None:
            h = default_hw[0]
        if w is None:
            w = default_hw[1]

        return _demo_batch_rfdetr(
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
        rfdetr_inputs = _batch_to_rfdetr_inputs(batch)
        images = rfdetr_inputs['images']
        targets = rfdetr_inputs['targets']
        image_size = rfdetr_inputs['image_size']

        device = images.device
        B, C, H, W = images.shape

        if return_loss:
            for target in targets:
                if getattr(self, 'segmentation_head', False) and 'masks' not in target:
                    if len(target['labels']):
                        raise ValueError('RF-DETR segmentation requires mask annotations')
                    target['masks'] = torch.zeros((0, H, W), dtype=torch.bool, device=device)
                if getattr(self, 'keypoint_names', None) and 'keypoints' not in target:
                    raise ValueError('RF-DETR keypoint training requires keypoint targets')

        # Move model components to device if needed
        first_param = next(self.model.parameters(), None)
        if first_param is not None and first_param.device != device:
            self.model = self.model.to(device)
            self.criterion = self.criterion.to(device)
            self.input_norm = self.input_norm.to(device)

        # Ensure input_norm is on same device as images
        if hasattr(self.input_norm, 'mean') and self.input_norm.mean.device != device:
            self.input_norm = self.input_norm.to(device)

        # Apply input normalization
        images_norm = self.input_norm(images)

        # Forward through model
        # RF-DETR model can accept raw tensor (auto-converts to NestedTensor)
        raw_outputs = self.model(images_norm, targets=targets if return_loss and len(targets) > 0 else None)

        outputs = {}

        # Compute loss if requested and labels are available
        if return_loss and len(targets) > 0:
            # Compute losses using SetCriterion
            loss_dict = self.criterion(raw_outputs, targets)

            # Apply weight dict to losses
            weight_dict = self.criterion.weight_dict
            loss_parts = OrderedDict()

            for k, v in loss_dict.items():
                if k in weight_dict:
                    weighted_loss = v * weight_dict[k]
                    loss_parts[k] = weighted_loss.mean().unsqueeze(0)

            # FitHarn sums these components for backward(). Including their
            # total here would count every loss (and its gradient) twice.
            outputs['loss_parts'] = loss_parts

        if return_result:
            with torch.no_grad():
                # Get target sizes for postprocessing
                target_sizes = torch.tensor([[H, W]] * B, device=device)

                # Apply postprocessor
                results = self.postprocess(raw_outputs, target_sizes)

                outputs['batch_results'] = data_containers.BatchContainer(
                    results, stack=False)

        return outputs


class RFDETR_Base(RFDETR_Detector):
    """RF-DETR Base detector."""

    def __init__(self, classes, channels='rgb', input_stats=None,
                 weight_path=None, score_thresh=0.0, num_queries=300, resolution=None):
        super().__init__(
            classes=classes,
            channels=channels,
            input_stats=input_stats,
            model_variant='base',
            weight_path=weight_path,
            score_thresh=score_thresh,
            num_queries=num_queries,
            resolution=resolution,
        )


class RFDETR_Large(RFDETR_Detector):
    """RF-DETR Large detector."""

    def __init__(self, classes, channels='rgb', input_stats=None,
                 weight_path=None, score_thresh=0.0, num_queries=300, resolution=None):
        super().__init__(
            classes=classes,
            channels=channels,
            input_stats=input_stats,
            model_variant='large',
            weight_path=weight_path,
            score_thresh=score_thresh,
            num_queries=num_queries,
            resolution=resolution,
        )


class RFDETR_Small(RFDETR_Detector):
    """RF-DETR Small detector."""

    def __init__(self, classes, channels='rgb', input_stats=None,
                 weight_path=None, score_thresh=0.0, num_queries=300, resolution=None):
        super().__init__(
            classes=classes,
            channels=channels,
            input_stats=input_stats,
            model_variant='small',
            weight_path=weight_path,
            score_thresh=score_thresh,
            num_queries=num_queries,
            resolution=resolution,
        )


class RFDETR_Medium(RFDETR_Detector):
    """RF-DETR Medium detector."""

    def __init__(self, classes, channels='rgb', input_stats=None,
                 weight_path=None, score_thresh=0.0, num_queries=300, resolution=None):
        super().__init__(
            classes=classes,
            channels=channels,
            input_stats=input_stats,
            model_variant='medium',
            weight_path=weight_path,
            score_thresh=score_thresh,
            num_queries=num_queries,
            resolution=resolution,
        )


class RFDETR_Nano(RFDETR_Detector):
    """RF-DETR Nano detector (smallest and fastest)."""

    def __init__(self, classes, channels='rgb', input_stats=None,
                 weight_path=None, score_thresh=0.0, num_queries=300, resolution=None):
        super().__init__(
            classes=classes,
            channels=channels,
            input_stats=input_stats,
            model_variant='nano',
            weight_path=weight_path,
            score_thresh=score_thresh,
            num_queries=num_queries,
            resolution=resolution,
        )
