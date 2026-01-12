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
        Transform RF-DETR PostProcess outputs into a list of kwimage.Detections objects.

        Args:
            outputs (Dict): dict containing 'batch_results' which is a list of
                dicts with keys 'scores', 'labels', 'boxes' (xyxy absolute format)

        Returns:
            List[kwimage.Detections]: One detection object per batch item
        """
        batch_results = outputs['batch_results']
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

                det = kwimage.Detections(
                    boxes=kwimage.Boxes(boxes, 'ltrb'),  # xyxy = ltrb format
                    scores=scores,
                    class_idxs=labels,
                    classes=self.classes
                )
            batch_dets.append(det)

        return batch_dets


def _batch_to_rfdetr_targets(batch, image_size, device=None):
    """
    Convert netharn-style batch labels to RF-DETR target format.

    RF-DETR expects targets as a list of dicts, one per image:
        - 'labels': Tensor of shape [num_targets] with class indices
        - 'boxes': Tensor of shape [num_targets, 4] in normalized cxcywh format

    Args:
        batch: Netharn batch dict with 'label' key containing:
            - 'tlbr' or 'cxywh': BatchContainer of boxes
            - 'class_idxs': BatchContainer of class indices
            - 'weight': Optional BatchContainer of weights
        image_size: Tuple (H, W) for normalizing boxes
        device: Device to place tensors on

    Returns:
        List of dicts, one per image
    """
    if 'label' not in batch:
        return []

    label = batch['label']
    H, W = image_size

    # Get boxes container - prefer tlbr, fall back to cxywh
    box_format = None
    if 'tlbr' in label:
        boxes_container = label['tlbr']
        box_format = 'tlbr'
    elif 'cxywh' in label:
        boxes_container = label['cxywh']
        box_format = 'cxywh'
    else:
        return []

    class_container = label['class_idxs']
    weight_container = label.get('weight', None)

    # Unwrap containers
    all_boxes_lists = []
    all_cidxs_lists = []
    all_weights_lists = []

    for device_idx, (device_data_boxes, device_data_cidxs) in enumerate(
            zip(boxes_container.data, class_container.data)):
        if isinstance(device_data_boxes, (list, tuple)):
            all_boxes_lists.extend(device_data_boxes)
            all_cidxs_lists.extend(device_data_cidxs)
            if weight_container is not None:
                if device_idx < len(weight_container.data):
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

    # Build target list
    targets = []
    for bx in range(len(all_boxes_lists)):
        boxes = all_boxes_lists[bx]
        cidxs = all_cidxs_lists[bx]

        if boxes is None or len(boxes) == 0:
            targets.append({
                'labels': torch.zeros(0, dtype=torch.long, device=device),
                'boxes': torch.zeros(0, 4, dtype=torch.float32, device=device),
            })
            continue

        # Convert to tensor if needed
        if not isinstance(boxes, torch.Tensor):
            boxes = torch.tensor(boxes, dtype=torch.float32, device=device)
        else:
            boxes = boxes.to(device=device, dtype=torch.float32)

        if not isinstance(cidxs, torch.Tensor):
            cidxs = torch.tensor(cidxs, dtype=torch.long, device=device)
        else:
            cidxs = cidxs.to(device=device, dtype=torch.long)

        # Apply weight filtering if available
        if weight_container is not None and bx < len(all_weights_lists):
            weights = all_weights_lists[bx]
            if weights is not None:
                if not isinstance(weights, torch.Tensor):
                    weights = torch.tensor(weights, device=device)
                else:
                    weights = weights.to(device=device)
                valid_mask = weights >= 0.1
                boxes = boxes[valid_mask]
                cidxs = cidxs[valid_mask]

        if len(boxes) == 0:
            targets.append({
                'labels': torch.zeros(0, dtype=torch.long, device=device),
                'boxes': torch.zeros(0, 4, dtype=torch.float32, device=device),
            })
            continue

        # Convert boxes to normalized cxcywh format
        if box_format == 'tlbr':
            # tlbr (x1, y1, x2, y2) -> cxcywh
            x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            cx = (x1 + x2) / 2.0 / W
            cy = (y1 + y2) / 2.0 / H
            w = (x2 - x1) / W
            h = (y2 - y1) / H
            boxes = torch.stack([cx, cy, w, h], dim=1)
        elif box_format == 'cxywh':
            # cxywh -> normalized cxcywh
            cx = boxes[:, 0] / W
            cy = boxes[:, 1] / H
            w = boxes[:, 2] / W
            h = boxes[:, 3] / H
            boxes = torch.stack([cx, cy, w, h], dim=1)

        # Clamp to valid range
        boxes = boxes.clamp(0, 1)

        targets.append({
            'labels': cidxs,
            'boxes': boxes,
        })

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

    # ImageNet normalization stats used by RF-DETR
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(self, classes, channels='rgb', input_stats=None,
                 model_variant='base', weight_path=None, score_thresh=0.0,
                 num_queries=300, resolution=None):
        """
        Args:
            classes: List of class names or kwcoco.CategoryTree
            channels: Input channel specification (default 'rgb')
            input_stats: Dict with 'mean' and 'std' for input normalization
                         If None, uses ImageNet stats
            model_variant: RF-DETR variant name (base, large, small, medium, nano)
            weight_path: Path to pretrained weights, True to auto-download, False/None for none
            score_thresh: Score threshold for detections (default 0.0)
            num_queries: Number of detection queries (default 300)
            resolution: Input resolution (uses variant default if None)
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
            'score_thresh': score_thresh,
            'num_queries': num_queries,
            'resolution': resolution,
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
        self.resolution = config['resolution']
        self.num_queries = config['num_queries']

        # Build model
        self.model, self.criterion, self.postprocess = self._build_model(
            config, weight_path
        )

        # Output decoder
        self.coder = RFDETR_Coder(self.classes, score_thresh=score_thresh)

    def _get_variant_config(self, variant, num_queries, resolution):
        """Get configuration for a model variant."""
        variant_configs = {
            'base': {
                'encoder': 'dinov2_windowed_small',
                'hidden_dim': 256,
                'patch_size': 14,
                'num_windows': 4,
                'dec_layers': 3,
                'sa_nheads': 8,
                'ca_nheads': 16,
                'dec_n_points': 2,
                'num_queries': 300,
                'num_select': 300,
                'projector_scale': ['P4'],
                'out_feature_indexes': [2, 5, 8, 11],
                'resolution': 560,
                'positional_encoding_size': 37,
                'pretrain_weights': 'rf-detr-base.pth',
            },
            'large': {
                'encoder': 'dinov2_windowed_base',
                'hidden_dim': 384,
                'patch_size': 14,
                'num_windows': 4,
                'dec_layers': 3,
                'sa_nheads': 12,
                'ca_nheads': 24,
                'dec_n_points': 4,
                'num_queries': 300,
                'num_select': 300,
                'projector_scale': ['P3', 'P5'],
                'out_feature_indexes': [2, 5, 8, 11],
                'resolution': 560,
                'positional_encoding_size': 37,
                'pretrain_weights': 'rf-detr-large.pth',
            },
            'small': {
                'encoder': 'dinov2_windowed_small',
                'hidden_dim': 256,
                'patch_size': 16,
                'num_windows': 2,
                'dec_layers': 3,
                'sa_nheads': 8,
                'ca_nheads': 16,
                'dec_n_points': 2,
                'num_queries': 300,
                'num_select': 300,
                'projector_scale': ['P4'],
                'out_feature_indexes': [3, 6, 9, 12],
                'resolution': 512,
                'positional_encoding_size': 32,
                'pretrain_weights': 'rf-detr-small.pth',
            },
            'medium': {
                'encoder': 'dinov2_windowed_small',
                'hidden_dim': 256,
                'patch_size': 16,
                'num_windows': 2,
                'dec_layers': 4,
                'sa_nheads': 8,
                'ca_nheads': 16,
                'dec_n_points': 2,
                'num_queries': 300,
                'num_select': 300,
                'projector_scale': ['P4'],
                'out_feature_indexes': [3, 6, 9, 12],
                'resolution': 576,
                'positional_encoding_size': 36,
                'pretrain_weights': 'rf-detr-medium.pth',
            },
            'nano': {
                'encoder': 'dinov2_windowed_small',
                'hidden_dim': 256,
                'patch_size': 16,
                'num_windows': 2,
                'dec_layers': 2,
                'sa_nheads': 8,
                'ca_nheads': 16,
                'dec_n_points': 2,
                'num_queries': 300,
                'num_select': 300,
                'projector_scale': ['P4'],
                'out_feature_indexes': [3, 6, 9, 12],
                'resolution': 384,
                'positional_encoding_size': 24,
                'pretrain_weights': 'rf-detr-nano.pth',
            },
        }

        if variant not in variant_configs:
            raise ValueError(f"Unknown variant: {variant}. "
                           f"Available: {list(variant_configs.keys())}")

        config = variant_configs[variant].copy()

        # Override with user-specified values
        if num_queries is not None:
            config['num_queries'] = num_queries
            config['num_select'] = num_queries
        if resolution is not None:
            config['resolution'] = resolution

        return config

    def _build_model(self, config, weight_path):
        """Build the RF-DETR model, criterion, and postprocessor."""
        from rfdetr.main import populate_args
        from rfdetr.models import build_model, build_criterion_and_postprocessors

        # Determine weight path
        if weight_path is True:
            actual_weight_path = config.get('pretrain_weights')
        elif weight_path is False or weight_path is None:
            actual_weight_path = None
        else:
            actual_weight_path = weight_path

        # Build args namespace
        args = populate_args(
            num_classes=self.num_classes,
            encoder=config['encoder'],
            hidden_dim=config['hidden_dim'],
            patch_size=config['patch_size'],
            num_windows=config['num_windows'],
            dec_layers=config['dec_layers'],
            sa_nheads=config['sa_nheads'],
            ca_nheads=config['ca_nheads'],
            dec_n_points=config['dec_n_points'],
            num_queries=config['num_queries'],
            num_select=config['num_select'],
            projector_scale=config['projector_scale'],
            out_feature_indexes=config['out_feature_indexes'],
            resolution=config['resolution'],
            positional_encoding_size=config['positional_encoding_size'],
            pretrain_weights=actual_weight_path,
            # Loss coefficients
            cls_loss_coef=1.0,
            bbox_loss_coef=5.0,
            giou_loss_coef=2.0,
            focal_alpha=0.25,
            # Standard settings
            two_stage=True,
            bbox_reparam=True,
            lite_refpoint_refine=True,
            aux_loss=True,
            group_detr=1,  # Use 1 for inference, higher for training
            ia_bce_loss=True,
            device='cuda' if torch.cuda.is_available() else 'cpu',
        )

        # Build model
        model = build_model(args)

        # Load pretrained weights if specified
        if actual_weight_path is not None:
            from rfdetr.main import download_pretrain_weights
            download_pretrain_weights(actual_weight_path)
            if actual_weight_path and torch.cuda.is_available():
                checkpoint = torch.load(actual_weight_path, map_location='cpu', weights_only=False)
            elif actual_weight_path:
                checkpoint = torch.load(actual_weight_path, map_location='cpu', weights_only=False)

            if 'model' in checkpoint:
                # Handle class mismatch
                checkpoint_num_classes = checkpoint['model']['class_embed.bias'].shape[0]
                if checkpoint_num_classes != self.num_classes + 1:
                    model.reinitialize_detection_head(self.num_classes + 1)
                model.load_state_dict(checkpoint['model'], strict=False)

        # Build criterion and postprocessor
        criterion, postprocess = build_criterion_and_postprocessors(args)

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
        if h is None:
            h = self.resolution
        if w is None:
            w = self.resolution

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

        # Move model components to device if needed
        if next(self.model.parameters()).device != device:
            self.model = self.model.to(device)
            self.criterion = self.criterion.to(device)

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

            # Compute total loss
            total_loss = sum(loss_parts.values())
            loss_parts['loss_total'] = total_loss.unsqueeze(0) if total_loss.dim() == 0 else total_loss

            outputs['loss_parts'] = loss_parts

        if return_result:
            with torch.no_grad():
                # Get target sizes for postprocessing
                target_sizes = torch.tensor([[H, W]] * B, device=device)

                # Apply postprocessor
                results = self.postprocess(raw_outputs, target_sizes)

                outputs['batch_results'] = results

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
