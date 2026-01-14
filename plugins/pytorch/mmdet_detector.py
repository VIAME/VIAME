# This file is part of VIAME, and is distributed under the
# OSI-approved BSD 3-Clause License. See top-level LICENSE.txt file or
# https://github.com/VIAME/VIAME/blob/master/LICENSE.txt for details.

from collections import namedtuple

import logging
import sys
import cv2
import mmcv
import numpy as np

logger = logging.getLogger(__name__)

from distutils.util import strtobool

from kwiver.vital.algo import ImageObjectDetector

from kwiver.vital.types import (
    BoundingBoxD, DetectedObject, DetectedObjectSet, DetectedObjectType
)

try:
    import viame.pytorch.learn.mmdet.register_modules
    import torch
    import torch.nn.functional as F
    from mmdet.models.builder import LOSSES
    from functools import partial
except ModuleNotFoundError:
    pass

_Option = namedtuple('_Option', ['attr', 'config', 'default', 'parse'])


class MMDetDetector(ImageObjectDetector):
    """
    Implementation of ImageObjectDetector class
    """

    # Config-option-based attribute specifications, used in __init__,
    # get_configuration, and set_configuration
    _options = [
        _Option('_net_config', 'net_config', '', str),
        _Option('_weight_file', 'weight_file', '', str),
        _Option('_class_names', 'class_names', '', str),
        _Option('_thresh', 'thresh', 0.01, float),
        _Option('_gpu_index', 'gpu_index', "0", str),
        _Option('_num_classes', 'num_classes', 1, int),
        _Option('_use_learn', 'use_learn', False, strtobool),
        _Option('_display_detections', 'display_detections', False, strtobool),
        _Option('_template', 'template', "", str),
        _Option('_auto_update_model', 'auto_update_model', True, strtobool),
        _Option('_rgb_to_bgr', 'rgb_to_bgr', True, strtobool),
    ]

    def __init__(self):
        ImageObjectDetector.__init__(self)
        for opt in self._options:
            setattr(self, opt.attr, opt.default)

    def get_configuration(self):
        # Inherit from the base class
        cfg = super(ImageObjectDetector, self).get_configuration()
        for opt in self._options:
            cfg.set_value(opt.config, str(getattr(self, opt.attr)))
        return cfg
    def register_new_losses(self):
        LOSSES._module_dict.pop('EQLv2', None)

        @LOSSES.register_module()
        class EQLv2(torch.nn.Module):
            def __init__(self,
                    use_sigmoid=True,
                    reduction='mean',
                    class_weight=None,
                    loss_weight=1.0,
                    num_classes=3,  # 1203 for lvis v1.0, 1230 for lvis v0.5
                    gamma=12,
                    mu=0.8,
                    alpha=4.0,
                    vis_grad=False):
                super().__init__()
                self.use_sigmoid = True
                self.reduction = reduction
                self.loss_weight = loss_weight
                self.class_weight = class_weight
                self.num_classes = num_classes
                self.group = True

                # cfg for eqlv2
                self.vis_grad = vis_grad
                self.gamma = gamma
                self.mu = mu
                self.alpha = alpha

                # initial variables
                self._pos_grad = None
                self._neg_grad = None
                self.pos_neg = None

                def _func(x, gamma, mu):
                    return 1 / (1 + torch.exp(-gamma * (x - mu)))
                self.map_func = partial(_func, gamma=self.gamma, mu=self.mu)
                # logger = get_root_logger()
                # logger.info(f"build EQL v2, gamma: {gamma}, mu: {mu}, alpha: {alpha}")

            def forward(self,
                        cls_score,
                        label,
                        weight=None,
                        avg_factor=None,
                        reduction_override=None,
                        **kwargs):
                self.n_i, self.n_c = cls_score.size()

                self.gt_classes = label
                self.pred_class_logits = cls_score

                def expand_label(pred, gt_classes):
                    target = pred.new_zeros(self.n_i, self.n_c)
                    target[torch.arange(self.n_i), gt_classes] = 1
                    return target

                target = expand_label(cls_score, label)
                pos_w, neg_w = self.get_weight(cls_score)
                weight = pos_w * target + neg_w * (1 - target)
                cls_loss = F.binary_cross_entropy_with_logits(cls_score, target,
                                                            reduction='none')
                cls_loss = torch.sum(cls_loss * weight) / self.n_i
                self.collect_grad(cls_score.detach(), target.detach(), weight.detach())
                return self.loss_weight * cls_loss

            def get_channel_num(self, num_classes):
                num_channel = num_classes + 1
                return num_channel

            def get_activation(self, cls_score):
                cls_score = torch.sigmoid(cls_score)
                n_i, n_c = cls_score.size()
                bg_score = cls_score[:, -1].view(n_i, 1)
                cls_score[:, :-1] *= (1 - bg_score)
                return cls_score

            def collect_grad(self, cls_score, target, weight):
                prob = torch.sigmoid(cls_score)
                grad = target * (prob - 1) + (1 - target) * prob
                grad = torch.abs(grad)

                # do not collect grad for objectiveness branch [:-1]
                pos_grad = torch.sum(grad * target * weight, dim=0)[:-1]
                neg_grad = torch.sum(grad * (1 - target) * weight, dim=0)[:-1]

                # dist.all_reduce(pos_grad)
                # dist.all_reduce(neg_grad)

                self._pos_grad += pos_grad
                self._neg_grad += neg_grad
                self.pos_neg = self._pos_grad / (self._neg_grad + 1e-10)

            def get_weight(self, cls_score):
                # we do not have information about pos grad and neg grad at beginning
                if self._pos_grad is None:
                    self._pos_grad = cls_score.new_zeros(self.num_classes)
                    self._neg_grad = cls_score.new_zeros(self.num_classes)
                    neg_w = cls_score.new_ones((self.n_i, self.n_c))
                    pos_w = cls_score.new_ones((self.n_i, self.n_c))
                else:
                    # the negative weight for objectiveness is always 1
                    neg_w = torch.cat([self.map_func(self.pos_neg), cls_score.new_ones(1)])
                    pos_w = 1 + self.alpha * (1 - neg_w)
                    neg_w = neg_w.view(1, -1).expand(self.n_i, self.n_c)
                    pos_w = pos_w.view(1, -1).expand(self.n_i, self.n_c)
                return pos_w, neg_w


    def set_configuration(self, cfg_in):
        cfg = self.get_configuration()
        cfg.merge_config(cfg_in)

        for opt in self._options:
            setattr(self, opt.attr, opt.parse(cfg.get_value(opt.config)))

        if self._use_learn:
            self.register_new_losses()

        if self._auto_update_model:
            from viame.pytorch.mmdet_compatibility import check_config_compatibility
            check_config_compatibility(self._net_config, self._weight_file, self._template)

        import matplotlib
        matplotlib.use('PS') # bypass multiple Qt load issues
        from mmdet.apis import init_detector

        gpu_string = 'cuda:' + str(self._gpu_index)
        mmdet_config = mmcv.Config.fromfile(self._net_config)
        if self._use_learn:
            def replace(conf, depth):
                if depth <= 0:
                    return
                try:
                    for k,v in conf.items():
                        if isinstance(v, dict):
                            replace(v, depth-1)
                        elif isinstance(v, list):
                            for element in v:
                                replace(element, depth-1)
                        else:
                            if k == 'num_classes':
                                conf[k] = self._num_classes
                            if k == 'CLASSES':
                                conf[k] = self.toolset['target_dataset'].categories
                except (AttributeError, TypeError, KeyError):
                    pass

            replace(mmdet_config, 500)
        self._model = init_detector(mmdet_config, self._weight_file, device=gpu_string)
        with open(self._class_names, "r") as in_file:
            self._labels = in_file.read().splitlines()

    def check_configuration(self, cfg):
        if not cfg.has_value("net_config"):
            logger.error("A network config file must be specified!")
            return False
        if not cfg.has_value("class_names"):
            logger.error("A class file must be specified!")
            return False
        if not cfg.has_value("weight_file"):
            logger.error("No weight file specified")
            return False
        return True

    def detect(self, image_data):
        input_image = image_data.asarray().astype('uint8')
        if self._rgb_to_bgr:
            input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)

        from mmdet.apis import inference_detector
        detections = inference_detector(self._model, input_image)

        if isinstance(detections, tuple):
            bbox_result, segm_result = detections
        else:
            bbox_result, segm_result = detections, None

        if np.size(bbox_result) > 0:
            bboxes = np.vstack(bbox_result)
        else:
            bboxes = []

        # convert segmentation masks
        masks = []
        if segm_result is not None:
            segms = mmcv.concat_list(segm_result)
            inds = np.where(bboxes[:, -1] > score_thr)[0]
            for i in inds:
                masks.append(maskUtils.decode(segms[i]).astype(np.bool))

        # collect labels
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]

        if np.size(labels) > 0:
            labels = np.concatenate(labels)
        else:
            labels = []

        # convert to kwiver format, apply threshold
        output = DetectedObjectSet()

        for bbox, label in zip(bboxes, labels):
            class_confidence = float(bbox[-1])
            if class_confidence < self._thresh:
                continue

            bbox_int = bbox.astype(np.int32)
            bounding_box = BoundingBoxD(bbox_int[0], bbox_int[1],
                                        bbox_int[2], bbox_int[3])

            class_name = self._labels[label]
            detected_object_type = DetectedObjectType(class_name, class_confidence)

            detected_object = DetectedObject(bounding_box,
                                             np.max(class_confidence),
                                             detected_object_type)
            output.add(detected_object)

        if np.size(labels) > 0 and self._display_detections:
            mmcv.imshow_det_bboxes(
                input_image,
                bboxes,
                labels,
                class_names=self._labels,
                score_thr=self._thresh,
                show=True)

        return output

def __vital_algorithm_register__():
    from kwiver.vital.algo import algorithm_factory

    # Register Algorithm
    implementation_name = "mmdet"

    if algorithm_factory.has_algorithm_impl_name(
      MMDetDetector.static_type_name(), implementation_name):
        return

    algorithm_factory.add_algorithm(implementation_name,
      "PyTorch MMDetection inference routine", MMDetDetector)

    algorithm_factory.mark_algorithm_as_loaded(implementation_name)
