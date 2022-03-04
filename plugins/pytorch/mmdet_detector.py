# This file is part of VIAME, and is distributed under the
# OSI-approved BSD 3-Clause License. See top-level LICENSE.txt file or
# https://github.com/VIAME/VIAME/blob/master/LICENSE.txt for details.

from collections import namedtuple
import sys

from distutils.util import strtobool
from kwiver.vital.algo import ImageObjectDetector
from kwiver.vital.types import (
    BoundingBoxD, DetectedObject, DetectedObjectSet, DetectedObjectType
)
import mmcv
import numpy as np


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
        _Option('_display_detections', 'display_detections', False, strtobool),
        _Option('_template', 'template', "", str),
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

    def set_configuration(self, cfg_in):
        cfg = self.get_configuration()
        cfg.merge_config(cfg_in)

        for opt in self._options:
            setattr(self, opt.attr, opt.parse(cfg.get_value(opt.config)))

        from viame.arrows.pytorch.mmdet_compatibility import check_config_compatibility
        check_config_compatibility(self._net_config, self._weight_file, self._template)

        import matplotlib
        matplotlib.use('PS') # bypass multiple Qt load issues
        from mmdet.apis import init_detector

        gpu_string = 'cuda:' + str(self._gpu_index)
        self._model = init_detector(self._net_config, self._weight_file, device=gpu_string)
        with open(self._class_names, "r") as in_file:
            self._labels = in_file.read().splitlines()

    def check_configuration(self, cfg):
        if not cfg.has_value("net_config"):
            print("A network config file must be specified!")
            return False
        if not cfg.has_value("class_names"):
            print("A class file must be specified!")
            return False
        if not cfg.has_value("weight_file"):
            print("No weight file specified")
            return False
        return True

    def detect(self, image_data):
        input_image = image_data.asarray().astype('uint8')

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
