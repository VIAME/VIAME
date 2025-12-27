# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

from __future__ import print_function

from kwiver.vital.algo import ImageObjectDetector

try:
    from kwiver.vital.types import BoundingBoxD
except ImportError:
    from kwiver.vital.types import BoundingBox as BoundingBoxD

from kwiver.vital.types import DetectedObjectSet
from kwiver.vital.types import DetectedObject
from kwiver.vital.types import DetectedObjectType

import scriptconfig as scfg
import numpy as np


class UltralyticsConfig(scfg.DataConfig):
    """
    The configuration for :class:`UltralyticsDetector`.
    """
    weight = scfg.Value(None, help='path to a checkpoint on disk')
    device = scfg.Value('auto', help='a torch device string or number')
    thresh = scfg.Value(0.1, help='confidence threshold')
    iou_thresh = scfg.Value(0.45, help='iou threshold for nms')

    def __post_init__(self):
        super().__post_init__()


class UltralyticsDetector(ImageObjectDetector):
    """
    Implementation of ImageObjectDetector class

    References:
        https://huggingface.co/atticus-carter/NOAA_AFSC_MML_Iceseals_31K

    Ignore:
        developer testing
        we pyenv3.8.19

        # Running inside VIAME

        # Double check this module is installed
        ls /opt/noaa/viame/lib/python3.10/site-packages/viame/arrows/pytorch/

        cd /opt/noaa/viame/examples/object_detection
        wget https://huggingface.co/atticus-carter/NOAA_AFSC_MML_Iceseals_31K/resolve/main/NOAA_AFSC_MML_Iceseals_31K.pt
        cp /opt/noaa/viame/configs/pipelines/templates/detector_ultralytics.pipe demo_detector_ultralytics.pipe
        sed -i 's|\[-MODEL-FILE-\]|NOAA_AFSC_MML_Iceseals_31K.pt|g' demo_detector_ultralytics.pipe
        sed -i 's|\[-WINDOW-OPTION-\]|original_and_resized|g' demo_detector_ultralytics.pipe

        export PYTHONIOENCODING=utf-8
        kwiver runner demo_detector_ultralytics.pipe \
                      -s input:video_filename=input_image_list_small_set.txt

    Example:
        >>> import sys, ubelt
        >>> sys.path.append(ubelt.expandpath('~/code/VIAME/plugins/pytorch/'))
        >>> from ultralytics_detector import *  # NOQA
        >>> from ultralytics_detector import _kwiver_to_kwimage_detections
        >>> import kwimage
        >>> #
        >>> weight = UltralyticsDetector.demo_weights()
        >>> image_data = UltralyticsDetector.demo_image()
        >>> #
        >>> self = UltralyticsDetector()
        >>> cfg_in = dict(
        >>>     weight=weight,
        >>>     device='cuda:0',
        >>> )
        >>> self.set_configuration(cfg_in)
        >>> detected_objects = self.detect(image_data)
        >>> dets = _kwiver_to_kwimage_detections(detected_objects)
        >>> print(f'dets = {ub.urepr(dets, nl=1)}')
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(image_data.asarray())
        >>> dets.draw()
        >>> kwplot.show_if_requested()
    """

    def __init__(self):
        ImageObjectDetector.__init__(self)

        # kwiver configuration variables
        self._config = UltralyticsConfig()
        self._wrapped = {
            'model': None,
            'classes': None,
        }

    @classmethod
    def demo_weights(cls):
        """
        Returns a path to a ultralytics deployed model

        Returns:
            PathLike: file path to YOLO checkpoint
        """
        import ubelt as ub
        weight_fpath = ub.grabdata(
            'https://huggingface.co/atticus-carter/NOAA_AFSC_MML_Iceseals_31K/resolve/main/NOAA_AFSC_MML_Iceseals_31K.pt',
            hash_prefix='04fb4e04ad7ae13b4c9a7d1b92a23c9535f5ab656b80224b9a4293639d18551b', hasher='sha256')
        return weight_fpath

    @classmethod
    def demo_image(cls):
        """
        Returns an image which can be run through the detector

        Returns:
            ImageContainer: an image to test on
        """
        from PIL import Image as PILImage
        from kwiver.vital.util import VitalPIL
        from kwiver.vital.types import ImageContainer
        import kwimage
        image_fpath = kwimage.grab_test_image_fpath()
        pil_img = PILImage.open(image_fpath)
        image_data = ImageContainer(VitalPIL.from_pil(pil_img))
        return image_data

    def get_configuration(self):
        # Inherit from the base class
        cfg = super(ImageObjectDetector, self).get_configuration()
        for key, value in self._config.items():
            cfg.set_value(key, str(value))
        return cfg

    def _build_model(self):
        from ultralytics import YOLO
        import kwcoco
        import torch
        device = self._config.device
        if device == 'auto':
            if torch.cuda.is_available():
                device = torch.device('cuda:0')
            else:
                device = torch.device('cpu')
        model = YOLO(self._config.weight)
        classes = kwcoco.CategoryTree.coerce(list(model.names.values()))
        self._wrapped['model'] = model
        self._wrapped['classes'] = classes
        self._wrapped['device'] = device

    def set_configuration(self, cfg_in):
        cfg = self.get_configuration()

        # Imports used across this func
        import os
        # HACK: merge config doesn't support dictionary input
        _vital_config_update(cfg, cfg_in)

        for key in self._config.keys():
            self._config[key] = str(cfg.get_value(key))

        if os.name == 'nt':
            os.environ["KWIMAGE_DISABLE_TORCHVISION_NMS"] = "1"

        self._build_model()
        return True

    def check_configuration(self, cfg):
        # if not cfg.has_value("deployed"):
        #     print("A network deploy file must be specified!")
        #     return False
        return True

    def detect(self, image_data):
        full_rgb = image_data.asarray()

        model = self._wrapped['model']
        classes = self._wrapped['classes']
        device = self._wrapped['device']

        results_image = model.predict(
            source=full_rgb,
            conf=self._config.thresh,   # Confidence threshold
            iou=self._config.iou_thresh,    # IoU threshold
            device=device
        )

        assert len(results_image) == 1
        result = results_image[0]
        detections = ultralytics_result_to_kwimage(result, classes)

        detections = detections.numpy()
        # flags = detections.scores >= self._thresh
        # detections = detections.compress(flags)

        # convert to kwiver format
        output = _kwimage_to_kwiver_detections(detections)
        return output


def ultralytics_result_to_kwimage(result, classes=None):
    import kwimage
    # TODO: make sure the model names map is always in order
    # if not then modify it to be in the correct order
    dets = kwimage.Detections(
        boxes=kwimage.Boxes(result.boxes.xywh, format='xywh'),
        scores=result.boxes.conf,
        class_idxs=result.boxes.cls.int(),
        classes=classes,
    )
    return dets


def _vital_config_update(cfg, cfg_in):
    """
    Treat a vital Config object like a python dictionary

    TODO: move to a utility module

    Args:
        cfg (kwiver.vital.config.config.Config): config to update
        cfg_in (dict | kwiver.vital.config.config.Config): new values
    """
    # vital cfg.merge_config doesnt support dictionary input
    if isinstance(cfg_in, dict):
        for key, value in cfg_in.items():
            if cfg.has_value(key):
                cfg.set_value(key, str(value))
            else:
                raise KeyError('cfg has no key={}'.format(key))
    else:
        cfg.merge_config(cfg_in)
    return cfg


def _kwiver_to_kwimage_detections(detected_objects):
    """
    Convert vital detected object sets to kwimage.Detections

    TODO: move to a utility module

    Args:
        detected_objects (kwiver.vital.types.DetectedObjectSet)

    Returns:
        kwimage.Detections
    """
    import ubelt as ub
    import kwimage
    boxes = []
    scores = []
    class_idxs = []

    classes = []
    if len(detected_objects) > 0:
        obj = ub.peek(detected_objects)
        try:
            classes = obj.type.all_class_names()
        except AttributeError:
            classes = obj.type().all_class_names()

    for obj in detected_objects:
        box = obj.bounding_box()
        ltrb = [box.min_x(), box.min_y(), box.max_x(), box.max_y()]
        score = obj.confidence()
        cname = obj.type().get_most_likely_class()
        cidx = classes.index(cname)
        boxes.append(ltrb)
        scores.append(score)
        class_idxs.append(cidx)

    dets = kwimage.Detections(
        boxes=kwimage.Boxes(np.array(boxes), 'ltrb'),
        scores=np.array(scores),
        class_idxs=np.array(class_idxs),
        classes=classes,
    )
    return dets


def _kwimage_to_kwiver_detections(detections):
    """
    Convert kwimage detections to kwiver deteted object sets

    TODO: move to a utility module

    Args:
        detected_objects (kwimage.Detections)

    Returns:
        kwiver.vital.types.DetectedObjectSet
    """
    from kwiver.vital.types.types import ImageContainer, Image

    segmentations = None
    # convert segmentation masks
    if 'segmentations' in detections.data:
        segmentations = detections.data['segmentations']

    try:
        boxes = detections.boxes.to_ltrb()
    except Exception:
        boxes = detections.boxes.to_tlbr()

    scores = detections.scores
    class_idxs = detections.class_idxs

    if not segmentations:
        # Placeholders
        segmentations = (None,) * len(boxes)

    # convert to kwiver format, apply threshold
    detected_objects = DetectedObjectSet()

    for tlbr, score, cidx, seg in zip(boxes.data, scores, class_idxs, segmentations):
        class_name = detections.classes[cidx]

        bbox_int = np.round(tlbr).astype(np.int32)
        bounding_box = BoundingBoxD(
            bbox_int[0], bbox_int[1], bbox_int[2], bbox_int[3])

        detected_object_type = DetectedObjectType(class_name, score)
        detected_object = DetectedObject(
            bounding_box, score, detected_object_type)
        if seg:
            mask = seg.to_relative_mask().numpy().data
            detected_object.mask = ImageContainer(Image(mask))

        detected_objects.add(detected_object)
    return detected_objects


def __vital_algorithm_register__():
    from kwiver.vital.algo import algorithm_factory

    # Register Algorithm
    implementation_name = "ultralytics"

    if algorithm_factory.has_algorithm_impl_name(
            UltralyticsDetector.static_type_name(), implementation_name):
        return

    algorithm_factory.add_algorithm(
        implementation_name, "PyTorch Ultralytics detection routine",
        UltralyticsDetector)

    algorithm_factory.mark_algorithm_as_loaded(implementation_name)
