# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

# -*- coding: utf-8 -*-
"""
CommandLine:
    # The GMM motion detector these processes host, as the size measurement
    # example runs it
    cd [viame-build]/install/examples/size_measurement
    bash measure_via_gmm_oriented_boxes.sh

SeeAlso
    ~/code/VIAME/packages/kwiver/vital/bindings/python/vital/types
"""

import numpy as np

from kwiver.vital.types import (
    BoundingBoxD,
    DetectedObject,
    DetectedObjectSet,
    DetectedObjectType,
    ImageContainer,
    ObjectTrackState,
    ObjectTrackSet,
    Point2d,
    Track,
)

from kwiver.sprokit.processes.kwiver_process import KwiverProcess
from kwiver.sprokit.pipeline import process
from kwiver.sprokit.pipeline import datum  # NOQA

import ubelt as ub
import os
import itertools as it

from . import stereo_algos as ctalgo

from kwiver.vital import vital_logging

logger = vital_logging.getLogger(__name__)
print = logger.info


# TODO: add something similar in sprokit proper
TMP_SPROKIT_PROCESS_REGISTRY = []


def tmp_sprokit_register_process(name=None, doc=''):
    def _wrp(cls):
        name_ = name
        if name is None:
            name_ = cls.__name__
        TMP_SPROKIT_PROCESS_REGISTRY.append((name_, doc, cls))
        return cls
    return _wrp


def opencv_setup_config(self, default_params):
    if isinstance(default_params, dict):
        default_params = list(it.chain(*default_params.values()))
    for pi in default_params:
        self.add_config_trait(pi.name, pi.name, str(pi.default), pi.doc)
        self.declare_config_using_trait(pi.name)


def tmp_smart_cast_config(self):
    # import ubelt as ub
    # import utool as ut
    config = {}
    keys = [k for k in list(self.available_config())
            if not k.startswith('_')]
    for key in keys:
        strval = self.config_value(key)
        # print('strval = {!r}'.format(strval))
        try:
            val = eval(strval, {}, {})
        except Exception:
            val = strval
        config[key] = val
    return config


@tmp_sprokit_register_process(name='gmm_motion_detector',
                              doc='preliminatry fish detection')
class GMMDetectFishProcess(KwiverProcess):
    """
    This process gets an image and detection_set as input, extracts each chip,
    does postprocessing and then sends the extracted chip to the output port.

    """
    # --------------------------------------------------------------------------
    def __init__(self, conf):
        print('conf = {!r}'.format(conf))
        logger.debug(' ----- init ' + self.__class__.__name__)
        KwiverProcess.__init__(self, conf)

        opencv_setup_config(self, ctalgo.GMMForegroundObjectDetector.default_params())

        # set up required flags
        optional = process.PortFlags()
        required = process.PortFlags()
        required.add(self.flag_required)

        #  declare our input port ( port-name,flags)
        self.declare_input_port_using_trait('image', required)

        self.declare_output_port_using_trait('detected_object_set', optional )

    # --------------------------------------------------------------------------
    def _configure(self):
        logger.debug(' ----- configure ' + self.__class__.__name__)
        config = tmp_smart_cast_config(self)
        print('detector config = {}'.format(ub.repr2(config, nl=2)))
        self.detector = ctalgo.GMMForegroundObjectDetector(**config)
        self._base_configure()

    # --------------------------------------------------------------------------
    def _dowork(self, img_container):
        """
        Helper to decouple the algorithm and pipeline logic

        CommandLine:
            xdoctest viame.processes.opencv.processes GMMDetectFishProcess._dowork

        Example:
            >>> from viame.processes.opencv.processes import *
            >>> from kwiver.vital.types import ImageContainer
            >>> import kwiver.sprokit.pipeline.config
            >>> # construct dummy process instance
            >>> conf = kwiver.sprokit.pipeline.config.empty_config()
            >>> self = GMMDetectFishProcess(conf)
            >>> self._configure()
            >>> # construct test data
            >>> from vital.util import VitalPIL
            >>> from PIL import Image as PILImage
            >>> pil_img = PILImage.open(ub.grabdata('https://i.imgur.com/Jno2da3.png'))
            >>> pil_img = PILImage.fromarray(np.zeros((512, 512, 3), dtype=np.uint8))
            >>> img_container = ImageContainer(VitalPIL.from_pil(pil_img))
            >>> # Initialize the background detector by sending 10 black frames
            >>> for i in range(10):
            >>>     empty_set = self._dowork(img_container)
            >>> # now add a white box that should be detected
            >>> np_img = np.zeros((512, 512, 3), dtype=np.uint8)
            >>> np_img[300:340, 220:380] = 255
            >>> img_container = ImageContainer.fromarray(np_img)
            >>> detection_set = self._dowork(img_container)
            >>> assert len(detection_set) == 1
            >>> obj = detection_set[0]
        """
        # This should be read as np.uint8
        np_img = img_container.asarray()

        detection_set = DetectedObjectSet()
        ct_detections = self.detector.detect(np_img)

        for detection in ct_detections:
            bbox = BoundingBoxD(*detection.bbox.coords)
            mask = detection.mask.astype(np.uint8)
            vital_mask = ImageContainer.fromarray(mask)
            dot = DetectedObjectType("Motion", 1.0)
            obj = DetectedObject(bbox, 1.0, dot, mask=vital_mask)
            detection_set.add(obj)
        return detection_set

    def _step(self):
        logger.debug(' ----- ' + self.__class__.__name__ + ' step')
        # grab image container from port using traits
        img_container = self.grab_input_using_trait('image')

        # Process image container
        detection_set = self._dowork(img_container)

        # Push the output
        self.push_to_port_using_trait('detected_object_set', detection_set)

        self._base_step()


def __sprokit_register__():

    from kwiver.sprokit.pipeline import process_factory

    module_name = 'python_' + __name__

    if process_factory.is_process_module_loaded(module_name):
        return

    for name, doc, cls in TMP_SPROKIT_PROCESS_REGISTRY:
        process_factory.add_process(name, doc, cls)

    process_factory.mark_process_module_as_loaded(module_name)
