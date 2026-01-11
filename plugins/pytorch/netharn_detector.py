# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

from __future__ import print_function

import logging

from kwiver.vital.algo import ImageObjectDetector

logger = logging.getLogger(__name__)

from .utilities import (
    vital_config_update,
    kwimage_to_kwiver_detections,
    register_vital_algorithm,
)


class NetharnDetector(ImageObjectDetector):
    """
    Implementation of ImageObjectDetector class

    CommandLine:
        xdoctest -m plugins/pytorch/netharn_detector.py NetharnDetector --show

    Example:
        >>> self = NetharnDetector()
        >>> image_data = self.demo_image()
        >>> deployed_fpath = self.demo_deployed()
        >>> cfg_in = dict(
        >>>     deployed=deployed_fpath,
        >>>     xpu='0',
        >>> )
        >>> self.set_configuration(cfg_in)
        >>> detected_objects = self.detect(image_data)
    """

    def __init__(self):
        ImageObjectDetector.__init__(self)

        # kwiver configuration variables
        self._kwiver_config = {
            'deployed': "",
            'thresh': 0.01,
            'xpu': "0",
            'batch_size': "auto",
            'input_string': ""
        }

        # netharn variables
        self._thresh = None
        self.predictor = None

    def demo_deployed(self):
        """
        Returns a path to a netharn deployed model

        Returns:
            str: file path to a scallop detector
        """
        import ubelt as ub
        url = 'https://data.kitware.com/api/v1/file/5dd3eb8eaf2e2eed3508d604/download'
        deployed_fpath = ub.grabdata(
            url, fname='deploy_MM_CascadeRCNN_myovdqvi_035_MVKVVR_fix3.zip',
            appname='viame', hash_prefix='22a1eeb18c9e5706f6578e66abda1e97',
            hasher='sha512')
        return deployed_fpath

    def demo_image(self):
        """
        Returns an image which can be run through the detector

        Returns:
            ImageContainer: an image of a scallop
        """
        from PIL import Image as PILImage
        from kwiver.vital.util import VitalPIL
        from kwiver.vital.types import ImageContainer
        import ubelt as ub
        url = 'https://data.kitware.com/api/v1/file/5dcf0d1faf2e2eed35fad5d1/download'
        image_fpath = ub.grabdata(
            url, fname='scallop.jpg', appname='viame',
            hash_prefix='3bd290526c76453bec7', hasher='sha512')
        pil_img = PILImage.open(image_fpath)
        image_data = ImageContainer(VitalPIL.from_pil(pil_img))
        return image_data

    def get_configuration(self):
        # Inherit from the base class
        cfg = super(ImageObjectDetector, self).get_configuration()
        for key, value in self._kwiver_config.items():
            cfg.set_value(key, str(value))
        return cfg

    def set_configuration(self, cfg_in):
        cfg = self.get_configuration()

        # Imports used across this func
        import os
        import torch
        from viame.arrows.pytorch.netharn.bio import detect_predict

        vital_config_update(cfg, cfg_in)

        for key in self._kwiver_config.keys():
            self._kwiver_config[key] = str(cfg.get_value(key))
        self._kwiver_config['thresh'] = float(self._kwiver_config['thresh'])

        self._thresh = float(self._kwiver_config['thresh'])

        if self._kwiver_config['batch_size'] == "auto":
            self._kwiver_config['batch_size'] = 2

            if torch.cuda.is_available():
                gpu_mem = 0
                if len(self._kwiver_config['xpu']) == 1 and \
                  self._kwiver_config['xpu'] != 0:
                    gpu_id = int(self._kwiver_config['xpu'])
                    gpu_mem = torch.cuda.get_device_properties(gpu_id).total_memory
                else:
                    self._gpu_count = torch.cuda.device_count()
                    for i in range( self._gpu_count ):
                        single_gpu_mem = torch.cuda.get_device_properties(i).total_memory
                    if gpu_mem == 0:
                        gpu_mem = single_gpu_mem
                    else:
                        gpu_mem = min(gpu_mem, single_gpu_mem)
                if gpu_mem > 9e9:
                    self._kwiver_config['batch_size'] = 4
                elif gpu_mem >= 7e9:
                    self._kwiver_config['batch_size'] = 3

        if os.name == 'nt':
            os.environ["KWIMAGE_DISABLE_TORCHVISION_NMS"] = "1"

        pred_config = detect_predict.DetectPredictConfig()
        pred_config['batch_size'] = self._kwiver_config['batch_size']
        pred_config['deployed'] = self._kwiver_config['deployed']
        if torch.cuda.is_available():
            pred_config['xpu'] = self._kwiver_config['xpu']
        else:
            pred_config['xpu'] = "cpu"
        self.predictor = detect_predict.DetectPredictor(pred_config)

        self.predictor._ensure_model()
        return True

    def check_configuration(self, cfg):
        if not cfg.has_value("deployed"):
            logger.error("A network deploy file must be specified!")
            return False
        return True

    def detect(self, image_data):
        full_rgb = image_data.asarray().astype('uint8')

        if len(self._kwiver_config['input_string']) > 0:
            dict_or_image = {self._kwiver_config['input_string']: full_rgb}
        else:
            dict_or_image = full_rgb

        predictor = self.predictor
        detections = predictor.predict(dict_or_image)

        # apply threshold
        flags = detections.scores >= self._thresh
        detections = detections.compress(flags)

        # convert to kwiver format
        output = kwimage_to_kwiver_detections(detections)
        return output


def __vital_algorithm_register__():
    register_vital_algorithm(
        NetharnDetector, "netharn", "PyTorch Netharn detection routine"
    )
