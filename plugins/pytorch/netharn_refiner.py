# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

from __future__ import print_function

from kwiver.vital.algo import RefineDetections

from kwiver.vital.types import BoundingBoxD
from kwiver.vital.types import DetectedObjectSet
from kwiver.vital.types import DetectedObject
from kwiver.vital.types import DetectedObjectType

from distutils.util import strtobool

import numpy as np  # NOQA
import ubelt as ub

import math
import cv2

from .utilities import safe_crop


class NetharnRefiner(RefineDetections):
    """
    Full-Frame Classifier around Detection Sets

    CommandLine:
        xdoctest -m plugins/pytorch/netharn_classifier.py NetharnRefiner

    Example:
        >>> self = NetharnRefiner()
        >>> image_data = self.demo_image()
        >>> deployed_fpath = self.demo_deployed()
        >>> cfg_in = dict(
        >>>     deployed=deployed_fpath,
        >>>     xpu='0',
        >>> )
        >>> self.set_configuration(cfg_in)
        >>> detected_objects = self.classify(image_data)
        >>> object_type = detected_objects[0].type()
        >>> class_names = object_type.all_class_names()
    """

    def __init__(self):
        RefineDetections.__init__(self)

        # kwiver configuration variables
        self._kwiver_config = {
            'deployed': "",
            'xpu': "0",
            'batch_size': "auto",
            'area_pivot': "0",
            'area_lower_bound': "0",
            'area_upper_bound': "0",
            'border_exclude': "-1",
            'chip_method' : "",
            'chip_width' : "",
            'chip_expansion' : "1.0",
            'average_prior': "False",
            'scale_type_file': ""
        }

        # netharn variables
        self._thresh = None
        self.predictor = None

    def get_configuration(self):
        # Inherit from the base class
        cfg = super(RefineDetections, self).get_configuration()
        for key, value in self._kwiver_config.items():
            cfg.set_value(key, str(value))
        return cfg

    def set_configuration(self, cfg_in):
        import torch
        from bioharn import clf_predict
        cfg = self.get_configuration()
        _vital_config_update(cfg, cfg_in)

        for key in self._kwiver_config.keys():
            self._kwiver_config[key] = str(cfg.get_value(key))

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
                    for i in range(self._gpu_count):
                        single_gpu_mem = torch.cuda.get_device_properties(i).total_memory
                    if gpu_mem == 0:
                        gpu_mem = single_gpu_mem
                    else:
                        gpu_mem = min(gpu_mem, single_gpu_mem)
                if gpu_mem > 9e9:
                    self._kwiver_config['batch_size'] = 4
                elif gpu_mem >= 7e9:
                    self._kwiver_config['batch_size'] = 3

        pred_config = clf_predict.ClfPredictConfig()
        pred_config['batch_size'] = self._kwiver_config['batch_size']
        pred_config['deployed'] = self._kwiver_config['deployed']
        pred_config['xpu'] = self._kwiver_config['xpu']
        pred_config['input_dims'] = 'native' # (256, 256)

        self.predictor = clf_predict.ClfPredictor(pred_config)
        self.predictor._ensure_model()
        self._average_prior = strtobool(self._kwiver_config['average_prior'])
        self._area_pivot = int(self._kwiver_config['area_pivot'])
        self._area_lower_bound = int(self._kwiver_config['area_lower_bound'])
        self._area_upper_bound = int(self._kwiver_config['area_upper_bound'])
        self._border_exclude = int(self._kwiver_config['border_exclude'])
        self._chip_expansion = float(self._kwiver_config['chip_expansion'])

        if self._area_pivot < 0:
            self._area_upper_bound = -self._area_pivot
        elif self._area_pivot > 0:
            self._area_lower_bound = self._area_pivot

        # Load scale based on type file if enabled
        self._target_type_scales = dict()
        if self._kwiver_config['scale_type_file']:
            fin = open(self._kwiver_config['scale_type_file'], 'r')
            for line in fin.readlines():
                line = line.rstrip()
                parsed_line = line.split()
                if len(parsed_line) < 1:
                    continue
                target_area = float(parsed_line[-1])
                type_str = str( ' '.join(parsed_line[:-1]))
                self._target_type_scales[type_str] = target_area

        return True

    def check_configuration(self, cfg):
        if not cfg.has_value("deployed"):
            print("A network deploy file must be specified!")
            return False
        return True

    def compute_scale_factor(self, detections, min_scale = 0.10, max_scale = 10.0):
        cumulative = 0.0
        count = 0
        for i, item in enumerate(detections):
            if item.type is None:
                continue
            class_lbl = item.type.get_most_likely_class()
            if not class_lbl in self._target_type_scales:
                continue
            box_width = item.bounding_box.width()
            box_height = item.bounding_box.height()
            box_area = float(box_width * box_height)
            if box_area < 1.0:
                continue
            cumulative += math.sqrt(self._target_type_scales[ class_lbl ] / box_area)
            count += 1
        if count == 0:
            output = 1.0
        else:
            output = cumulative / count
        if output >= max_scale:
            output = max_scale
        if output <= min_scale:
            output = min_scale
        print("Computed image dim scale factor: " + str( output ))
        return output

    def refine(self, image_data, detections):

        if len(detections) == 0:
            return detections

        img = image_data.asarray().astype('uint8')
        predictor = self.predictor
        scale = 1.0

        img_max_x = np.shape(img)[1]
        img_max_y = np.shape(img)[0]

        if self._target_type_scales:
            scale = self.compute_scale_factor(detections)
            if scale != 1.0:
                img_max_x = int(img_max_x * scale)
                img_max_y = int(img_max_y * scale)
                img = cv2.resize(img, (img_max_x, img_max_y))

        # Extract patches for ROIs
        image_chips = []
        detection_ids = []

        for i, det in enumerate(detections):
            # Extract chip for this detection
            bbox = det.bounding_box

            bbox_min_x = int(bbox.min_x() * scale)
            bbox_max_x = int(bbox.max_x() * scale)
            bbox_min_y = int(bbox.min_y() * scale)
            bbox_max_y = int(bbox.max_y() * scale)

            if self._kwiver_config['chip_method'] == "fixed_width" or \
               self._kwiver_config['chip_method'] == "native_square":
                if self._kwiver_config['chip_method'] == "fixed_width":
                    chip_width = int( self._kwiver_config['chip_width'] )
                else:
                    chip_width = max( ( bbox_max_x - bbox_min_x ), ( bbox_max_y - bbox_min_y ) )
                half_width = int( chip_width / 2 )

                bbox_min_x = int( ( bbox_min_x + bbox_max_x ) / 2 ) - half_width
                bbox_min_y = int( ( bbox_min_y + bbox_max_y ) / 2 ) - half_width
                bbox_max_x = bbox_min_x + chip_width
                bbox_max_y = bbox_min_y + chip_width

            if self._chip_expansion != 1.0:
                bbox_width = int( ( bbox_max_x - bbox_min_x ) * self._chip_expansion )
                bbox_height = int( ( bbox_max_y - bbox_min_y ) * self._chip_expansion )

                bbox_min_x = int( ( bbox_min_x + bbox_max_x ) / 2 - bbox_width / 2 )
                bbox_min_y = int( ( bbox_min_y + bbox_max_y ) / 2 - bbox_height / 2 )
                bbox_max_x = bbox_min_x + bbox_width
                bbox_max_y = bbox_min_y + bbox_height

            if self._border_exclude > 0:
                if bbox_min_x <= self._border_exclude:
                    continue
                if bbox_min_y <= self._border_exclude:
                    continue
                if bbox_max_x >= img_max_x - self._border_exclude:
                    continue
                if bbox_max_y >= img_max_y - self._border_exclude:
                    continue
            else:
                if bbox_min_x < 0:
                    bbox_min_x = 0
                if bbox_min_y < 0:
                    bbox_min_y = 0
                if bbox_max_x > img_max_x:
                    bbox_max_x = img_max_x
                if bbox_max_y > img_max_y:
                    bbox_max_y = img_max_y

            bbox_area = ( bbox_max_x - bbox_min_x ) * ( bbox_max_y - bbox_min_y )

            if self._area_lower_bound > 0 and bbox_area < self._area_lower_bound:
                continue
            if self._area_upper_bound > 0 and bbox_area > self._area_upper_bound:
                continue

            crop = safe_crop( img, bbox_min_x, bbox_min_y, bbox_max_x, bbox_max_y )
            image_chips.append( crop )
            detection_ids.append( i )

        # Run classifier on ROIs
        classifications = list(predictor.predict(image_chips))

        # Put classifications back into detections
        output = DetectedObjectSet()

        for i, det in enumerate(detections):
            if len(detection_ids) == 0 or i != detection_ids[0]:
                output.add(det)
                continue

            new_class = classifications[0]

            if new_class.data.get('prob', None) is not None:
                # If we have a probability for each class, uses that
                class_names = list(new_class.classes)
                class_scores = list(new_class.prob)
            else:
                # Otherwise we only have the score for the predicted class
                class_names = [ new_class.classes[new_class.cidx] ]
                class_scores = [ new_class.conf ]

            if self._average_prior and det.type is not None:
                priors = det.type
                prior_names = priors.class_names()
                for name in prior_names:
                    if name in class_names:
                        class_scores[ class_names.index(name) ] += priors.score(name)
                    else:
                        class_names.append(name)
                        class_scores.append(priors.score(name))
                for i in range(len(class_scores)):
                    class_scores[i] = class_scores[i] * 0.5

            detected_object_type = DetectedObjectType(class_names, class_scores)
            det.type = detected_object_type

            output.add(det)
            detection_ids.pop(0)
            classifications.pop(0)

        return output


def _vital_config_update(cfg, cfg_in):
    """
    Treat a vital Config object like a python dictionary

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

def __vital_algorithm_register__():
    from kwiver.vital.algo import algorithm_factory

    # Register Algorithm
    implementation_name = "netharn"

    if not algorithm_factory.has_algorithm_impl_name(
            NetharnRefiner.static_type_name(), implementation_name):
        algorithm_factory.add_algorithm(
            implementation_name, "PyTorch Netharn refiner routine",
            NetharnRefiner)

        algorithm_factory.mark_algorithm_as_loaded(implementation_name)
