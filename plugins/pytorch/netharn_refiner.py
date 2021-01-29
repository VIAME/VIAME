# ckwg +29
# Copyright 2021 by Kitware, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice,
#  this list of conditions and the following disclaimer.
#
#  * Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.
#
#  * Neither name of Kitware, Inc. nor the names of any contributors may be used
#  to endorse or promote products derived from this software without specific
#  prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from __future__ import print_function

try:
    # Handle new kwiver structure
    from kwiver.vital.algo import RefineDetections

    from kwiver.vital.types import BoundingBox
    from kwiver.vital.types import DetectedObjectSet
    from kwiver.vital.types import DetectedObject
    from kwiver.vital.types import DetectedObjectType

except ImportError:
    # Handle old kwiver structure
    from vital.algo import RefineDetections

    from vital.types import BoundingBox
    from vital.types import DetectedObjectSet
    from vital.types import DetectedObject
    from vital.types import DetectedObjectType

import numpy as np  # NOQA
import ubelt as ub


class NetharnRefiner(RefineDetections):
    """
    Full-Frame Classifier

    Note: there is no kwiver base class for classifiers, so we abuse the
    detector interface.

    CommandLine:
        xdoctest -m ~/code/VIAME/plugins/pytorch/netharn_classifier.py NetharnRefiner --show

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
        >>> cname_to_prob = {cname: object_type.score(cname) for cname in class_names}
        >>> print('cname_to_prob = {}'.format(ub.repr2(cname_to_prob, nl=1, precision=4)))
    """

    def __init__(self):
        ImageObjectDetector.__init__(self)

        # kwiver configuration variables
        self._kwiver_config = {
            'deployed': "",
            'xpu': "0",
            'batch_size': "auto",
            'area_pivot': "0",
            'border_exclude': "-1"
        }

        # netharn variables
        self._thresh = None
        self.predictor = None

    def get_configuration(self):
        # Inherit from the base class
        cfg = super(ImageObjectDetector, self).get_configuration()
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

        pred_config = clf_predict.ClfPredictConfig()
        pred_config['batch_size'] = self._kwiver_config['batch_size']
        pred_config['deployed'] = self._kwiver_config['deployed']
        pred_config['xpu'] = self._kwiver_config['xpu']
        pred_config['input_dims'] = 'native'
        # (256, 256)
        self.predictor = clf_predict.ClfPredictor(pred_config)
        self.predictor._ensure_model()
        self._area_pivot = int(self._kwiver_config['area_pivot'])
        self._border_exclude = int(self._kwiver_config['border_exclude'])
        return True

    def check_configuration(self, cfg):
        if not cfg.has_value("deployed"):
            print("A network deploy file must be specified!")
            return False
        return True

    def refine(self, image_data, detections):

        if len( detections ) == 0:
            return detections

        img = image_data.asarray().astype('uint8')
        predictor = self.predictor

        # Extract patches for ROIs
        image_chips = []
        detection_ids = []

        for i, det in enumerate( detections ):
            # Extract chip for this detection
            bbox = det.bounding_box()

            bbox_min_x = int( bbox.min_x() )
            bbox_max_x = int( bbox.max_x() )
            bbox_min_y = int( bbox.min_y() )
            bbox_max_y = int( bbox.max_y() )

            bbox_width = bbox_max_x - bbox_min_x
            bbox_height = bbox_max_y - bbox_min_y

            bbox_area = bbox_width * bbox_height

            if self._area_pivot > 1 and bbox_area < self._area_pivot:
                continue
            if self._area_pivot < -1 and bbox_area > -self._area_pivot:
                continue

            if self._border_exclude > 0:
                if bbox_min_x <= self._border_exclude:
                    continue
                if bbox_min_y <= self._border_exclude:
                    continue
                if bbox_max_x >= img_max_x - self._border_exclude:
                    continue
                if bbox_max_y >= img_max_y - self._border_exclude:
                    continue

            crop = img[ bbox_min_y:bbox_max_y, bbox_min_x:bbox_max_x ]
            image_chips.append( crop )
            detection_ids.append( i )

        # Run classifier on ROIs
        classifications = list( predictor.predict( image_chips ) )

        # Put classifications back into detections
        output = DetectedObjectSet()

        for i, det in enumerate( detections ):
            if len( detections_ids ) == 0 or i != detection_ids[0]:
                output.add( det )
                continue

            new_class = classifications[0]

            if new_class.data.get('prob', None) is not None:
                # If we have a probability for each class, uses that
                class_names = list(new_class.classes)
                class_prob = new_class.prob
                detected_object_type = DetectedObjectType(class_names, class_prob)
                det.set_type(detected_object_type)
            else:
                # Otherwise we only have the score for the predicted calss
                class_name = new_class.classes[new_class.cidx]
                class_score = new_class.conf
                detected_object_type = DetectedObjectType(class_name, class_score)
                det.set_type(detected_object_type)

            output.add( det )
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
    """
    Note:

        We may be able to refactor somethign like this

        # In vital.py

        def _register_algorithm(cls, name=None, desc=''):
            if name is None:
                name = cls.__name__
            from vital.algo import algorithm_factory
            if not algorithm_factory.has_algorithm_impl_name(cls.static_type_name(), name):
                algorithm_factory.add_algorithm(name, desc, cls)
                algorithm_factory.mark_algorithm_as_loaded(name)

        def register_algorithm(name=None, desc=''):
            '''
            POC refactor of __vital_algorithm_register__ into a decorator
            '''
            def _wrapper(cls):
                _register_algorithm(cls, name, desc)
                return cls
            return _wrapper

        def lazy_register(cls, name=None, desc=''):
            ''' Alternate Proof-of-Concept '''
            def __vital_algorithm_register__():
                return _register_algorithm(cls, name, desc)
            return __vital_algorithm_register__

        # Then in your class
        import vital
        @vial.register_algorithm(desc="PyTorch Netharn classification routine")
        class MyAlgorithm(BaseAlgo):
            ...

        # OR if the currenty lazy structure is important
        import vital
        class MyAlgorithm(BaseAlgo):
            ...

        __vital_algorithm_register__ = vital.lazy_register(MyAlgorithm, desc="PyTorch Netharn classification routine")

        # We could also play with adding class member variables for the lazy
        # initialization. There is lots of room to make this better / easier.
    """
    from vital.algo import algorithm_factory

    # Register Algorithm
    implementation_name = "netharn"

    if not algorithm_factory.has_algorithm_impl_name(
            NetharnRefiner.static_type_name(), implementation_name ):
        algorithm_factory.add_algorithm(
            implementation_name, "PyTorch Netharn refiner routine",
            NetharnRefiner )

        algorithm_factory.mark_algorithm_as_loaded( implementation_name )
