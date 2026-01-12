# This file is part of VIAME, and is distributed under the
# OSI-approved BSD 3-Clause License. See top-level LICENSE.txt file or
# https://github.com/VIAME/VIAME/blob/master/LICENSE.txt for details.

from collections import namedtuple
import json

import cv2
from distutils.util import strtobool
from kwiver.vital.algo import ImageObjectDetector
from kwiver.vital.types import (
    BoundingBoxD, DetectedObject, DetectedObjectSet, DetectedObjectType
)

import pickle
import mmcv
from mmdet.apis import inference_detector
import numpy as np
import torch
import sys

from collections import namedtuple

_Option = namedtuple('_Option', ['attr', 'config', 'default', 'parse'])


class ReMaxMMDetDetector(ImageObjectDetector):
    """
    This class is meant as a template for inference to be ran
    on new MMDet model and ReMax model pairs. The training process
    needs to be ran first to generate a set of remax weights for
    the desired dataset and object detection model. The set_configuration
    file is where the remax model, mmdet config and mmdet models are loaded
    and initialized. the detect function runs an image through the
    object detection algorithm, then uses the features of the output
    to generate a novelty probability for the given bounding box.
    The same normalization needs to be used for both the training and
    inference pipelines.
    """

    # Config-option-based attribute specifications, used in __init__,
    # get_configuration, and set_configuration
    _options = [
        _Option('_display_detections', 'display_detections', '', str),
        _Option('_net_config', 'net_config', '', str),
        _Option('_remax_model_file', 'remax_model_file', '', str),
        _Option('_weight_file', 'weight_file', '', str),
        _Option('_class_names', 'class_names', '', str),
        _Option('_thresh', 'thresh', 0.1, float),
        _Option('_gpu_index', 'gpu_index', "0", str),
        _Option('_template', 'template', "", str),
        _Option('_device', 'device', "", str),
        _Option('_rgb_to_bgr', 'rgb_to_bgr', "", str),
        _Option('_norm_degree', 'norm_degree', 1, int),
        _Option('_num_classes', 'num_classes', 60, int),
        _Option('_template', 'template', "", str),
        _Option('_auto_update_model', 'auto_update_model', True, strtobool),

    ]
    def __init__(self):
        ImageObjectDetector.__init__(self)
        for opt in self._options:
            setattr(self, opt.attr, opt.default)



    def load_model( self ):
        remax_file = open(self._remax_model_file, 'rb')
        self.remax = pickle.load(remax_file)
        remax_file.close()


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

        import matplotlib
        matplotlib.use('PS') # bypass multiple Qt load issues
        from mmdet.apis import init_detector

        gpu_string = 'cuda:' + str(self._gpu_index)
        mmdet_config = mmcv.Config.fromfile(self._net_config)
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
                        # print(k,v)
                        if k == 'num_classes':
                            conf[k] = self._num_classes
                        if k == 'CLASSES':
                            conf[k] = self.toolset['target_dataset'].categories
            except:
                pass
        self._config = mmdet_config
        replace(mmdet_config, 500)
        self._model = init_detector(mmdet_config, self._weight_file, device=gpu_string)
        self.load_model()
        with open(self._class_names, "r") as in_file:
            self._labels = in_file.read().splitlines()
    
    def __getstate__( self ):
        return self.__dict__

    def __setstate__( self, dict ):
        self.__dict__ = dict

    def check_configuration(self, cfg):
        return True

    def detect(self, image_data):
        input_image = image_data.asarray().astype('uint8')
        if self._rgb_to_bgr:
            input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
        # Running inference on the original detector 
        # May need to modify the extraction of bboxes from the output
        # depending on how custom mmdet model outputs boxes
        detections = inference_detector(self._model, input_image)
        if isinstance(detections, tuple):
            bbox_result, _ = detections
        else:              
            bbox_result, _ = detections, None
        if np.size(bbox_result) > 0:
            bboxes = np.vstack(bbox_result)
        else:
            bboxes = np.array([])

        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
    
        if np.size(labels) > 0:
            labels = np.concatenate(labels)
        else:
            labels = []
            return DetectedObjectSet()
        output = DetectedObjectSet()
        input_image = torch.from_numpy(input_image).permute(2,0,1)
        feats = self._model.extract_feat(input_image.float().unsqueeze(0).cuda())
        feat_rois = torch.zeros(bboxes.shape)
        feat_rois[:, 1:] = torch.from_numpy(bboxes)[:, :4]

        # Requires that the roi head of the Object Detection mmdet model 
        # has a bbox_roi_extractor attribute. May need to be modified
        # for the specific roi_head that is used
        bbox_feats = self._model.roi_head.bbox_roi_extractor(
        feats, feat_rois.cuda())
        test_data = torch.linalg.norm(bbox_feats, dim=1, ord=self._norm_degree)
        prob_test = []

        for row in test_data:
            # generate novelty probability from feature vector
            sample_ReScore = self.remax.ReScore(row)
            if sample_ReScore.isnan().any():
                raise Exception
            prob_test.append(sample_ReScore.view(-1))

        from PIL import Image, ImageDraw
        img = Image.fromarray(image_data.asarray())
        img1 = ImageDraw.Draw(img)
        prob_test = torch.cat(prob_test,dim=0)
        names = []
        for prob in prob_test:
            if prob not in names:
                names.append(str(prob.item()))
        for bbox, label, novelty_prob in zip(bboxes, labels, prob_test):
            img1.rectangle([(bbox[0], bbox[1]), (bbox[2], bbox[3])])
            class_confidence = float(bbox[-1])
            if class_confidence < self._thresh:
                continue
            bbox_int = bbox.astype('uint16')
            bounding_box = BoundingBoxD(bbox_int[0], bbox_int[1],
                                        bbox_int[2], bbox_int[3])
            class_name = self._labels[label]
            detected_object_type = DetectedObjectType(class_name, class_confidence)
            detected_object = DetectedObject(bounding_box,
                                             np.max(class_confidence),
                                             detected_object_type)
            
            # add in novelty prob attribute to detected object
            detected_object.add_note(":novelty=" + str(novelty_prob.item()))
            output.add(detected_object)
        if True and self._display_detections:

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
    implementation_name = "example_detector"

    if algorithm_factory.has_algorithm_impl_name(
      ReMaxMMDetDetector.static_type_name(), implementation_name):
        return

    algorithm_factory.add_algorithm(implementation_name,
      "ReMax inference routine", ReMaxMMDetDetector)

    algorithm_factory.mark_algorithm_as_loaded(implementation_name)
