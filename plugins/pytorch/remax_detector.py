# This file is part of VIAME, and is distributed under the
# OSI-approved BSD 3-Clause License. See top-level LICENSE.txt file or
# https://github.com/VIAME/VIAME/blob/master/LICENSE.txt for details.
from __future__ import print_function
from __future__ import division
from collections import namedtuple
import sys

import cv2
from distutils.util import strtobool
from kwiver.vital.algo import ImageObjectDetector
from kwiver.vital.types import (
    BoundingBoxD, DetectedObject, DetectedObjectSet, DetectedObjectType
)

from kwiver.vital.algo import TrainDetector


import os
from pathlib import Path
import pickle
import mmcv
import numpy as np
import torch
import scipy
from shutil import copyfile
import sys
import ubelt as ub
import yaml

from collections import namedtuple


from .remax.util.slconfig import SLConfig
from .remax.util import box_ops
from .remax.model.dino import build_dino
from .remax.util.coco import build as build_dataset

from .remax.ReMax import ReMax


_Option = namedtuple('_Option', ['attr', 'config', 'default', 'parse'])


class ReMaxDetector(ImageObjectDetector):
    """
    Implementation of ImageObjectDetector class
    """

    # Config-option-based attribute specifications, used in __init__,
    # get_configuration, and set_configuration
    _options = [
        _Option('_net_config', 'net_config', '', str),
        _Option('_remax_config_file', 'remax_config_file', '', str),
        _Option('_remax_model_file', 'remax_model_file', '', str),
        _Option('_weight_file', 'weight_file', '', str),
        _Option('_class_names', 'class_names', '', str),
        _Option('_thresh', 'thresh', 0.01, float),
        _Option('_gpu_index', 'gpu_index', "0", str),
        _Option('_num_classes', 'num_classes', 1, int),
        _Option('_display_detections', 'display_detections', False, strtobool),
        _Option('_template', 'template', "", str),
        _Option('_auto_update_model', 'auto_update_model', True, strtobool),
        _Option('_rgb_to_bgr', 'rgb_to_bgr', True, strtobool),
    ]

    def __init__(self):
        ImageObjectDetector.__init__(self)
        for opt in self._options:
            setattr(self, opt.attr, opt.default)


    def load_model( self ):
        self.dino_config = SLConfig.fromfile(self._net_config)
        self.dino_config.device = 'cuda'
        self.dino_config.checkpoint_path = self._weight_file
        self.dino_config
        self.model, self.criterion, self.postprocessors = build_dino(self.dino_config)
        checkpoint = torch.load(self._weight_file, map_location='cpu')
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()
        print(self._remax_model_file)
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
        cfg.merge_config( cfg_in )
        for opt in self._options:
            setattr(self, opt.attr, opt.parse(cfg.get_value(opt.config)))
        p = Path(self._remax_config_file)
        r = p.read_text()
        config_file = yaml.safe_load(r)
        self.config = config_file['params']
        
        self.ckpt = 0 # TODO: not sure about this
        self.stage = 'base' # TODO: also not sure about this 

        device = self.config['device']
        if ub.iterable(device):
            self.device = device
        else:
            if device == -1:
                self.device = list(range(torch.cuda.device_count()))
            else:
                self.device = [device]
        if len(self.device) > torch.cuda.device_count():
            self.device = self.device[:torch.cuda.device_count()]
            
        self.load_model()
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
        if self._rgb_to_bgr:
            input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
        inputs = torch.from_numpy(input_image).permute(2,0,1).unsqueeze(0).type(torch.float).cuda()
        output = self.model.cuda()(inputs)
        output = self.postprocessors['bbox'](output, torch.Tensor([[1.0, 1.0]]).cuda())[0]
        logits = output['logits']
        scores = output['scores']
        labels = output['labels']
        feats = output['feats']
        bboxes = output['boxes']
        #bboxes = box_ops.box_xyxy_to_cxcywh(output['boxes'])
        pred_label = [str(int(item)) for item in labels]
        pred_feats = feats  
        test_feats = {}
        for i in range(len(pred_label)):
            label = pred_label[i]
            feats = pred_feats[i]
            if label not in test_feats.keys():
                test_feats[label] = feats.unsqueeze(0)
            else:
                test_feats[label] = torch.cat((test_feats[label], feats.unsqueeze(0)), 0)
        prob_test = []
        test_data = []
        output = DetectedObjectSet()
        if len(test_feats.keys()) == 0:
            return output
        for cls in test_feats.keys():
            test_data.append(test_feats[cls])

        test_data = torch.cat(test_data, dim=0)
        test_data = torch.linalg.norm(test_data, dim=1, ord=1)

        #max_index_test = torch.max(test_data, dim=1).indices
        # convert to kwiver format, apply threshold
        for row in test_data:
            max_value_row = torch.max(row, dim=0).values

            sample_ReScore = self.remax.ReScore(max_value_row)
            if sample_ReScore.isnan().any():
                raise Exception
            prob_test.append(sample_ReScore.view(-1))
        
        prob_test = torch.cat(prob_test,dim=0)
        names = []
        for bbox, score, novelty_prob in zip(bboxes, scores, prob_test):
            class_confidence = float(bbox[-1])
            if score < 0.1:
                continue
            new_bbox = torch.tensor([bbox[0]*inputs.shape[2], bbox[1]*inputs.shape[3], bbox[2]*inputs.shape[2], bbox[3]*inputs.shape[3]])
            bbox_int = new_bbox.type(torch.int32)
            bounding_box = BoundingBoxD(bbox_int[0], bbox_int[1],
                                        bbox_int[2], bbox_int[3])
            class_name = str(score.item())
            if class_name not in names:
                names.append(str(class_name))
            #print(type(class_confidence), type(prob_test))
            detected_object_type = DetectedObjectType(class_name, novelty_prob)
            detected_object = DetectedObject(bounding_box,
                                             novelty_prob,
                                             detected_object_type)
            #detected_object.add_note(":novelty_prob=" + str(novelty_prob))
            output.add(detected_object)
        inds = []
        for score in scores:
            if str(score.item()) in names:
                inds.append(names.index(str(score.item())))
        if labels.size()[0] > 0 and self._display_detections:
            mmcv.imshow_det_bboxes(
                input_image,
                bboxes,
                inds,
                class_names=names,
                score_thr=novelty_prob,
                show=True)
        return output

def __vital_algorithm_register__():
    from kwiver.vital.algo import algorithm_factory

    # Register Algorithm
    implementation_name = "remax"

    if algorithm_factory.has_algorithm_impl_name(
      ReMaxDetector.static_type_name(), implementation_name):
        return

    algorithm_factory.add_algorithm(implementation_name,
      "ReMax inference routine", ReMaxDetector)

    algorithm_factory.mark_algorithm_as_loaded(implementation_name)
