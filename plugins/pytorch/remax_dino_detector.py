# This file is part of VIAME, and is distributed under the
# OSI-approved BSD 3-Clause License. See top-level LICENSE.txt file or
# https://github.com/VIAME/VIAME/blob/master/LICENSE.txt for details.
from __future__ import print_function
from __future__ import division
from collections import namedtuple
import json
import cv2
from kwiver.vital.algo import ImageObjectDetector
from kwiver.vital.types import (
    BoundingBoxD, DetectedObject, DetectedObjectSet, DetectedObjectType
)
import pickle
import mmcv
import numpy as np
import torch
import sys
import ubelt as ub

from collections import namedtuple


from viame.pytorch.remax.util.slconfig import SLConfig
from viame.pytorch.remax.model.dino import build_dino

_Option = namedtuple('_Option', ['attr', 'config', 'default', 'parse'])


class ReMaxDINODetector(ImageObjectDetector):
    """
    Implementation of ImageObjectDetector class
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

    ]

    def __init__(self):
        ImageObjectDetector.__init__(self)
        for opt in self._options:
            setattr(self, opt.attr, opt.default)


    def load_model( self ):
        self.dino_config = SLConfig.fromfile(self._net_config)
        self.dino_config.device = 'cuda'
        self.dino_config.checkpoint_path = self._weight_file
        self.model, self.criterion, self.postprocessors = build_dino(self.dino_config)
        checkpoint = torch.load(self._weight_file, map_location='cpu')
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()
        remax_file = open(self._remax_model_file, 'rb')
        self.remax = pickle.load(remax_file)
        remax_file.close()
        self.results = dict()

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
        
        self.ckpt = 0
        device = self._device
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
        test_data = torch.linalg.norm(test_data, dim=1, ord=self._norm_degree)

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
        for bbox, score, novelty_prob, label in zip(bboxes, scores, prob_test, labels):
            class_confidence = float(bbox[-1])
            if score < self._thresh:
                continue
            new_bbox = torch.tensor([bbox[0]*inputs.shape[2], bbox[1]*inputs.shape[3], bbox[2]*inputs.shape[2], bbox[3]*inputs.shape[3]])
            bbox_int = new_bbox.type(torch.int32)
            bounding_box = BoundingBoxD(bbox_int[0], bbox_int[1],
                                        bbox_int[2], bbox_int[3])
            class_name = self._labels[int(label)]
            detected_object_type = DetectedObjectType(class_name, class_confidence)
            detected_object = DetectedObject(bounding_box,
                                             np.max(class_confidence),
                                             detected_object_type)
            detected_object.add_note(":novelty=" + str(novelty_prob.item()))
            output.add(detected_object)

        inds = []
        for score in scores:
            if str(score.item()) in names:
                inds.append(names.index(str(score.item())))
        if labels.size()[0] > 0 and self._display_detections:
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
    implementation_name = "detector_remax_dino"

    if algorithm_factory.has_algorithm_impl_name(
      ReMaxDINODetector.static_type_name(), implementation_name):
        return

    algorithm_factory.add_algorithm(implementation_name,
      "ReMax inference routine", ReMaxDINODetector)

    algorithm_factory.mark_algorithm_as_loaded(implementation_name)