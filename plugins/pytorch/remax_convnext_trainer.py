# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

# pylint: disable=wrong-import-order
# ckwg +29
# Copyright 2019 by Kitware, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#    * Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
#    * Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
#    * Neither name of Kitware, Inc. nor the names of any contributors may be used
#    to endorse or promote products derived from this software without specific
#    prior written permission.
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
from __future__ import division

import time
import os
import pickle
import mmcv
import numpy as np
import torch
import scipy
import sys
import torchvision.transforms as transforms

from collections import namedtuple

from distutils.util import strtobool
from kwiver.vital.algo import TrainDetector

from .remax.util.coco import CocoDetection

from .remax.ReMax import ReMax

try:
    import learn.algorithms.MMDET.register_modules
    import torch
    import torch.nn.functional as F
    from mmdet.models.builder import LOSSES
    from functools import partial
    use_learn = True
except ModuleNotFoundError:
    use_learn = False

_Option = namedtuple('_Option', ['attr', 'config', 'default', 'parse'])

class ReMaxConvNextTrainer( TrainDetector ):
    _options = [
        _Option('_gpu_count', 'gpu_count', -1, int),
        _Option('_output_directory', 'output_directory', '', str),
        _Option('_debug_mode', 'debug_mode', False, bool),
        _Option('_feature_cache', 'feature_cache', '', str),
        _Option('_net_config', 'net_config', '', str),
        _Option('_weight_file', 'weight_file', '', str),
        _Option('_gpu_index', 'gpu_index', "0", str),
        _Option('_num_classes', 'num_classes', 60, int),
        _Option('_norm_degree', 'norm_degree', 1, int),
        _Option('_template', 'template', "", str),
        _Option('_auto_update_model', 'auto_update_model', True, strtobool),
    ]
    def __init__( self ):
        TrainDetector.__init__(self)
        for opt in self._options:
            setattr(self, opt.attr, opt.default)

        self.image_root = ''
        self._config_file = ""
        self._seed_weights = ""
        self._train_directory = "deep_training"
        self._output_directory = "category_models"
        self._output_prefix = "custom_cfrnn"
        self._pipeline_template = ""
        self._gpu_count = -1
        self._random_seed = "none"
        self._tmp_training_file = "training_truth.pickle"
        self._tmp_validation_file = "validation_truth.pickle"
        self._validate = True
        self._gt_frames_only = False
        self._launcher = "pytorch"  # "none, pytorch, slurm, or mpi" 
        self._train_in_new_process = True
        self._categories = []

    def check_configuration( self, cfg ):
        return True
    
    def get_configuration(self):
        # Inherit from the base class
        cfg = super( TrainDetector, self ).get_configuration()

        cfg.set_value( "config_file", self._config_file )
        cfg.set_value( "seed_weights", self._seed_weights )
        cfg.set_value( "train_directory", self._train_directory )
        cfg.set_value( "output_directory", self._output_directory )
        cfg.set_value( "output_prefix", self._output_prefix )
        cfg.set_value( "pipeline_template", self._pipeline_template )
        cfg.set_value( "gpu_count", str( self._gpu_count ) )
        cfg.set_value( "random_seed", str( self._random_seed ) )
        cfg.set_value( "validate", str( self._validate ) )
        cfg.set_value( "gt_frames_only", str( self._gt_frames_only ) )
        cfg.set_value( "launcher", str( self._launcher ) )
        cfg.set_value( "train_in_new_process", str( self._train_in_new_process ) )
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
        self.register_new_losses()
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
        
    def __getstate__( self ):
        return self.__dict__

    def __setstate__( self, dict ):
        self.__dict__ = dict
    
    def bbox_iou(self, boxA, boxB):
        # https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
        # ^^ corrected.

        # Determine the (x, y)-coordinates of the intersection rectangle
        t0 = time.time()
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interW = xB - xA + 1
        interH = yB - yA + 1

        # Correction: reject non-overlapping boxes
        if interW <=0 or interH <=0 :
            return -1.0

        interArea = interW * interH
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou


    def match_bboxes(self, bbox_gt, bbox_pred, IOU_THRESH=0.3):
        '''
        Given sets of true and predicted bounding-boxes,
        determine the best possible match.
        Parameters
        ----------
        bbox_gt, bbox_pred : N1x4 and N2x4 np array of bboxes [x1,y1,x2,y2]. 
        The number of bboxes, N1 and N2, need not be the same.
        
        Returns
        -------
        (idxs_true, idxs_pred, ious, labels)
            idxs_true, idxs_pred : indices into gt and pred for matches
            ious : corresponding IOU value of each match
            labels: vector of 0/1 values for the list of detections
        '''
        n_true = bbox_gt.shape[0]
        n_pred = bbox_pred.shape[0]
        MAX_DIST = 1.0
        MIN_IOU = 0.0
        # NUM_GT x NUM_PRED
        iou_matrix = np.zeros((n_true, n_pred))
        for i in range(n_true):
            for j in range(n_pred):
                iou_matrix[i, j] = self.bbox_iou(bbox_gt[i,:], bbox_pred[j,:])
        if n_pred > n_true:
            # there are more predictions than ground-truth - add dummy rows
            diff = n_pred - n_true
            iou_matrix = np.concatenate( (iou_matrix,
                                            np.full((diff, n_pred), MIN_IOU)),
                                        axis=0)
        if n_true > n_pred:
            # more ground-truth than predictions - add dummy columns
            diff = n_true - n_pred
            iou_matrix = np.concatenate( (iou_matrix,
                                            np.full((n_true, diff), MIN_IOU)),
                                        axis=1)

        # call the Hungarian matching
        idxs_true, idxs_pred = scipy.optimize.linear_sum_assignment(1 - iou_matrix)
        if (not idxs_true.size) or (not idxs_pred.size):
            ious = np.array([])
        else:
            ious = iou_matrix[idxs_true, idxs_pred]

        # remove dummy assignments
        sel_pred = idxs_pred<n_pred
        idx_pred_actual = idxs_pred[sel_pred]
        idx_gt_actual = idxs_true[sel_pred]
        ious_actual = iou_matrix[idx_gt_actual, idx_pred_actual]
        sel_valid = (ious_actual > IOU_THRESH)
        label = sel_valid.astype(int)

        return idx_gt_actual[sel_valid], idx_pred_actual[sel_valid], ious_actual[sel_valid], label
    
    def build_dataset ( self ):
        """ Creates and returns a dataloader for the OD model
        """
        self._config.coco_path = str(os.path.join(self._train_directory, 'train_data_coco.json')) # the path of coco
        self._config.fix_size = False
        self._config.image_root = self.image_root
        self._config.modelname = None
        self._config.masks = False
        transformations = transforms.Compose([
                        transforms.PILToTensor()
                    ])
        #build_dataset(image_set='train', args=self.dino_config)
        return CocoDetection(self.image_root, self._config.coco_path,
                             transforms=transformations, return_masks=self._config.masks)
    
    def get_features(self, dataloader):
        """ Creates and returns a nxm torch tensor of features
            retrieved from the detections of the OD algorithm
            where n is the number of detections and m is the
            length of the feature vector
        

        Args:
            dataloader (_type_): _description_

        Returns:
            _type_: _description_
            _type_: _description_
        """
        from mmdet.apis import inference_detector
        train_data = []
        for image, targets in dataloader:
            image = image.permute(1,2,0)
            input_image = image.cpu().detach().numpy().astype('uint8')
            detections = inference_detector(self._model, input_image)
            if isinstance(detections, tuple):
                bbox_result, _ = detections
            else:              
                bbox_result, _ = detections, None
        
            if np.size(bbox_result) > 0:
                bboxes = np.vstack(bbox_result)
            else:
                continue

            image = image.permute(2,0,1)
            feats = self._model.extract_feat(image.float().unsqueeze(0).cuda())
            feat_rois = torch.zeros(bboxes.shape)
            feat_rois[:, 1:] = torch.from_numpy(bboxes)[:, :4]
            #feat_rois = self.scale_rois(feat_rois, image)
            bbox_feats = self._model.roi_head.bbox_roi_extractor[2](
            feats, feat_rois.cuda())
            match_targets = torch.empty(targets['boxes'].shape)
            for i in range(len(targets['boxes'])):
                match_targets[i][0] = targets['boxes'][i][0]
                match_targets[i][1] = targets['boxes'][i][1]
                match_targets[i][2] = targets['boxes'][i][0] + targets['boxes'][i][2]
                match_targets[i][3] = targets['boxes'][i][1] + targets['boxes'][i][3]
            bboxes = torch.tensor(bboxes)
            _, idxs_pred, _, _ = self.match_bboxes(targets['boxes'], bboxes[:,:4])
            train_feats = bbox_feats[idxs_pred]
            train_data.append(train_feats.detach().cpu())
        train_data = torch.cat(train_data, dim=0)

        if self._debug_mode and not os.path.exists(self._feature_cache):
            train_data = torch.save(train_data, self._feature_cache)
        return train_data
    
    def update_model( self ):
        dataset_train = self.build_dataset()
        self._debug_mode = False
        if self._debug_mode and os.path.exists(self._feature_cache):
            print("using feature cache")
            with open(self._feature_cache, 'rb') as f:
                train_data = torch.load(f)
        else:
            print("generating bbox features")
            train_data = self.get_features(dataset_train)
            saved = torch.save(train_data, self._feature_cache)
        train_data = torch.linalg.norm(train_data, dim=1, ord=self._norm_degree)
        
        self.remax_model = ReMax(train_data)

        self.save_final_model()

    def save_final_model( self ):
        output_model_name = "remax.pkl"
        output_model = os.path.join( self._output_directory,
            output_model_name )
        
        with open(output_model, 'wb') as f:
            pickle.dump(self.remax_model, f)


        if not os.path.exists( output_model ):
            print( "\nModel failed to finsh training\n" )
            sys.exit( 0 )

        # Output additional completion text
        print( "\nWrote finalized model to " + output_model )

    def add_data_from_disk( self, categories, train_files, train_dets, test_files, test_dets ):
        print('add_data_from_disk')

        if len( train_files ) != len( train_dets ):
            print( "Error: train file and groundtruth count mismatch" )
            return
        
        cats = []
        self.cats = categories.all_class_names()
        for cat in self.cats:
            cat_id = categories.get_class_id(cat)
            cats.append({'name': cat, 'id': int(cat_id)})

        for split in [train_files, test_files]:
            is_train = ( split == train_files )
            num_images = len(split)
            
            images = []
            annotations = []
            annotation_id = 0

            for index in range(num_images):
                filename = split[index]
                if not self.image_root:
                    self.image_root = os.path.dirname(filename) # TODO: there might be a better way to get this?
                    print("self.image_root", self.image_root)

                img = mmcv.image.imread(filename)
                height, width = img.shape[:2]
                targets = train_dets[index] if is_train else test_dets[index]

                image_dct = {'file_name': filename, # dataset.root + '/' + dataset.image_fnames[index],
                    'height': height,
                    'width': width,
                    'id': int(index)
                }

                image_anno_ctr = 0

                for target in targets:
                    bbox = [  target.bounding_box.min_x(),
                              target.bounding_box.min_y(),
                              target.bounding_box.max_x(),
                              target.bounding_box.max_y() ] # tlbr
                    
                    # skip bboxes with 0 width or height
                    if (bbox[2] - bbox[0]) <= 0 or (bbox[3] - bbox[1]) <= 0:
                        continue

                    class_lbl = target.type.get_most_likely_class()
                    if categories is not None:
                        class_id = categories.get_class_id( class_lbl )
                    else:
                        if class_lbl not in self._categories:
                            self._categories.append( class_lbl )
                        class_id = self._categories.index( class_lbl )

                    annotation_dct = {'bbox': [bbox[0], bbox[1], (bbox[2]-bbox[0]), (bbox[3]-bbox[1])],  # coco annotations in file need x, y, width, height
                        'image_id': int(index),
                        'area': (bbox[2]-bbox[0]) * (bbox[3]-bbox[1]),
                        'category_id': class_id,
                        'id': annotation_id,
                        'iscrowd': 0
                    }
                    annotations.append(annotation_dct)
                    annotation_id += 1
                    image_anno_ctr += 1

                images.append(image_dct)

            coco_format_json = dict(
                images=images,
                annotations=annotations,
                categories=cats)

            fn = 'train_data_coco.json' if is_train else 'test_data_coco.json'
            output_file = os.path.join(self._train_directory, fn)
            mmcv.dump(coco_format_json, output_file)
            
            print(f"Transformed the dataset into COCO style: {output_file} "
                  f"Num Images {len(images)} and Num Annotations: {len(annotations)}")
            
def __vital_algorithm_register__():
    from kwiver.vital.algo import algorithm_factory

    # Register Algorithm
    implementation_name = "remax_convnext"

    if algorithm_factory.has_algorithm_impl_name(
      ReMaxConvNextTrainer.static_type_name(), implementation_name):
        return

    algorithm_factory.add_algorithm(implementation_name,
      "PyTorch MMDetection inference routine", ReMaxConvNextTrainer)

    algorithm_factory.mark_algorithm_as_loaded(implementation_name)
