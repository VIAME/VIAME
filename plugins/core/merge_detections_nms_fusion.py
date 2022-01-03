# ckwg +29
# Copyright 2022 by Kitware, Inc.
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

from kwiver.vital.algo import MergeDetections

from kwiver.vital.types import BoundingBoxD
from kwiver.vital.types import DetectedObjectSet
from kwiver.vital.types import DetectedObject
from kwiver.vital.types import DetectedObjectType

import numpy as np
import pandas as pd

import os
import ast
import time
import random

#from map_boxes import *
#from ensemble_boxes import *

##############################################################################
# Inspired by https://github.com/ZFTurbo/Weighted-Boxes-Fusion
# Using metrics from https://github.com/ZFTurbo/Mean-Average-Precision-for-Boxes
##############################################################################

def bb_intersection_over_union(A, B):
    xA = max(A[0], B[0])
    yA = max(A[2], B[2])
    xB = min(A[1], B[1])
    yB = min(A[3], B[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)

    if interArea == 0:
        return 0.0

    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (A[2] - A[0]) * (A[3] - A[1])
    boxBArea = (B[2] - B[0]) * (B[3] - B[1])

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def find_matching_box(boxes_list, new_box, match_iou):
    best_iou = match_iou
    best_index = -1
    for i in range(len(boxes_list)):
        box = boxes_list[i]
        iou = bb_intersection_over_union(box, new_box)
        if iou > best_iou:
            best_index = i
            best_iou = iou

    return best_index, best_iou

def get_pseudo_label(single_preds, multi_preds, img_ids, match_iou, label_flag, single_flag, file_path):
    if label_flag:
        if os.path.exists(file_path):
            with open(file_path,'rb') as f:
                single_preds = np.load(f, allow_pickle=True)
        else:
          for cur_id in img_ids:
            idx1 = single_preds[:,0] == cur_id
            cur_preds = single_preds[idx1]
            idx2 = multi_preds[:,0] == cur_id
            cur_multi = multi_preds[idx2]
            for i in range(cur_preds.shape[0]):
                best_idx, best_iou = find_matching_box(cur_multi[:,2:6], cur_preds[i,2:6], match_iou)
                if best_idx > 0:
                    cur_preds[i,1] = cur_multi[best_idx,1]

            single_preds[idx1,1] = cur_preds[:,1]

          idx = single_preds[:,1] > -1 if single_flag else single_preds[:,1] > 0
          single_preds = single_preds[idx]
        
          with open(file_path,'wb') as f:
            np.save(f, single_preds)
            
    return single_preds

def collect_boxlist(preds_set, cur_id):
    label_dic = {'generic_object_proposal':-1, 'sealion':0, 'Bull':1, 'Dead Pup':2, 'Fem':3, 'furseal':4, 'Juv':5, 'Pup':6, 'SAM':7}
    boxes_list, scores_list, labels_list = [], [], []
    for i in range(len(preds_set)):
        preds = preds_set[i]
        idx = preds[:,0] == cur_id
        boxes = preds[idx,3:]
        boxes = boxes[:,[0,2,1,3]]
        scores = preds[idx,2].T
        labels = preds[idx,1].T
        boxes_list.append(boxes)
        scores_list.append(scores)
        labels_list.append(labels)

    return boxes_list, scores_list, labels_list

def ensemble_box(preds_set, weights, iou_thr, skip_box_thr, sigma, fusion_type):
    assert len(preds_set) == len(weights)
    all_boxes = []
    all_labels = []
    all_ids = []

    boxes_list, scores_list, labels_list = collect_boxlist(preds_set, cur_id)
    if fusion_type == 'nmw':
        boxes, scores, labels = non_maximum_weighted(boxes_list, scores_list,
          labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    elif fusion_type == 'wbf':
        boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list,
          labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)

    ids = np.tile(cur_id, (boxes.shape[0], 1))
    cur_boxes = np.concatenate((np.expand_dims(scores, 1), boxes[:,[0,2,1,3]]), 1)

    if len(all_boxes):
        all_boxes = np.append(all_boxes, cur_boxes, 0)
        all_labels = np.append(all_labels, labels, 0)
        all_ids = np.append(all_ids, ids, 0)
    else:
        all_boxes = cur_boxes
        all_labels = labels
        all_ids = ids
           
    all_labels = np.array([int(label) for label in all_labels])

    res = pd.DataFrame(all_ids, columns=['ImageId'])
    res['LabelName'] = all_labels
    res['Conf'] = all_boxes[:,0]
    res['XMin'] = all_boxes[:,1]
    res['XMax'] = all_boxes[:,2]
    res['YMin'] = all_boxes[:,3]
    res['YMax'] = all_boxes[:,4]
    res = res[['ImageId', 'LabelName', 'Conf', 'XMin', 'XMax', 'YMin', 'YMax']].values

    return res

class MergeDetectionsNMSFusion( MergeDetections ):
    """
    Implementation of MergeDetections class
    """
    def __init__( self ):
        MergeDetections.__init__( self )

        self._fusion_type = 'nmw' # 'nmw', 'wbf'
        self._match_iou = 0.6
        self._iou_thr = 0.5
        self._skip_box_thr = 0.0001
        self._sigma = 0.1
        self._fusion_weights = [2, 0.5]

        # TODO: Delete
        self._height = 3840
        self._width = 5760

    def get_configuration( self ):
        cfg = super( MergeDetections, self ).get_configuration()

        cfg.set_value( "fusion_type", self._fusion_type )
        cfg.set_value( "match_iou", str( self._match_iou ) )
        cfg.set_value( "iou_thr", str( self._iou_thr ) )
        cfg.set_value( "skip_box_thr", str( self._skip_box_thr ) )
        cfg.set_value( "sigma", str( self._sigma ) )
        cfg.set_value( "fusion_weights", str( self._fusion_weights ) )
        return cfg

    def set_configuration( self, cfg_in ):
        cfg = self.get_configuration()
        cfg.merge_config( cfg_in )

        self._fusion_type = str( cfg.get_value( "fusion_type" ) )
        self._match_iou = float( cfg.get_value( "match_iou" ) )
        self._iou_thr = float( cfg.get_value( "iou_thr" ) )
        self._skip_box_thr = float( cfg.get_value( "skip_box_thr" ) )
        self._sigma = float( cfg.get_value( "sigma" ) )
        self._fusion_weights = ast.literal_eval( cfg.get_value( "fusion_weights" ) )

        return True

    def check_configuration( self, cfg ):
        return True

    def merge( self, det_sets ):

        # Get detection HL info in a list
        pred_sets = []
        for det_set in det_sets:
            pred_set = []
            for det in det_set:
                # Extract box info for this det
                bbox = det.bounding_box

                bbox_min_x = int( bbox.min_x() )
                bbox_max_x = int( bbox.max_x() )
                bbox_min_y = int( bbox.min_y() )
                bbox_max_y = int( bbox.max_y() )

                # Extract type info for this det
                if det.type is None:
                    continue

                #class_names = list( det.type.class_names() )
                #class_scores = [ det.type.score( n ) for n in class_names ]
                class_name = det.type.get_most_likely_class()
                class_score = det.type.score( class_name )

                pred_set.append( [ bbox_min_x, bbox_min_y, bbox_max_x, bbox_max_y,
                  class_name, class_score ] )
            pred_sets.append( pred_set )
  
        # Run merging algorithm
        #ensemble_preds = ensemble_box( preds_set, self._fusion_weights,
        #  self._iou_thr, self._skip_box_thr, self._sigma, self._fusion_type )
        ensemble_preds = []
  
        # Compile output detections
        output = DetectedObjectSet()

        for pred in ensemble_preds:
            score = pred[5]
            bbox = BoundingBoxD( pred[0], pred[1], pred[2], pred[3] )
            dot = DetectedObjectType( pred[4], score )
            det = DetectedObject( bbox, score, dot )
            output.add( det )

        return output

def __vital_algorithm_register__():
    from kwiver.vital.algo import algorithm_factory

    # Register Algorithm
    implementation_name  = "nms_fusion"

    if algorithm_factory.has_algorithm_impl_name(
      MergeDetectionsNMSFusion.static_type_name(), implementation_name ):
        return

    algorithm_factory.add_algorithm( implementation_name,
      "Fusion of multiple different detections", MergeDetectionsNMSFusion )

    algorithm_factory.mark_algorithm_as_loaded( implementation_name )
