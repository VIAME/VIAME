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
import time
import random
import csv
import pdb
import ast

from map_boxes import *
from ensemble_boxes import *

##############################################################################
# Inspired by https://github.com/ZFTurbo/Weighted-Boxes-Fusion
# Using metrics from https://github.com/ZFTurbo/Mean-Average-Precision-for-Boxes
##############################################################################

def bb_intersection_over_union( A, B ):
    xA = max( A[0], B[0] )
    yA = max( A[1], B[1] )
    xB = min( A[2], B[2] )
    yB = min( A[3], B[3] )

    # Compute the area of intersection rectangle
    interArea = max( 0, xB - xA ) * max( 0, yB - yA )

    if interArea == 0:
        return 0.0

    # Compute the area of both the prediction and ground-truth rectangles
    boxAArea = ( A[2] - A[0] ) * ( A[3] - A[1] )
    boxBArea = ( B[2] - B[0] ) * ( B[3] - B[1] )

    iou = interArea / float( boxAArea + boxBArea - interArea )
    return iou

def find_matching_box( boxes_list, new_box, match_iou ):
    best_iou = match_iou
    best_index = -1
    for i in range( len( boxes_list ) ):
        box = boxes_list[i]
        iou = bb_intersection_over_union( box, new_box )
        if iou > best_iou:
            best_index = i
            best_iou = iou
    return best_index, best_iou

def ensemble_box(boxes_list, scores_list, labels_list, weights, iou_thr,
                 skip_box_thr, sigma, fusion_type):

    assert len( boxes_list ) == len( weights )
    assert len( boxes_list ) == len( scores_list )
    assert len( boxes_list ) == len( labels_list )

    all_boxes = []
    all_labels = []
    all_ids = []

    if fusion_type == 'nmw':
        boxes, scores, labels = non_maximum_weighted( boxes_list, scores_list, \
          labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr )
    elif fusion_type == 'wbf':
        boxes, scores, labels = weighted_boxes_fusion( boxes_list, scores_list, \
          labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr )

    return boxes, scores, labels

class MergeDetectionsNMSFusion( MergeDetections ):
    """
    Implementation of MergeDetections class
    """
    def __init__( self ):
        MergeDetections.__init__( self )

        # Configuration settings for fusion
        self._fusion_type = 'nmw' # 'nmw', 'wbf'
        self._match_iou = 0.6
        self._iou_thr = 0.75
        self._skip_box_thr = 0.0001
        self._sigma = 0.1
        self._fusion_weights = [1, 1.5, 1]

        # These parameters should be deprecated longer-term
        self._height = 3840
        self._width = 5760

        # Complex parameters
        self._label_dic = { 'Background':0, 'Bull':1, 'Fem':2,
          'Juv':3, 'Pup':4, 'SAM':5, 'Furseal':6, 'Pup':7, 'Adult':8 }
        self._pseudo_dic = { 8: [ 1, 2, 3, 5 ] }
        self._pseudo_ind = { 1: [ 2, 0 ] }
        self._id_dic = {}

    def get_configuration( self ):
        cfg = super( MergeDetections, self ).get_configuration()

        cfg.set_value( "fusion_type", self._fusion_type )
        cfg.set_value( "match_iou", str( self._match_iou ) )
        cfg.set_value( "iou_thr", str( self._iou_thr ) )
        cfg.set_value( "skip_box_thr", str( self._skip_box_thr ) )
        cfg.set_value( "sigma", str( self._sigma ) )
        cfg.set_value( "fusion_weights", str( self._fusion_weights ) )

        cfg.set_value( "height", str( self._height ) )
        cfg.set_value( "width", str( self._width ) )

        cfg.set_value( "label_dic", str( self._label_dic ) )
        cfg.set_value( "pseudo_dic", str( self._pseudo_dic ) )
        cfg.set_value( "pseudo_ind", str( self._pseudo_ind ) )

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

        self._height = float( cfg.get_value( "height" ) )
        self._width = float( cfg.get_value( "width" ) )

        self._label_dic = ast.literal_eval( cfg.get_value( "label_dic" ) )
        self._pseudo_dic = ast.literal_eval( cfg.get_value( "pseudo_dic" ) )
        self._pseudo_ind = ast.literal_eval( cfg.get_value( "pseudo_ind" ) )

        for label in self._label_dic:
            label_id = self._label_dic[ label ]
            if not label_id in self._id_dic:
                self._id_dic[ label_id ] = label

        return True

    def check_configuration( self, cfg ):
        return True

    def merge( self, det_sets ):

        # Get detection HL info in a few lists
        boxes_list = []
        scores_list = []
        labels_list = []

        for det_set in det_sets:
            box_list = []
            score_list = []
            label_list = []

            for det in det_set:
                bbox = det.bounding_box

                bbox_min_x = float( bbox.min_x() ) / self._width
                bbox_max_x = float( bbox.max_x() ) / self._width
                bbox_min_y = float( bbox.min_y() ) / self._height
                bbox_max_y = float( bbox.max_y() ) / self._height

                if det.type is None:
                    continue

                class_name = det.type.get_most_likely_class()
                class_score = det.type.score( class_name )

                if not class_name in self._label_dic:
                    continue

                box_list.append( [ bbox_min_x, bbox_min_y, bbox_max_x, bbox_max_y ] )
                score_list.append( class_score )
                label_list.append( self._label_dic[ class_name ] )

            boxes_list.append( box_list )
            scores_list.append( score_list )
            labels_list.append( label_list )

        # Utilize pseudonym lists when upsampling categories
        for chk_list_ind in self._pseudo_ind:
            for chk_ind in range( len( boxes_list[ chk_list_ind ] ) ):
                orig_label = labels_list[ chk_list_ind ][ chk_ind ]
                if not orig_label in self._pseudo_dic:
                    continue
                for other_list_ind in self._pseudo_ind[ chk_list_ind ]:
                    best_idx, best_iou = find_matching_box(
                        boxes_list[ other_list_ind ],
                        boxes_list[ chk_list_ind ][ chk_ind ],
                        self._match_iou )
                    if best_idx > 0:
                        new_label = labels_list[ other_list_ind ][ best_idx ]
                        if new_label in self._pseudo_dic[ orig_label ]:
                            labels_list[ chk_list_ind ][ chk_ind ] = new_label
                            break

        # Run merging algorithm
        boxes_list, scores_list, labels_list = ensemble_box(
          boxes_list, scores_list, labels_list,
          self._fusion_weights, self._iou_thr,
          self._skip_box_thr, self._sigma, self._fusion_type )

        # Compile output detections
        output = DetectedObjectSet()

        for i in range( len( boxes_list ) ):
            score = scores_list[i]
            box = boxes_list[i]
            label_id = labels_list[i]
            bbox = BoundingBoxD( box[0] * self._width, box[2] * self._height,
                                 box[1] * self._width, box[3] * self._height )
            dot = DetectedObjectType( self._id_dic[ label_id ], score )
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
