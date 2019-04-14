# ckwg +29
# Copyright 2019 by Kitware, Inc.
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
from __future__ import division

from vital.algo import TrainDetector

from vital.types import BoundingBox
from vital.types import CategoryHierarchy
from vital.types import DetectedObjectSet
from vital.types import DetectedObject

from PIL import Image
from distutils.util import strtobool

import argparse
import numpy as np
import torch
import pickle
import os

from mmcv import Config

from mmdet import __version__
from mmdet.datasets import get_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.apis import init_dist
from mmdet.apis import get_root_logger
from mmdet.apis import set_random_seed

from mmdet.datasets.custom import CustomDataset


class MMTrainDetector( TrainDetector ):
  """
  Implementation of TrainDetector class
  """
  def __init__( self ):
    TrainDetector.__init__( self )

    self._config_file = ""
    self._seed_weights = ""
    self._train_directory = "deep_training"
    self._gpu_count = -1
    self._integer_labels = "true"
    self._launcher = "none"    # "none, pytorch, slurm, or mpi" 
    self._random_seed = "none"
    self._validate = "false"
    self._tmp_annotation_file = "annotations.pickle"

  def get_configuration( self ):
    # Inherit from the base class
    cfg = super( TrainDetector, self ).get_configuration()

    cfg.set_value( "config_file", self._config_file )
    cfg.set_value( "seed_weights", self._seed_weights )
    cfg.set_value( "train_directory", self._train_directory )
    cfg.set_value( "gpu_count", str( self._gpu_count ) )
    cfg.set_value( "integer_labels", str( self._integer_labels ) )
    cfg.set_value( "launcher", str( self._launcher ) )
    cfg.set_value( "random_seed", str( self._random_seed ) )
    cfg.set_value( "validate", str( self._validate ) )

    return cfg

  def set_configuration( self, cfg_in ):
    cfg = self.get_configuration()
    cfg.merge_config( cfg_in )

    self._config_file = str( cfg.get_value( "config_file" ) )
    self._seed_weights = str( cfg.get_value( "seed_weights" ) )
    self._train_directory = str( cfg.get_value( "train_directory" ) )
    self._gpu_count = int( cfg.get_value( "gpu_count" ) )
    self._integer_labels = strtobool( cfg.get_value( "integer_labels" ) )
    self._launcher = str( cfg.get_value( "launcher" ) )
    self._validate = strtobool( cfg.get_value( "validate" ) )

    self._training_data = []

    self._cfg = Config.fromfile( self._config_file )

    if self._cfg.get( 'cudnn_benchmark', False ):
      torch.backends.cudnn.benchmark = True

    if self._train_directory is not None:
      self._cfg.work_dir = self._train_directory
      self._groundtruth_store = os.path.join(
        self._train_directory, self._tmp_annotation_file )
      if not os.path.exists( self._train_directory ):
        os.mkdir( self._train_directory )
    else:
      self._groundtruth_store = self._tmp_annotation_file

    if self._seed_weights is not None:
      self._cfg.resume_from = self._seed_weights

    if self._gpu_count > 0:
      self._cfg.gpus = self._gpu_count
    else:
      self._cfg.gpus = torch.cuda.device_count()

    if self._cfg.checkpoint_config is not None:
      self._cfg.checkpoint_config.meta = dict(
        mmdet_version=__version__, config=self._cfg.text )

    if self._launcher == 'none':
      self._distributed = False
    else:
      self._distributed = True
      init_dist( self._launcher, **cfg.dist_params )

    self._logger = get_root_logger( self._cfg.log_level )
    self._logger.info( 'Distributed training: {}'.format( self._distributed ) )

    if self._random_seed is not "none":
      logger.info( 'Set random seed to {}'.format( self._random_seed ) )
      set_random_seed( int( self._random_seed ) )

    self._model = build_detector(
      self._cfg.model, train_cfg=self._cfg.train_cfg, test_cfg=self._cfg.test_cfg )

  def check_configuration( self, cfg ):
    if not cfg.has_value( "config_file" ) or len(cfg.get_value( "config_file")) == 0:
      print( "A config file must be specified!" )
      return False
    return True

  def add_data_from_disk( self, categories, train_files, train_dets, test_files, test_dets ):
    if len( train_files ) != len( train_dets ):
      print( "Error: train file and groundtruth count mismatch" )
      return

    for filename, groundtruth in zip( train_files, train_dets ):
      entry = dict()

      im = Image.open( filename, 'r' )
      width, height = im.size

      annotations = dict()

      boxes = np.ndarray( ( 0, 4 ) )
      labels = np.ndarray( 0 )

      for i, item in enumerate( groundtruth ):

        obj_id = item.type().get_most_likely_class()

        if categories.has_class_id( obj_id ):

          obj_box = [ [ item.bounding_box().min_x(),
                        item.bounding_box().min_y(),
                        item.bounding_box().max_x(),
                        item.bounding_box().max_y() ] ]

          boxes = np.append( boxes, obj_box, axis = 0 )

          # removes synonynms
          if self._integer_labels:
            labels = np.append( labels, categories.get_class_id( obj_id ) )
          else:
            labels = np.append( labels, categories.get_class_name( obj_id ) )

      annotations["bboxes"] = boxes.astype( np.float32 )
      annotations["labels"] = labels.astype( np.int_ )

      entry["filename"] = filename
      entry["width"] = width
      entry["height"] = height
      entry["ann"] = annotations

      self._training_data.append( entry )

  def update_model( self ):

    with open( self._groundtruth_store, 'wb' ) as fp:
      pickle.dump( self._training_data, fp )

    train_dataset = CustomDataset(
      self._groundtruth_store,
      '.',
      self._cfg.data.train.img_scale,
      self._cfg.data.train.img_norm_cfg,
      size_divisor = self._cfg.data.train.size_divisor,
      flip_ratio = self._cfg.data.train.flip_ratio,
      with_mask = self._cfg.data.train.with_mask,
      with_crowd = self._cfg.data.train.with_crowd,
      with_label = self._cfg.data.train.with_label )

    train_detector(
      self._model,
      train_dataset,
      self._cfg,
      distributed = self._distributed,
      validate = self._validate,
      logger = self._logger )

def __vital_algorithm_register__():
  from vital.algo import algorithm_factory

  # Register Algorithm
  implementation_name  = "mmdet"

  if algorithm_factory.has_algorithm_impl_name(
      MMTrainDetector.static_type_name(), implementation_name ):
    return

  algorithm_factory.add_algorithm( implementation_name,
    "PyTorch MMDetection training routine", MMTrainDetector )

  algorithm_factory.mark_algorithm_as_loaded( implementation_name )
