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

from vital.algo import TrainDetector

from vital.types import BoundingBox
from vital.types import CategoryHierarchy
from vital.types import DetectedObjectSet
from vital.types import DetectedObject

class MMTrainDetector( TrainDetector ):
  """
  Implementation of TrainDetector class
  """
  def __init__( self ):
    TrainDetector.__init__(self)
    self._model_file = ""

  def get_configuration(self):
    # Inherit from the base class
    cfg = super(TrainDetector, self).get_configuration()
    cfg.set_value( "model_file", self._model_file )
    return cfg

  def set_configuration( self, cfg_in ):
    cfg = self.get_configuration()
    cfg.merge_config( cfg_in )
    self._model_file = str( cfg.get_value( "model_file" ) )

  def check_configuration( self, cfg ):
    if not cfg.has_value("center_x") or len(cfg.get_value( "model_file")) == 0:
      print( "A model file must be specified!" )
      return False
    return True

  def add_data_from_disk( categories, train_files, train_dets, test_files, test_dets ):
    print( "!! " + str( categories.all_class_names() ) )

  def update_model(self):
    return

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
