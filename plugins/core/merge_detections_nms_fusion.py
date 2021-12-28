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

from kwiver.vital.algo import MergeDetections

from kwiver.vital.types import BoundingBoxD
from kwiver.vital.types import DetectedObjectSet
from kwiver.vital.types import DetectedObject
from kwiver.vital.types import DetectedObjectType

import numpy as np

class MergeDetectionsNMSFusion( MergeDetections ):
  """
  Implementation of MergeDetections class
  """
  def __init__( self ):
    MergeDetections.__init__( self )

  def get_configuration( self ):
    cfg = super( MergeDetections, self ).get_configuration()
    return cfg

  def set_configuration( self, cfg_in ):
    return

  def check_configuration( self, cfg ):
    return True

  def merge( self, sets ):

    outputs = []

    for dets in sets:
        []

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
