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

from vital.algo import ImageObjectDetector

from vital.types import BoundingBox
from vital.types import DetectedObjectSet
from vital.types import DetectedObject

import numpy as np

import mmcv

from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import inference_detector

class MMTestDetector( ImageObjectDetector ):
  """
  Implementation of ImageObjectDetector class
  """
  def __init__( self ):
    ImageObjectDetector.__init__(self)
    self._net_config = ""
    self._weight_file = ""
    self._class_names = ""
    self._thresh = 0.01
    self._gpu_index = "0"

  def get_configuration(self):
    # Inherit from the base class
    cfg = super(ImageObjectDetector, self).get_configuration()
    cfg.set_value( "net_config", self._net_config )
    cfg.set_value( "weight_file", self._weight_file )
    cfg.set_value( "class_names", self._class_names )
    cfg.set_value( "thresh", str( self._thresh ) )
    cfg.set_value( "gpu_index", self._gpu_index )
    return cfg

  def set_configuration( self, cfg_in ):
    cfg = self.get_configuration()
    cfg.merge_config( cfg_in )

    self._net_config = str( cfg.get_value( "net_config" ) )
    self._weight_file = str( cfg.get_value( "weight_file" ) )
    self._class_names = str( cfg.get_value( "class_names" ) )
    self._thresh = float( cfg.get_value( "thresh" ) )
    self._gpu_index = str( cfg.get_value( "gpu_index" ) )

    self._cfg = mmcv.Config.fromfile( self._net_config )
    self._cfg.model.pretrained = None
    self._model = build_detector( self._cfg.model, test_cfg=self._cfg.test_cfg )
    _ = load_checkpoint( self._model, self._weight_file )

  def check_configuration( self, cfg ):
    if not cfg.has_value( "net_config" ) or len( cfg.get_value( "net_config") ) == 0:
      print( "A network config file must be specified!" )
      return False

    return True

  def detect( self, image_data ):
    #input_image = image_data.asarray().astype( 'uint8' )[...,::-1]
    input_image = image_data.asarray().astype( 'uint8' )
    print( np.shape( input_image ) )

    gpu_string = 'cuda:' + str( self._gpu_index )
    detections = inference_detector( self._model, input_image, self._cfg, device=gpu_string )

    class_names = [ 'fish' ] * 10000

    if isinstance( detections, tuple ):
      bbox_result, segm_result = detections
    else:
      bbox_result, segm_result = detections, None

    bboxes = np.vstack( bbox_result )

    # convert segmentation masks
    masks = []
    if segm_result is not None:
      segms = mmcv.concat_list( segm_result )
      inds = np.where( bboxes[:, -1] > score_thr )[0]
      for i in inds:
        masks.append( maskUtils.decode( segms[i] ).astype( np.bool ) )

    # collect labels
    labels = [
        np.full( bbox.shape[0], i, dtype=np.int32 )
        for i, bbox in enumerate( bbox_result )
    ]

    labels = np.concatenate( labels )

    # convert to kwiver format, apply threshold
    output = []

    for entry in []:
       output.append( DetectedObject( BoundingBox( 1,1,2,2 ) ) )

    mmcv.imshow_det_bboxes(
        input_image,
        bboxes,
        labels,
        class_names=class_names,
        score_thr=-100.0,
        show=True)

    return DetectedObjectSet( output )

def __vital_algorithm_register__():
  from vital.algo import algorithm_factory

  # Register Algorithm
  implementation_name  = "mmdet"

  if algorithm_factory.has_algorithm_impl_name(
      MMTestDetector.static_type_name(), implementation_name ):
    return

  algorithm_factory.add_algorithm( implementation_name,
    "PyTorch MMDetection testing routine", MMTestDetector )

  algorithm_factory.mark_algorithm_as_loaded( implementation_name )
