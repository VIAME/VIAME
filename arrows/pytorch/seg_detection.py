# ckwg +28
# Copyright 2017 by Kitware, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
#  * Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
#  * Neither name of Kitware, Inc. nor the names of any contributors may be used
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
from __future__ import absolute_import

import torch
from torchvision import models, transforms
from torch.autograd import Variable
from torch import nn
import numpy as np
import scipy as sp
#import cv2

from PIL import Image as pilImage

from sprokit.pipeline import process
from kwiver.kwiver_process import KwiverProcess
from vital.types import Image
from vital.types import DetectedObject
from vital.types import DetectedObjectSet

from kwiver.arrows.pytorch.fcn_segmentation import FCN_Segmentation
from kwiver.arrows.pytorch.fcn_models import FCN16s


class seg_detection(KwiverProcess):

    # ----------------------------------------------
    def __init__(self, conf):
        KwiverProcess.__init__(self, conf)

        self.add_config_trait("fcn_model_path", "fcn_model_path",
                              '/home/bdong/HiDive_project/pytorch-fcn/trained_model/model_best.pth.tar',
                              'Trained PyTorch fcn model.')
        self.declare_config_using_trait('fcn_model_path')

        self.add_config_trait("fcn_flag", "fcn_flag", 'True', 'define whether use FCN based or contour based detection')
        self.declare_config_using_trait('fcn_flag')
    

        # set up required flags
        optional = process.PortFlags()
        required = process.PortFlags()
        required.add(self.flag_required)

        #  input port ( port-name,flags)
        self.declare_input_port_using_trait('image', required)

        #  output port ( port-name,flags)
        self.declare_output_port_using_trait('detected_object_set', optional)

    # ----------------------------------------------
    def _configure(self):

        fcn_flag = self.config_value('fcn_flag')
        self._fcn_flag = True if fcn_flag == 'True' else False 

        fcn_seg_model_path = self.config_value('fcn_model_path')

        fcn_model = FCN16s(n_class=21).cuda()
        checkpoint = torch.load(fcn_seg_model_path)
        fcn_model.load_state_dict(checkpoint['model_state_dict'])

        self._fcn_seg = FCN_Segmentation(fcn_model)
        self._frame_id = 0

        self._base_configure()

    # ----------------------------------------------
    def _step(self):

        # grab image container from port using traits
        in_img_c = self.grab_input_using_trait('image')

        # Get current frame and give it to app feature extractor
        im = in_img_c.get_image().get_pil_image()
        im = np.array(im, dtype=np.uint8)
        
        dos, cv_pred, lbl_pred_overlap = self._fcn_seg(im, fcn_flag=self._fcn_flag)
        #cv2.imwrite('det_frame_{}.jpg'.format(self._frame_id), cv_pred)
        #if lbl_pred_overlap is not None:
            #cv2.imwrite('lbl_frame_{}.jpg'.format(self._frame_id), lbl_pred_overlap)

        print('dos length {}'.format(dos.size()))
        self._frame_id += 1

        self.push_to_port_using_trait('detected_object_set', dos)
        self._base_step()


# ==================================================================
def __sprokit_register__():
    from sprokit.pipeline import process_factory

    module_name = 'python:kwiver.seg_detection'

    if process_factory.is_process_module_loaded(module_name):
        return

    process_factory.add_process('seg_detection', 'segmentation based detection',
                                seg_detection)

    process_factory.mark_process_module_as_loaded(module_name)

