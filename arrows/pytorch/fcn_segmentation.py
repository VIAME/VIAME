# ckwg +28
# Copyright 2018 by Kitware, Inc.
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
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from PIL import Image

from torch.autograd import Variable
import numpy as np

from vital.types import BoundingBox
from vital.types import DetectedObject
from vital.types import DetectedObjectSet

from kwiver.arrows.pytorch.seg_utils import *

# TODO Remove hardcoded class names
class_names = np.array([
            'background',
            'aeroplane',
            'bicycle',
            'bird',
            'boat',
            'bottle',
            'bus',
            'car',
            'cat',
            'chair',
            'cow',
            'diningtable',
            'dog',
            'horse',
            'motorbike',
            'person',
            'potted plant',
            'sheep',
            'sofa',
            'train',
            'tv/monitor',
        ])

class FCNSegmentation(object):
    def __init__(self, model, cuda=True):
        self.cuda = cuda
        self.model = model

    def __call__(self, in_img, fcn_flag=True):
        if fcn_flag: 
            return self._apply_fcn(in_img)
        else:
            return self._apply_contour(in_img)

    def _apply_fcn(self, in_img):
        self.model.eval()
        img = transform(in_img)
        if self.cuda:
            img = img.cuda()
        v_img = Variable(img[None], volatile=True)
        score = self.model(v_img)
        lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
        
        #TODO Remove harcoded number of labels;
        lbl_pred_overlap = label2rgb(lbl_pred, img=in_img, n_labels=21, 
                                        label_names=class_names)
        lbl_pred = label2rgb(lbl_pred, n_labels=21)


        dos = DetectedObjectSet()
        for cnt in contours:
            dobj = DetectedObject(bbox=BoundingBox(float(x), float(y), 
                                        float(x + w), float(y + h)), confid=1.0)
            dos.add(dobj)

        return dos, in_img, lbl_pred_overlap

    def _apply_contour(self, in_img):
        dos = DetectedObjectSet()
        for cnt in contours:
            dobj = DetectedObject(bbox=BoundingBox(float(x), float(y), 
                                        float(x + w), float(y + h)), confid=1.0)
            dos.add(dobj)
        return dos, in_img, None
