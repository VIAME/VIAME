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
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import torch
import torch.utils.data as data
import torch.nn as nn
from torchvision import models, transforms, datasets

from PIL import Image as pilImage

from vital.types import BoundingBox

class ResnetDataLoader(data.Dataset):
    def __init__(self, bbox_list, transform, frame_img, in_size, MOT_flag):
        self._frame_img = pilImage.new("RGB", frame_img.size)
        self._frame_img.paste( frame_img )
        self._transform = transform
        self._bbox_list = bbox_list
        self._mot_flag = MOT_flag
        self._in_size = in_size

    def __getitem__(self, index):
        bb = self._bbox_list[index] if self.mot_flag else self._bbox_list[index].bounding_box()

        # unwrap
        min_x, min_y, max_x, max_y = map(float, ( bb.min_x(), bb.min_y(),  \
                                                    bb.max_x(), bb.max_y() )

        c_x = ( min_x + max_x ) / 2
        c_y = ( min_y + max_y ) / 2

        padding = 1.1 * max( max_x - min_x, max_y - min_y ) / 2

        # crop a square image
        im = self._frame_img.crop(
            (
                c_x - padding,
                c_y - padding,
                c_x + padding,
                c_y + padding
            )
        )

        im = im.resize((self._in_size, self._in_size), pilImage.BILINEAR) # this should be >197
        im.convert('RGB') # probably unneeded

        if self._transform is not None:
            im = self._transform(im)

        return im

    def __len__(self):
        return len(self._bbox_list) if self._mot_flag else self._bbox_list.size()
    

class ResnetFeatureExtractor(object):
    """
    Obtain the appearance features from a trained pytorch resnet50
    model
    """

    def __init__(self, resnet_model_path, img_size, batch_size, gpu_list=None):
        if gpu_list is None:
            gpu_list = [x for x in range(torch.cuda.device_count())]
            target_gpu = 0 # I assume this is just hardcoding in using the first GPU
        else:
            target_gpu = gpu_list[0]

        self._device = torch.device("cuda:{}".format(self._target_gpu))

        # load the resnet50 model. Maybe this shouldn't be hardcoded?
        self._resnet_model = models.resnet50().to(self._device)
        # changing the number of output layers, to allow for loading the model
        # might not be necessary 
        num_ftrs = self._resnet_model.fc.in_features
        self._resnet_model.fc = nn.Linear(num_ftrs, 46)

        #snapshot = torch.load(resnet_model_path)
        #self._resnet_model.load_state_dict(snapshot['state_dict']) # the snapshot is saved as a dict. state_dict is the key for model weights
        #print('Model loaded from {}'.format(resnet_model_path))
        #self._resnet_model = torch.nn.DataParallel(self._resnet_model, device_ids=GPU_list)

        print( resnet_model_path )
        weights = torch.load( resnet_model_path )['state_dict']
        self._resnet_model.load_state_dict( weights )
        self._resnet_model = nn.Sequential(*list(self._resnet_model.children())[:-1])

        self._resnet_model.train( False ) # is this the same as eval() ?
        self._resnet_model.cuda() # move the model to the GPU
 
        self._transform = transforms.Compose([
            transforms.Scale(img_size),# I'm not sure I like having this in here but I guess it makes it more general. Maybe we just pass in square images and let this resize.
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self._img_size = img_size
        self._b_size = batch_size

    def __call__(self, bbox_list, mot_flag):
        return self._obtain_feature(bbox_list, MOT_flag)

    def _obtain_feature(self, bbox_list, mot_flag):
        kwargs = {'num_workers': 0, 'pin_memory': True}
        if self.frame is not None:
            bbox_loader_class = resnetDataLoader(bbox_list, self._transform, self.frame, self._img_size, MOT_flag) 
        else:
            raise ValueError("Trying to create a resnetDataLoader without input frame")
        bbox_loader = torch.utils.data.DataLoader(bbox_loader_class, batch_size=self._b_size, shuffle=False, **kwargs)
        for idx, imgs in enumerate(bbox_loader):
            v_imgs = imgs.to(self._device)
            output = self._resnet_model(v_imgs)
            if idx == 0:
                app_features = output.data
            else:
                app_features = torch.cat((app_features, output.data), dim=0)
        return app_features.cpu()
