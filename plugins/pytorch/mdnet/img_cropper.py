import sys
sys.path.insert(0,'./modules')
from roi_align.modules.roi_align import RoIAlign
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import numpy as np

import time

import matplotlib.pyplot as plt
import matplotlib.patches as patches

class imgCropper(nn.Module):
    def __init__(self, img_size):
        super(imgCropper, self).__init__()
        self.isCuda = False
        self.img_size = img_size
        self.roi_align_model = RoIAlign(img_size,img_size, 1. )

    def gpuEnable(self):
        self.roi_align_model = self.roi_align_model.cuda()
        self.isCuda = True

    def forward(self, image, roi):
        aligned_image_var = self.roi_align_model(image, roi)
        return aligned_image_var

    def crop_image(self,image, box, result_size):
        ## constraint = several box from common 1 image
        ishape = image.shape
        cur_image_var = np.reshape(image, (1, ishape[0], ishape[1], ishape[2]))
        cur_image_var = cur_image_var.transpose(0, 3, 1, 2)
        cur_image_var = cur_image_var.astype('float32')
        cur_image_var = Variable(torch.from_numpy(cur_image_var).float())


        roi = np.copy(box)
        roi[:,2:4] += roi[:,0:2]
        roi = np.concatenate((np.zeros((roi.shape[0], 1)), roi), axis=1)
        roi = Variable(torch.from_numpy(roi).float())

        if self.isCuda:
            cur_image_var = cur_image_var.cuda()
            roi = roi.cuda()

        self.roi_align_model.aligned_width = result_size[0]
        self.roi_align_model.aligned_height = result_size[1]
        cropped_image = self.forward(cur_image_var, roi)

        return cropped_image, cur_image_var

    def crop_several_image(self,img_list,target_list):
        ## constraint = one to one matching between image and target
        ## exception handling
        assert(len(target_list) == len(img_list))

        ## image crop
        torch.cuda.synchronize()
        start_time = time.time()
        cur_images = torch.squeeze(torch.stack(img_list, 0))
        torch.cuda.synchronize()
        print '10 image stacking time:{}'.format(time.time() - start_time)

        ishape = cur_images.size()

        # Extract sample features and get target location
        sample_rois = np.array(target_list)
        sample_rois[:,2:4] += sample_rois[:,0:2]
        batch_num = np.reshape(np.arange(0,len(sample_rois)),(len(sample_rois),1))
        sample_rois = np.concatenate( (batch_num, sample_rois), axis=1)
        sample_rois = Variable(torch.from_numpy(sample_rois.astype('float32')))
        if self.isCuda:
            sample_rois = sample_rois.cuda()
            cur_images = cur_images.cuda()

        cropped_images = self.forward(cur_images, sample_rois)


        return cropped_images



