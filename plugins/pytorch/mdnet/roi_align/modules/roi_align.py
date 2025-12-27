# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

from torch.nn.modules.module import Module
from ..functions.roi_align import RoIAlignFunction, RoIAlignAdaFunction
from torch.nn.functional import avg_pool2d, max_pool2d

class RoIAlign(Module):

    def __init__(self, aligned_height, aligned_width, spatial_scale, sample_num=0):
        super(RoIAlign, self).__init__()
        self.aligned_height = int(aligned_height)
        self.aligned_width = int(aligned_width)
        # self.out_size = (self.aligned_height, self.aligned_width)
        self.spatial_scale = float(spatial_scale)
        self.sample_num = int(sample_num)

    def forward(self, features, rois):
        return RoIAlignFunction.apply(features, rois,
          (int(self.aligned_height), int(self.aligned_width)),
          self.spatial_scale, self.sample_num)

class RoIAlignAvg(Module):

    def __init__(self, aligned_height, aligned_width, spatial_scale, sample_num=0):
        super(RoIAlignAvg, self).__init__()
        self.aligned_height = int(aligned_height)
        self.aligned_width = int(aligned_width)
        self.out_size = (self.aligned_height, self.aligned_width)
        self.spatial_scale = float(spatial_scale)
        self.sample_num = int(sample_num)
        self.out_size_2 = (aligned_height + 1, aligned_width + 1)

    def forward(self, features, rois):
        pooled = RoIAlignFunction.apply(features, rois,
          (int(self.aligned_height) + 1, int(self.aligned_width) + 1),
          self.spatial_scale, self.sample_num)
        return avg_pool2d(pooled, kernel_size=2, stride=1)

class RoIAlignMax(Module):

    def __init__(self, aligned_height, aligned_width, spatial_scale, sample_num=0):
        super(RoIAlignMax, self).__init__()
        self.aligned_height = int(aligned_height)
        self.aligned_width = int(aligned_width)
        self.out_size = (self.aligned_height, self.aligned_width)
        self.spatial_scale = float(spatial_scale)
        self.sample_num = int(sample_num)
        self.out_size_2 = (aligned_height + 4, aligned_width + 4)

    def forward(self, features, rois):
        pooled = RoIAlignFunction.apply(features, rois,
          (int(self.aligned_height) + 4, int(self.aligned_width) + 4),
          self.spatial_scale, self.sample_num)
        return max_pool2d(pooled, kernel_size=3, stride=2)

class RoIAlignAdaMax(Module):

    def __init__(self, aligned_height, aligned_width, spatial_scale, sample_num=0):
        super(RoIAlignAdaMax, self).__init__()
        self.aligned_height = int(aligned_height)
        self.aligned_width = int(aligned_width)
        self.out_size = (self.aligned_height, self.aligned_width)
        self.spatial_scale = float(spatial_scale)
        self.sample_num = int(sample_num)
        self.out_size_2 = (aligned_height + 4, aligned_width + 4)

    def forward(self, features, rois):
        pooled = RoIAlignAdaFunction.apply(features, rois,
          (int(self.aligned_height) + 4, int(self.aligned_width) + 4),
          self.spatial_scale, self.sample_num)
        return max_pool2d(pooled, kernel_size=3, stride=2)


class RoIAlignAda(Module):

    def __init__(self, aligned_height, aligned_width, spatial_scale, sample_num=0):
        super(RoIAlignAda, self).__init__()

        self.out_size = (aligned_height, aligned_width)
        self.spatial_scale = float(spatial_scale)
        self.sample_num = int(sample_num)

    def forward(self, features, rois):
        return RoIAlignAdaFunction.apply(features, rois,
          (int(self.aligned_height), int(self.aligned_width)),
          self.spatial_scale, self.sample_num)
