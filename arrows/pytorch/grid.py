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
import numpy as np
import torch


class Grid(object):
    def __init__(self, grid_row=15, grid_cols=15, target_neighborhood_w=7):
        self._grid_rows = grid_row
        self._grid_cols = grid_cols
        self._target_neighborhood_w = target_neighborhood_w
        self._half_cell_w = int(self._target_neighborhood_w // 2)

    def __call__(self, im_size, bbox_list, mot_flag=False):
        return self.obtain_grid_feature_list(im_size, bbox_list, mot_flag)

    def obtain_grid_feature_list(self, im_size, bbox_list, mot_flag):
        r"""
            The output of the function is a grid feature list for
            each corresponding bbox of current frame/image

            A grid feature records which cells in the configured
            neighborhood have at least one bonuding box in them.
        """
        self.img_w, self.img_h = im_size

        # calculate grid cell height and width
        cell_h = self.img_h / self._grid_rows
        cell_w = self.img_w / self._grid_cols

        # initial all gridcell to 0
        grid = torch.FloatTensor(self._grid_rows, self._grid_cols).zero_()

        bbox_id_centerIDX = []
        # build the grid for current image
        for item in bbox_list:
            bb = item if mot_flag else item.bounding_box()

            x = int(bb.min_x())
            y = int(bb.min_y())
            w = int(bb.width())
            h = int(bb.height())

            # bbox center
            c_w = min(x + w / 2, self.img_w - 1)
            c_h = min(y + h / 2, self.img_h - 1)

            # cell idxs
            row_idx = int(c_h // cell_h)
            col_idx = int(c_w // cell_w)

            bbox_id_centerIDX.append((row_idx, col_idx))

            # Assertion for corner cases
            assert row_idx < grid.shape[0]
            assert col_idx < grid.shape[1]
            grid[row_idx, col_idx] = 1

        grid_feature_list = []
        # obtain grid feature for each bbox
        for row_idx, col_idx in bbox_id_centerIDX:
            # top left corner's the neighborhood grid
            neighborhood_grid_top = row_idx - self._half_cell_w
            neighborhood_grid_left = col_idx - self._half_cell_w

            neighborhood_grid = torch.FloatTensor(self._target_neighborhood_w,
                                            self._target_neighborhood_w).zero_()

            for r in range(self._target_neighborhood_w):
                for c in range(self._target_neighborhood_w):
                    if (0 <= neighborhood_grid_top + r < grid.size(0)
                        and 0 <= neighborhood_grid_left + c < grid.size(1)):
                        neighborhood_grid[r, c] = grid[neighborhood_grid_top + r,
                                                    neighborhood_grid_left + c]

            grid_feature_list.append(neighborhood_grid.view(neighborhood_grid.numel()))

        return grid_feature_list
