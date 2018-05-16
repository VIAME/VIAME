import numpy as np
import torch


class grid(object):
    def __init__(self, grid_row=15, grid_cols=15, target_neighborhood_w=7):
        self._grid_rows = grid_row
        self._grid_cols = grid_cols
        self._target_neighborhood_w = target_neighborhood_w
        self._half_cell_w = int(self._target_neighborhood_w // 2)

    def __call__(self, im_size ,bbox_list, MOT_flag=False):
        return self.obtainGridFeatureList(im_size, bbox_list, MOT_flag)

    @property
    def img_h(self):
        return self._img_h

    @img_h.setter
    def img_h(self, val):
        self._img_h = val

    @property
    def img_w(self):
        return self._img_w

    @img_w.setter
    def img_w(self, val):
        self._img_w = val
    
    def obtainGridFeatureList(self, im_size, bbox_list, MOT_flag):
        r"""
            The output of the function is a grid feature list for each corresponding bbox of current frame/image
        """
        self._img_w, self._img_h = im_size

        # calculate grid cell height and width
        cell_h = self._img_h / self._grid_rows
        cell_w = self._img_w / self._grid_cols

        # initial all gridcell to 0
        grid = torch.FloatTensor(self._grid_rows, self._grid_cols).zero_()

        bbox_id_centerIDX = {}
        # build the grid for current image
        for idx, item in enumerate(bbox_list):
            if MOT_flag is True:
                bb = item
            else:
                bb = item.bounding_box()

            x, y, w, h = int(bb.min_x()), int(bb.min_y()), int(bb.width()), int(bb.height())

            # bbox center
            c_w = min(x + w / 2, self._img_w - 1)
            c_h = min(y + h / 2, self._img_h - 1)

            # cell idxs
            row_idx = int(c_h // cell_h)
            col_idx = int(c_w // cell_w)

            bbox_id_centerIDX[idx] = tuple((row_idx, col_idx))

            try:
                grid[row_idx, col_idx] = 1
            except IndexError:
                print('c_h:{}, c_w:{}, row_idx:{}, col_idx:{}'.format(c_h, c_w, row_idx, col_idx))

        grid_feature_list = []
        # obtain grid feature for each bbox
        for idx in range(len(bbox_id_centerIDX)):
            # top left corner's the neighborhood grid
            neighborhood_grid_top = bbox_id_centerIDX[idx][0] - self._half_cell_w
            neighborhood_grid_left = bbox_id_centerIDX[idx][1] - self._half_cell_w

            neighborhood_grid = torch.FloatTensor(self._target_neighborhood_w, self._target_neighborhood_w).zero_()

            for r in range(self._target_neighborhood_w):
                for c in range(self._target_neighborhood_w):
                    if 0 <= neighborhood_grid_top + r < grid.size(0) and 0 <= neighborhood_grid_left + c < grid.size(
                        1):
                        neighborhood_grid[r, c] = grid[neighborhood_grid_top + r, neighborhood_grid_left + c]

            grid_feature_list.append(neighborhood_grid.view(neighborhood_grid.numel()))

        return grid_feature_list

