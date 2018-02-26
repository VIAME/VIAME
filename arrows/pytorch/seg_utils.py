from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import six
import scipy.misc
import scipy.ndimage
import skimage.color
import warnings
import numpy as np

import torch

def bitget(byteval, idx):
    return ((byteval & (1 << idx)) != 0)


def label_colormap(N=256):
    cmap = np.zeros((N, 3))
    for i in six.moves.range(0, N):
        id = i
        r, g, b = 0, 0, 0
        for j in six.moves.range(0, 8):
            r = np.bitwise_or(r, (bitget(id, 0) << 7 - j))
            g = np.bitwise_or(g, (bitget(id, 1) << 7 - j))
            b = np.bitwise_or(b, (bitget(id, 2) << 7 - j))
            id = (id >> 3)

        cmap[i, 0] = r
        cmap[i, 1] = g
        cmap[i, 2] = b
    cmap = cmap.astype(np.float32) / 255
    cmap = (cmap * 255).astype(np.uint8)
    return cmap

def transform(img):
    mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
    img = img[:, :, ::-1]  # RGB -> BGR
    img = img.astype(np.float64)
    img -= mean_bgr
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).float()
    return img
